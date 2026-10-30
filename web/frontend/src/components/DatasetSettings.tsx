import { useCallback, useEffect, useRef, useState } from 'react'
import { Database, HardDrive, Info, Save } from 'lucide-react'
import { api, formatBytes } from '../api'
import type { DatasetPolicy, DatasetSettingsResponse } from '../types'
import { DatasetStorageSettings } from './DatasetStorage'
import { DatasetOperationProgress } from './DatasetOperationProgress'
import '../datasetSettings.css'

export interface DatasetSettingsStatus {
  loading: boolean
  checkedAt: Date | null
  locationCheckedAt: Date | null
  error: string
}

type PolicyDraft = Record<keyof DatasetPolicy, string>
type LocationStatus = { loading: boolean; checkedAt: Date | null; error: string }
const fields: Array<{ key: keyof DatasetPolicy; label: string; unit: string; max: number; help: string }> = [
  { key: 'storage_quota_bytes', label: '平台中转配额', unit: '字节', max: Number.MAX_SAFE_INTEGER, help: '按平台中转归档占用及执行前空间估算检查配额；0 表示不设配额，仍检查实际磁盘空间。调低配额可能使进行中传输的后续写入因配额不足而失败。' },
  { key: 'failed_retention_seconds', label: '失败中转快照保留', unit: '秒', max: 3650 * 86400, help: '到期后允许回收符合条件的失败快照；0 表示不自动清理。失败上传暂存保留供重试，取消后按上传暂存规则回收。缩短保留期可能影响已有失败操作的直接重试。' },
  { key: 'refresh_read_bytes_per_second', label: '训练期间重新加载读取上限', unit: '字节 / 秒', max: 1024 ** 4, help: '0 表示不按字节限速。适用于本机重新加载和平台中心快照校验；远程源目录导出由 Agent 自身策略控制。' },
  { key: 'refresh_io_pressure_pct', label: '训练期间磁盘压力暂停阈值', unit: '%', max: 100, help: '达到 Linux I/O 压力阈值时暂停相关读取；0 关闭此项。系统不支持压力统计时仍执行其他资源限制。' },
  { key: 'refresh_min_free_bytes', label: '重新加载最低可用空间', unit: '字节', max: Number.MAX_SAFE_INTEGER, help: '可用空间低于此值时暂停相关读取；0 关闭此项阈值，打包与校验仍检查实际所需空间。' },
]
const kinds: Record<string, string> = { upload: '上传', register: '登记', refresh: '重新加载', migrate: '复制或迁移', check: '目录检查', discover: '目录发现', delete: '文件删除', unregister: '删除登记' }
const cleanupLabels: Record<string, string> = { active: '处理中', reserved: '等待释放', pending: '等待清理', retained: '按规则保留', orphan: '待回收临时文件' }

function toDraft(policy: DatasetPolicy): PolicyDraft {
  return Object.fromEntries(fields.map(({ key }) => [key, String(policy[key])])) as PolicyDraft
}
function dateLabel(value: string | null | undefined) {
  if (!value) return '尚无记录'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}
function duration(value: number) {
  if (value === 0) return '不自动清理'
  if (value % 86400 === 0) return `${value / 86400} 天`
  if (value % 3600 === 0) return `${value / 3600} 小时`
  return `${value.toLocaleString()} 秒`
}
function size(value: number | null) { return value == null ? '待统计' : formatBytes(value) }
function policyHint(key: keyof DatasetPolicy, value: string) {
  const amount = Number(value)
  if (!value || !Number.isSafeInteger(amount) || amount < 0) return ''
  if (key === 'failed_retention_seconds') return duration(amount)
  if (key === 'refresh_io_pressure_pct') return amount ? `${amount}%` : '已关闭'
  if (!amount) return key === 'storage_quota_bytes' ? '不设配额' : key === 'refresh_read_bytes_per_second' ? '不限速' : '已关闭此项阈值'
  return `${formatBytes(amount)}${key === 'refresh_read_bytes_per_second' ? ' / 秒' : ''}`
}

export function DatasetSettings({ refreshToken = 0, onStatusChange }: { refreshToken?: number; onStatusChange?: (status: DatasetSettingsStatus) => void }) {
  const [data, setData] = useState<DatasetSettingsResponse | null>(null)
  const [draft, setDraft] = useState<PolicyDraft | null>(null)
  const [baseRevision, setBaseRevision] = useState<number | null>(null)
  const [dirty, setDirty] = useState(false)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [saveError, setSaveError] = useState('')
  const [notice, setNotice] = useState('')
  const [checkedAt, setCheckedAt] = useState<Date | null>(null)
  const [locationStatus, setLocationStatus] = useState<LocationStatus>({ loading: true, checkedAt: null, error: '' })
  const dirtyRef = useRef(false)
  const savingRef = useRef(false)
  const controllerRef = useRef<AbortController | null>(null)
  const saveControllerRef = useRef<AbortController | null>(null)
  const dataRef = useRef<DatasetSettingsResponse | null>(null)
  const previousRefreshToken = useRef(refreshToken)
  const onStatusChangeRef = useRef(onStatusChange)
  useEffect(() => { onStatusChangeRef.current = onStatusChange }, [onStatusChange])
  useEffect(() => {
    onStatusChangeRef.current?.({ loading: loading || saving || locationStatus.loading, checkedAt, locationCheckedAt: locationStatus.checkedAt, error: error || locationStatus.error })
  }, [loading, saving, locationStatus, checkedAt, error])

  const acceptData = useCallback((next: DatasetSettingsResponse) => {
    dataRef.current = next
    setData(next)
    setCheckedAt(new Date())
    if (!dirtyRef.current) { setDraft(toDraft(next.policy)); setBaseRevision(next.revision) }
  }, [])

  useEffect(() => {
    let active = true
    let running = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const manualRefresh = previousRefreshToken.current !== refreshToken
    previousRefreshToken.current = refreshToken
    async function load(refresh = false) {
      if (!active || running || document.hidden) return
      if (savingRef.current) { timer = setTimeout(() => void load(refresh), 1000); return }
      running = true
      const controller = new AbortController()
      controllerRef.current = controller
      setLoading(true)
      try {
        const next = await api.datasetSettings(controller.signal, refresh)
        if (!active || controller.signal.aborted) return
        acceptData(next); setError('')
      } catch (reason) {
        if (active && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : '数据集设置读取失败')
      } finally {
        running = false
        if (active && !controller.signal.aborted) setLoading(false)
        if (active) {
          const next = dataRef.current
          timer = setTimeout(() => void load(), next?.storage.sampling || next?.storage.status === 'pending' || (next?.operations.active_count || 0) > 0 ? 5000 : 30_000)
        }
      }
    }
    function visible() { if (!document.hidden) { clearTimeout(timer); void load() } }
    void load(manualRefresh)
    document.addEventListener('visibilitychange', visible)
    return () => { active = false; controllerRef.current?.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible) }
  }, [acceptData, refreshToken])
  useEffect(() => () => saveControllerRef.current?.abort(), [])

  function edit(key: keyof DatasetPolicy, value: string) {
    if (!draft) return
    dirtyRef.current = true; setDirty(true); setNotice(''); setSaveError('')
    setDraft({ ...draft, [key]: value })
  }
  function resetToServer() {
    if (!data) return
    dirtyRef.current = false; setDirty(false); setDraft(toDraft(data.policy)); setBaseRevision(data.revision); setSaveError(''); setNotice('已载入服务器当前设置')
  }
  async function save(event: React.FormEvent) {
    event.preventDefault()
    if (!draft || !data || baseRevision == null || savingRef.current || data.revision !== baseRevision) return
    const policy = {} as DatasetPolicy
    for (const field of fields) {
      const value = Number(draft[field.key])
      if (!/^\d+$/.test(draft[field.key]) || !Number.isSafeInteger(value) || value < 0 || value > field.max) {
        setSaveError(`${field.label}需为 0 至 ${field.max.toLocaleString()} 的整数`); return
      }
      policy[field.key] = value
    }
    savingRef.current = true; setSaving(true); setSaveError(''); setNotice('')
    controllerRef.current?.abort(); setLoading(false)
    const controller = new AbortController(); saveControllerRef.current = controller
    try {
      const next = await api.updateDatasetSettings(baseRevision, policy, controller.signal)
      if (controller.signal.aborted) return
      dirtyRef.current = false; setDirty(false); acceptData(next); setError('')
      setNotice('设置已保存；平台后续资源检查使用新值。正在执行的操作不会因此重启；调低配额可能影响后续写入，资源限制可能使相关读取暂停。')
    } catch (reason) {
      if (controller.signal.aborted) return
      const message = reason instanceof Error ? reason.message : '设置保存失败'
      if ((reason as { status?: number }).status === 409) {
        setSaveError(`设置已被其他操作更新，草稿已保留。请先载入服务器当前设置再修改。${message}`)
        try { const next = await api.datasetSettings(controller.signal); if (!controller.signal.aborted) acceptData(next) } catch { /* Keep the draft and the last successful response. */ }
      } else setSaveError(message)
    } finally { if (!controller.signal.aborted) { savingRef.current = false; setSaving(false) } }
  }

  const storage = data?.storage
  const outdated = !!data && !!draft && data.revision !== baseRevision
  const partial = storage?.status === 'partial'
  const workerError = data?.worker.active_error || data?.worker.refresh_active_error
  return <div className="dataset-settings" aria-busy={loading || saving}>
    <p className="dataset-settings-intro"><Info size={15} />训练期间可查看和配置；打开本页不会扫描原始数据集。目录检查与后台处理仍按路径锁和资源限制执行。</p>
    <DatasetStorageSettings refreshToken={refreshToken} onStatusChange={setLocationStatus} />
    {error && <div className="alert error" role="alert">刷新失败：{error}{data ? '；以下保留最近一次成功读取的数据。' : ''}</div>}
    {!data && <div className="settings-empty-state" role="status">{loading ? '正在读取数据集设置…' : '尚未成功读取数据集设置，请点击下方“重新检查”。'}</div>}
    {data && storage && <>
      {data.policy_error && <div className="alert error" role="alert">实例设置读取异常：{data.policy_error}。以下显示服务器实际使用的设置。</div>}
      <section className="dataset-settings-section" aria-labelledby="dataset-space-title">
        <div className="dataset-settings-heading"><h4 id="dataset-space-title"><Database size={15} />原始数据集与平台临时空间</h4></div>
        <div className="dataset-settings-stats">
          {(['local', 'remote'] as const).map(scope => { const item = storage.registered_bytes[scope]; return <article key={scope}><span>{scope === 'local' ? '本机' : '远程'}原始数据集 · 登记统计</span><strong>{item.unknown_count ? '至少 ' : ''}{formatBytes(item.bytes)}</strong><small>{item.count} 个登记{item.unknown_count ? ` · ${item.unknown_count} 个缺少大小统计` : ''}</small></article> })}
          <article><span>平台临时文件{partial ? ' · 已观察到' : ''}</span><strong>{size(storage.total_bytes)}</strong><small>中转 {size(storage.transfer_bytes)} · 上传暂存 {size(storage.upload_staging_bytes)}</small></article>
          <article><span>中转配额占用{partial ? ' · 已观察到' : ''}</span><strong>{size(storage.quota_used_bytes)} / {data.policy.storage_quota_bytes ? formatBytes(data.policy.storage_quota_bytes) : '不限'}</strong><small>{data.policy.storage_quota_bytes && storage.status === 'complete' ? `剩余 ${size(storage.quota_remaining_bytes)}` : '只计入中转归档，不等于全部临时文件'}</small></article>
        </div>
        <p className="dataset-settings-note">原始大小来自数据库登记，包含历史统计，不代表实时磁盘扫描；删除登记后原始文件仍保留，且不再计入这里。登记汇总读取时间：{dateLabel(storage.registered_bytes.measured_at)}。</p>
        <div className={`dataset-settings-sampling ${storage.status === 'error' || partial ? 'warning' : ''}`} role="status">{storage.sampling ? '正在后台采样临时目录，页面可继续操作。' : storage.status === 'pending' ? '等待后台采样临时目录。' : partial ? '当前为部分统计，仅代表已观察到的占用，实际占用可能更大。' : storage.status === 'error' ? '临时空间统计失败，显示已有结果。' : '临时空间统计完成。'}{storage.sampled_at ? ` 最近采样：${dateLabel(storage.sampled_at)}。` : ''}{storage.error ? ` ${storage.error}` : ''}</div>
        <dl className="dataset-settings-paths"><div><dt>平台中转目录</dt><dd><code>{data.paths.transfer}</code></dd></div><div><dt>浏览器上传暂存目录</dt><dd><code>{data.paths.upload_staging}</code></dd></div><div><dt><HardDrive size={13} /> 中转所在磁盘</dt><dd>可用 {size(storage.disk.free_bytes)} / {size(storage.disk.total_bytes)}</dd></div></dl>
      </section>
      <section className="dataset-settings-section" aria-labelledby="dataset-cleanup-title">
        <div className="dataset-settings-heading"><h4 id="dataset-cleanup-title">临时文件清理</h4><span>{data.operations.failed_count} 个失败操作</span></div>
        <ul className="dataset-settings-rules"><li>成功、取消操作的中心中转快照在处理和传输释放后自动清理；锁尚未释放时暂缓清理。</li><li>浏览器上传暂存{data.cleanup.upload_staging_retention_seconds ? `保留完成回执后按 ${duration(data.cleanup.upload_staging_retention_seconds)} 期限回收` : '已关闭按保留期限自动清理'}。</li><li>失败中转快照：{duration(data.cleanup.failed_retention_seconds)}。失败上传暂存供重试保留，取消后按上传暂存规则回收；发布回执和提交中的文件仍受保护。</li></ul>
        {storage.operations.length ? <details className="dataset-settings-details"><summary>查看临时文件占用 · {storage.displayed_operation_count} 项{storage.truncated_operation_count ? `，另有 ${storage.truncated_operation_count} 项未展示` : ''}</summary><div className="dataset-settings-operation-list">{storage.operations.map(op => <article key={op.id}><div><strong>{op.name || op.id}</strong><span>{cleanupLabels[op.cleanup_state] || op.cleanup_state} · {op.sampled_complete ? '' : '至少 '}{formatBytes(op.total_bytes)}</span></div><small>{op.cleanup_reason}</small><small>中转临时 {formatBytes(op.transfer_bytes)}（其中归档 {formatBytes(op.bundle_bytes)}） · 上传暂存 {formatBytes(op.upload_staging_bytes)}</small></article>)}</div></details> : <p className="dataset-settings-note">{storage.status === 'complete' ? '当前采样未发现临时文件占用。' : '暂未采集到临时文件明细。'}</p>}
      </section>
      <section className="dataset-settings-section" aria-labelledby="dataset-worker-title">
        <div className="dataset-settings-heading"><h4 id="dataset-worker-title">后台处理</h4><span>{data.worker.status === 'running' ? '服务运行中' : data.worker.status === 'error' ? '服务异常' : '服务未运行'} · {data.operations.active_count} 个进行中</span></div>
        {workerError && <div className="alert error" role="alert">{workerError}</div>}
        {data.operations.active.length ? <div className="dataset-settings-operation-list">{data.operations.active.map(op => <article key={op.id}><div><strong>{op.name || kinds[op.kind] || op.kind}</strong><span>{op.node_name || (op.node_id === 'local' ? '本机' : '远程')} · {kinds[op.kind] || op.kind}</span></div><DatasetOperationProgress operation={op} /></article>)}</div> : <p className="dataset-settings-note">暂无进行中的数据集后台操作。</p>}
        {data.operations.active_count > data.operations.active.length && <p className="dataset-settings-note">此处仅展示部分进行中的操作，全部记录可在数据集管理页面查看。</p>}
        {!!data.remote_nodes.length && <details className="dataset-settings-details"><summary>远程服务器与上次上报 · {data.remote_nodes.length} 台</summary><div className="dataset-settings-operation-list">{data.remote_nodes.map(node => <article key={node.id}><div><strong>{node.name}</strong><span className={`settings-status-badge ${node.online ? 'online' : 'offline'}`}>{node.online ? '在线' : '离线 · 保留上次上报'}</span></div><small>最近上报：{dateLabel(node.last_seen_at)} · Agent {node.agent_version || '版本未知'}</small>{(() => { const catalogue = storage.registered_bytes.nodes?.find(item => item.node_id === node.id); return <small>登记数据集：{catalogue?.count || 0} 个 · {catalogue?.unknown_count ? '至少 ' : ''}{formatBytes(catalogue?.bytes || 0)}{catalogue?.unknown_count ? ` · ${catalogue.unknown_count} 个缺少大小统计` : ''}（历史登记统计）</small> })()}<small>训练中读取：{node.concurrent_reads == null ? '未上报能力' : node.concurrent_reads ? '支持' : '不支持'} · 并行重新加载：{node.concurrent_refresh == null ? '未上报能力' : node.concurrent_refresh ? '支持' : '不支持'}</small>{node.dataset_roots.length ? <div className="dataset-settings-root-list">{node.dataset_roots.map(root => <code key={root}>{root}</code>)}</div> : <small>尚未上报数据集目录</small>}</article>)}</div></details>}
      </section>
      {draft && <form onSubmit={event => void save(event)} className="dataset-settings-policy" aria-label="数据集处理策略">
        <details className="dataset-settings-details"><summary>高级处理设置{dirty ? ' · 有未保存修改' : ''}</summary>
          <p className="dataset-settings-note">当前来源：{data.policy_source === 'instance' ? '此平台实例保存的设置' : '服务器环境配置'}。保存后平台后续资源检查使用新值；不修改 Agent 的源目录读取限速，也不改变上传、迁移的训练保护规则。</p>
          <div className="form-grid">{fields.map(field => <label key={field.key} className={field.key === 'storage_quota_bytes' || field.key === 'failed_retention_seconds' ? '' : 'span-2'}>{field.label}<span className="dataset-settings-number"><input type="number" required min={0} max={field.max} step={1} inputMode="numeric" aria-label={field.label} value={draft[field.key]} disabled={saving} onChange={event => edit(field.key, event.target.value)} /><span>{field.unit}</span></span><span className="dataset-settings-value">{policyHint(field.key, draft[field.key])}</span><small>{field.help}</small></label>)}</div>
          {outdated && <div className="settings-inline-error" role="alert">服务器设置已有更新，草稿保留。请载入服务器当前设置后再修改，避免覆盖其他人的配置。</div>}
          {saveError && <div className="alert error" role="alert">{saveError}</div>}
          {notice && <div className="settings-inline-status" role="status">{notice}</div>}
          <div className="dataset-settings-policy-actions"><button type="button" className="button ghost compact" disabled={saving} onClick={resetToServer}>载入服务器当前设置</button><button type="button" className="button ghost compact" disabled={saving} onClick={() => { setDraft(toDraft(data.defaults)); setBaseRevision(data.revision); dirtyRef.current = true; setDirty(true); setSaveError(''); setNotice('已载入环境默认值，点击保存后生效。') }}>载入环境默认值</button><button type="submit" className="button primary compact" disabled={saving || !dirty || outdated}><Save size={14} />{saving ? '正在保存…' : '保存设置'}</button></div>
        </details>
      </form>}
      <aside className="dataset-settings-explanation"><strong>登记与文件的区别</strong><p>“删除登记”只删除页面对应的数据库记录；“永久删除文件”会删除实际目录。更换默认保存位置只影响新上传，不搬动历史数据。日常上传、登记和操作历史请在“数据集管理”页面处理。</p></aside>
    </>}
  </div>
}
