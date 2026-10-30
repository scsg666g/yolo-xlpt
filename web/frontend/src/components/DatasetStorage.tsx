import { useEffect, useId, useRef, useState } from 'react'
import { api, formatBytes } from '../api'
import type { ComputeNode, DatasetLocation, ManagedDataset } from '../types'
import { useModalFocus } from '../useModalFocus'
import { ChevronDown, MoreHorizontal, X } from 'lucide-react'
import { DatasetServerSelect } from './DatasetServerSelect'


export function DatasetStorageSettings({ defaultExpanded = false, refreshToken = 0, onStatusChange }: { defaultExpanded?: boolean; refreshToken?: number; onStatusChange?: (status: { loading: boolean; checkedAt: Date | null; error: string }) => void }) {
  const section = useRef<HTMLElement>(null)
  const contentId = useId()
  const [expanded, setExpanded] = useState(defaultExpanded)
  const [locations, setLocations] = useState<DatasetLocation[]>([])
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [error, setError] = useState('')
  const [loadError, setLoadError] = useState('')
  const [busy, setBusy] = useState(false)
  const [adding, setAdding] = useState(false)
  const checkedAt = useRef<Date | null>(null)
  const onStatusChangeRef = useRef(onStatusChange)
  useEffect(() => { onStatusChangeRef.current = onStatusChange }, [onStatusChange])
  const [form, setForm] = useState({ node_id: 'local', name: '', path: '', create: false, is_default: false })
  useEffect(() => {
    let active = true
    let running = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let controller: AbortController | null = null
    async function load() {
      if (!active || running || document.hidden) return
      running = true; controller = new AbortController()
      onStatusChangeRef.current?.({ loading: true, checkedAt: checkedAt.current, error: '' })
      try {
        const [locationResult, nodeResult] = await Promise.allSettled([api.datasetLocations(controller.signal), api.computeNodes(controller.signal)])
        if (active && !controller.signal.aborted) {
          const failures: string[] = []
          if (locationResult.status === 'fulfilled') { setLocations(locationResult.value); checkedAt.current = new Date() }
          else failures.push(`保存位置：${locationResult.reason instanceof Error ? locationResult.reason.message : '读取失败'}`)
          if (nodeResult.status === 'fulfilled') setNodes(nodeResult.value)
          else failures.push(`服务器列表：${nodeResult.reason instanceof Error ? nodeResult.reason.message : '读取失败'}`)
          const message = failures.join('；')
          setLoadError(message)
          onStatusChangeRef.current?.({ loading: false, checkedAt: checkedAt.current, error: message })
        }
      } catch (e) { if (active && !controller.signal.aborted) {
        const message = e instanceof Error ? e.message : '保存位置读取失败'
        setLoadError(message); onStatusChangeRef.current?.({ loading: false, checkedAt: checkedAt.current, error: message })
      } }
      finally { running = false; if (active) timer = setTimeout(load, 5000) }
    }
    function visible() { if (!document.hidden) { clearTimeout(timer); void load() } }
    void load(); document.addEventListener('visibilitychange', visible)
    return () => { active = false; controller?.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible) }
  }, [refreshToken])
  useEffect(() => {
    const root = section.current
    function outside(event: PointerEvent) {
      root?.querySelectorAll<HTMLDetailsElement>('.dataset-location-more[open]').forEach(menu => {
        if (!menu.contains(event.target as Node)) menu.open = false
      })
    }
    function escape(event: KeyboardEvent) {
      if (event.key !== 'Escape' || event.isComposing) return
      const menu = root?.querySelector<HTMLDetailsElement>('.dataset-location-more[open]')
      if (menu) { event.preventDefault(); event.stopPropagation(); menu.open = false; menu.querySelector('summary')?.focus() }
    }
    document.addEventListener('pointerdown', outside)
    root?.addEventListener('keydown', escape)
    return () => { document.removeEventListener('pointerdown', outside); root?.removeEventListener('keydown', escape) }
  }, [])
  async function action(work: () => Promise<unknown>) {
    setBusy(true); setError('')
    try { await work(); setLocations(await api.datasetLocations()) } catch (e) { setError(e instanceof Error ? e.message : '目录操作失败') } finally { setBusy(false) }
  }
  return <section ref={section} className="dataset-storage-panel" aria-busy={busy}>
    <div className="dataset-section-heading">
      <button type="button" className="dataset-storage-toggle" aria-expanded={expanded} aria-controls={contentId} onClick={() => setExpanded(value => !value)}>
        <ChevronDown size={16} /><strong>数据集保存位置</strong><span>{locations.length} 个位置 · {expanded ? '收起' : '展开'}</span>
      </button>
      <button type="button" className="button compact ghost" onClick={() => { setExpanded(true); setAdding(value => !value) }}>{adding ? '取消添加' : '添加保存位置'}</button>
    </div>
    {error && <div className="alert error" role="alert">{error}</div>}
    {loadError && <div className="alert error" role="alert">刷新未全部成功：{loadError}{checkedAt.current ? '；未成功的部分保留上次数据。' : ''}</div>}
    <div id={contentId} className="dataset-storage-content" hidden={!expanded}>
    <p>默认位置仅影响新上传；旧数据仍按登记的服务器和实际路径管理。相对路径以所选服务器项目目录为基准。</p>
    {adding && <div className="form-grid dataset-location-form">
      <label>服务器<DatasetServerSelect label="保存位置所在服务器" value={form.node_id} disabled={busy} onChange={v => setForm({ ...form, node_id: v })} options={[{ id: 'local', name: '本机（中台）' }, ...nodes.filter(n => n.kind === 'agent' && n.enabled).map(n => ({ id: n.id, name: n.name, status: n.status === 'online' ? '在线' : '离线' }))]} /></label>
      <label>位置名称<input value={form.name} disabled={busy} maxLength={160} placeholder="例如：训练数据盘" onChange={e => setForm({ ...form, name: e.target.value })} /></label>
      <label className="span-2">服务器目录<input value={form.path} disabled={busy} placeholder="例如：数据集，或已配置白名单内的绝对路径" onChange={e => setForm({ ...form, path: e.target.value })} /></label>
      <label><span><input type="checkbox" disabled={busy} checked={form.create} onChange={e => setForm({ ...form, create: e.target.checked })} /> 目录不存在时创建</span></label>
      <label><span><input type="checkbox" disabled={busy} checked={form.is_default} onChange={e => setForm({ ...form, is_default: e.target.checked })} /> 校验通过后设为默认</span></label>
      <button type="button" className="button primary" disabled={busy || !form.name.trim() || !form.path.trim()} onClick={() => void action(async () => { await api.createDatasetLocation(form); setAdding(false) })}>登记并校验</button>
    </div>}
    <div className="dataset-location-groups">{Array.from(new Set(locations.map(l => l.node_id))).map(nodeId => <section key={nodeId} className="dataset-location-group"><h3>{nodes.find(n => n.id === nodeId)?.name || (nodeId === 'local' ? '本机（中台）' : nodeId)}<small>{locations.filter(l => l.node_id === nodeId).length} 个位置</small></h3><div className="dataset-location-list">{locations.filter(l => l.node_id === nodeId).map(l => <article key={l.id}>
      <div><strong>{l.name}{l.is_default ? ' · 默认' : ''}{!l.enabled ? ' · 已禁用' : ''}</strong>
        <code>{l.resolved_path || l.path}</code><small>{l.status === 'ready' ? '上次校验通过' : l.status === 'error' ? '校验失败' : '等待检查'}{l.details.free_bytes != null ? ` · 可用 ${formatBytes(l.details.free_bytes)} / ${formatBytes(l.details.total_bytes || 0)}` : ''}{l.details.checked_at ? ` · ${new Date(l.details.checked_at).toLocaleString()}` : ''}</small>
        {l.details.message && <small role="alert">{l.details.message}</small>}</div>
      <div className="dataset-inline-actions"><button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.checkDatasetLocation(l.id))}>检查</button>{l.status !== 'ready' && <button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.checkDatasetLocation(l.id, true))}>创建并检查</button>}
        <button type="button" className="button compact ghost" disabled={busy || l.status !== 'ready' || l.is_default} onClick={() => void action(() => api.updateDatasetLocation(l.id, true, true))}>设为默认</button>
        <details className="dataset-location-more"><summary aria-label={`更多操作：${l.name}`} title="更多操作"><MoreHorizontal size={18} /></summary><div><button type="button" className="button compact ghost" disabled={busy} onClick={e => { e.currentTarget.closest('details')?.removeAttribute('open'); void action(() => api.updateDatasetLocation(l.id, !l.enabled, false)) }}>{l.enabled ? '禁用' : '启用'}</button>
        <button type="button" className="button compact danger" disabled={busy || l.is_default} onClick={e => { e.currentTarget.closest('details')?.removeAttribute('open'); void action(() => api.forgetDatasetLocation(l.id)) }}>移除配置</button></div></details></div>
    </article>)}</div></section>)}</div>
    </div>
  </section>
}

export function DatasetStorageModal({ onClose }: { onClose: () => void }) {
  const ref = useModalFocus<HTMLDivElement>(true, onClose)
  useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = previous }
  }, [])
  return <div className="modal-backdrop" onMouseDown={onClose}><div ref={ref} className="modal dataset-storage-modal" role="dialog" aria-modal="true" aria-labelledby="dataset-storage-title" tabIndex={-1} onMouseDown={e => e.stopPropagation()}>
    <header><h2 id="dataset-storage-title">保存位置配置</h2><button type="button" className="icon-button" aria-label="关闭保存位置" onClick={onClose}><X size={20} /></button></header>
    <div className="dataset-storage-modal-body"><DatasetStorageSettings defaultExpanded /></div>
    <footer><button type="button" className="button ghost" onClick={onClose}>关闭</button></footer>
  </div></div>
}

export function DatasetActionModal({ dataset, onClose, onDone }: { dataset?: ManagedDataset; onClose: () => void; onDone: () => void }) {
  const [locations, setLocations] = useState<DatasetLocation[]>([])
  const [target, setTarget] = useState('')
  const [name, setName] = useState(dataset?.name || '')
  const [remove, setRemove] = useState(false)
  const [ownership, setOwnership] = useState('external')
  const [acknowledge, setAcknowledge] = useState(false)
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const ref = useModalFocus<HTMLFormElement>(true, onClose, { canClose: !busy })
  useEffect(() => {
    const controller = new AbortController()
    void api.datasetLocations(controller.signal).then(ls => {
      if (controller.signal.aborted) return
      const valid = ls.filter(l => l.enabled && l.status === 'ready')
      setLocations(valid); setTarget(valid.find(l => l.is_default)?.id || valid[0]?.id || '')
    }).catch((e: Error) => { if (!controller.signal.aborted) setError(e.message) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { controller.abort(); document.body.style.overflow = previous }
  }, [])
  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (busy || loading || !target) return
    setBusy(true); setError('')
    try {
      if (dataset) await api.migrateDataset(dataset.id, target, name, remove)
      else await api.registerDataset(target, name, ownership, acknowledge)
      onDone(); onClose()
    } catch (e) { setError(e instanceof Error ? e.message : '提交失败') } finally { setBusy(false) }
  }
  const location = locations.find(l => l.id === target)
  const serverName = (l: DatasetLocation) => l.node_name || (l.node_id === 'local' ? '本机服务器' : l.node_id)
  const serverOptions = Array.from(new Set(locations.map(l => l.node_id))).map(id => ({ id, name: serverName(locations.find(l => l.node_id === id)!) }))
  const serverLocations = locations.filter(l => l.node_id === location?.node_id)
  const destination = location ? `${(location.resolved_path || location.path).replace(/\/+$/, '')}/${name || '文件夹名称'}` : ''
  return <div className="modal-backdrop" onMouseDown={() => { if (!busy) onClose() }}><form ref={ref} className="modal dataset-action-modal" role="dialog" aria-modal="true" aria-labelledby="dataset-action-title" tabIndex={-1} onSubmit={submit} onMouseDown={e => e.stopPropagation()}>
    <header><div><h2 id="dataset-action-title">{dataset ? '复制或迁移数据集' : '登记服务器已有数据集'}</h2><p>{dataset ? '选择目标服务器和保存位置，校验成功后完成复制或迁移。' : '将服务器上已有的数据集加入平台，无需重新上传。'}</p></div><button type="button" className="icon-button" aria-label="关闭数据集操作" disabled={busy} onClick={onClose}><X size={20} /></button></header>
    <div className="modal-body form-grid dataset-action-body" aria-busy={loading}>
      {dataset && <section className="dataset-action-path"><strong>源目录 · {dataset.node_name || (dataset.node_id === 'local' ? '本机服务器' : dataset.node_id)}</strong><code>{dataset.directory}</code><small>原任务配置和历史路径保留；有引用时只能复制。</small></section>}
      <div className="dataset-action-field"><span>{dataset ? '目标服务器' : '已有目录所在服务器'}</span><DatasetServerSelect label={dataset ? '目标服务器' : '已有目录所在服务器'} value={location?.node_id || ''} options={serverOptions} disabled={busy || loading || !locations.length} onChange={id => { const candidates = locations.filter(l => l.node_id === id); setTarget((candidates.find(l => l.is_default) || candidates[0]).id) }} /></div>
      <label>{dataset ? '目标保存位置' : '已有目录所在位置'}<select required aria-label={dataset ? '目标保存位置' : '已有目录所在位置'} title={location?.resolved_path || location?.path} value={target} disabled={busy || loading || !locations.length} onChange={e => setTarget(e.target.value)}>{!target && <option value="">{loading ? '正在加载保存位置…' : '暂无可用保存位置'}</option>}{serverLocations.map(l => <option key={l.id} value={l.id} title={l.resolved_path || l.path}>{l.name}{l.is_default ? '（默认）' : ''}{serverLocations.some(other => other.id !== l.id && other.name === l.name) ? ` · ${l.resolved_path || l.path}` : ''}</option>)}</select></label>
      {!loading && !locations.length && !error && <p className="dataset-action-help" role="status">暂无已启用且校验通过的保存位置。请关闭此窗口，在“保存位置”中添加并校验服务器目录。</p>}
      <label>数据集文件夹名称<input required maxLength={160} value={name} disabled={busy} onChange={e => setName(e.target.value)} placeholder="例如：安全帽_v2" aria-describedby="dataset-action-destination" /></label>
      <section className="dataset-action-path" id="dataset-action-destination"><strong>{dataset ? '目标实际目录' : '实际目录'}{location ? ` · ${serverName(location)}` : ''}</strong>{destination ? <code>{destination}</code> : <small>{loading ? '正在加载保存位置…' : '选择可用保存位置后显示完整目录'}</small>}</section>
      {!dataset && <><aside className="dataset-action-help"><strong>登记格式 · YOLO 目标检测</strong><p>目录需包含 images/train、images/val、labels 和 data.yaml，且数据文件均在该目录内。标签每行使用“类别 中心x 中心y 宽 高”五列格式。</p><p>此登记入口暂不校验分割、姿态、旋转框或分类数据集；这些任务可在创建训练任务时配置对应服务器的数据路径。</p><p>提交后将在后台完整校验图片、标签及 SHA-256；远程校验需经过中台传输。</p></aside><label>文件管理方式<select value={ownership} disabled={busy} onChange={e => setOwnership(e.target.value)}><option value="external">仅登记：保留外部文件所有权</option><option value="managed">接管：允许平台永久删除文件</option></select><small>{ownership === 'external' ? '仅建立登记信息，平台不会删除原始数据集文件。' : '接管后可在平台删除数据集及其实际文件，请确认目录用途。'}</small></label>{ownership === 'managed' && <label className="dataset-action-check"><span><input type="checkbox" checked={acknowledge} disabled={busy} onChange={e => setAcknowledge(e.target.checked)} />我确认该目录可由平台管理与删除</span></label>}</>}
      {dataset && <label className="dataset-action-check"><span><input type="checkbox" checked={remove} disabled={busy || dataset.references.length > 0 || dataset.ownership !== 'managed'} onChange={e => setRemove(e.target.checked)} />目标校验保存成功后，删除源目录</span><small>未勾选时只复制。任何任务（含历史和回收站）引用源目录时不会清理源文件。</small></label>}
      {error && <div className="alert error" role="alert">{error}</div>}
    </div><footer><button type="button" className="button ghost" disabled={busy} onClick={onClose}>取消</button><button type="submit" className="button primary" disabled={busy || loading || !target || (ownership === 'managed' && !dataset && !acknowledge)}>{busy ? '正在提交…' : '提交后台处理'}</button></footer>
  </form></div>
}
