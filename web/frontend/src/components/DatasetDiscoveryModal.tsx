import { useEffect, useRef, useState } from 'react'
import { ChevronLeft, RefreshCw, Search, X } from 'lucide-react'
import { api } from '../api'
import type { ComputeNode, DatasetBatchRequest, DatasetCandidate, DatasetCandidatePage, DatasetLocation, DatasetRegistrationBatch, DatasetScan, DatasetScanRequest } from '../types'
import { useModalFocus } from '../useModalFocus'
import { DatasetServerSelect } from './DatasetServerSelect'
import { datasetPhaseLabel } from '../datasetUploadSession'

const states: Record<string, string> = { ready: '可登记', registered: '已登记', registering: '登记中／待处理', needs_attention: '需要整理', unsupported: '暂不支持', inaccessible: '无法检查', scan_incomplete: '扫描未完成' }
const operationStates: Record<string, string> = { pending: '等待处理', running: '处理中', committing: '提交中', completed: '已完成', failed: '失败', cancelled: '已取消', skipped: '已跳过' }
const scanKey = 'dataset-discovery:last-scan'
function requestId() { return Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('') }
function remember(id: string) { try { sessionStorage.setItem(scanKey, id) } catch { /* Server history remains available. */ } }
function recalled() { try { return sessionStorage.getItem(scanKey) } catch { return null } }

export function DatasetDiscoveryModal({ onClose, onChanged }: { onClose: () => void; onChanged: () => void }) {
  const [mode, setMode] = useState<'setup' | 'results' | 'review'>('setup')
  const [locations, setLocations] = useState<DatasetLocation[]>([])
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [scans, setScans] = useState<DatasetScan[]>([])
  const [identity, setIdentity] = useState('')
  const [scan, setScan] = useState<DatasetScan | null>(null)
  const [data, setData] = useState<DatasetCandidatePage | null>(null)
  const [batches, setBatches] = useState<DatasetRegistrationBatch[]>([])
  const [server, setServer] = useState('all')
  const [locationIds, setLocationIds] = useState<string[]>([])
  const [limits, setLimits] = useState({ depth: 3, max_candidates: 500, max_directories: 10000, max_seconds: 120 })
  const [page, setPage] = useState(1)
  const [state, setState] = useState('all')
  const [query, setQuery] = useState('')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<DatasetCandidate[]>([])
  const [ownership, setOwnership] = useState<'external' | 'managed'>('external')
  const [acknowledge, setAcknowledge] = useState(false)
  const [loading, setLoading] = useState(true)
  const [listLoading, setListLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [listError, setListError] = useState('')
  const [notice, setNotice] = useState('')
  const [refresh, setRefresh] = useState(0)
  const scanRequest = useRef<DatasetScanRequest | null>(null)
  const batchRequest = useRef<DatasetBatchRequest | null>(null)
  const close = () => { if (!busy) { if (mode === 'review') setMode('results'); else onClose() } }
  const ref = useModalFocus<HTMLDivElement>(true, close, { canClose: !busy })
  useEffect(() => {
    const controller = new AbortController()
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    void Promise.all([api.datasetLocations(controller.signal), api.computeNodes(controller.signal), api.datasetScans(controller.signal)]).then(([ls, ns, ss]) => {
      if (controller.signal.aborted) return
      setLocations(ls); setNodes(ns); setScans(ss)
      const id = ss.find(s => s.id === recalled())?.id || ss[0]?.id || ''
      if (id) { setIdentity(id); setMode('results') }
    }).catch((e: Error) => { if (!controller.signal.aborted) setError(e.message) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => { controller.abort(); document.body.style.overflow = previous }
  }, [])
  useEffect(() => { const timer = setTimeout(() => { setSearch(query); setPage(1) }, 250); return () => clearTimeout(timer) }, [query])
  useEffect(() => {
    if (!identity) return
    let active = true, running = false
    let controller: AbortController | null = null
    let timer: ReturnType<typeof setTimeout> | undefined
    async function load() {
      if (!active || running || document.hidden) return
      running = true; controller = new AbortController()
      try {
        const params = new URLSearchParams({ page: String(page), state, q: search, node_id: server })
        const [nextScan, nextData, nextBatches] = await Promise.all([api.datasetScan(identity, controller.signal), api.datasetCandidates(identity, params, controller.signal), api.datasetRegistrationBatches(identity, controller.signal)])
        if (active) { setScan(nextScan); setData(nextData); setBatches(nextBatches); setListError(''); setListLoading(false); if (nextData.page !== page) setPage(nextData.page) }
      } catch (e) { if (active && !controller.signal.aborted) { setListError(e instanceof Error ? e.message : '扫描结果读取失败'); setListLoading(false) } }
      finally { running = false; if (active) timer = setTimeout(load, 3000) }
    }
    function visible() { if (!document.hidden) { clearTimeout(timer); void load() } }
    setListLoading(true); void load(); document.addEventListener('visibilitychange', visible)
    return () => { active = false; controller?.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible) }
  }, [identity, page, state, search, server, refresh])
  useEffect(() => { ref.current?.focus() }, [mode, ref])

  const nodeName = (id: string) => nodes.find(n => n.id === id)?.name || locations.find(l => l.node_id === id)?.node_name || (id === 'local' ? '本机服务器' : id)
  const serverIds = Array.from(new Set(['local', ...locations.map(l => l.node_id), ...(scan?.operations.map(o => o.source_node || o.node_id) || [])]))
  const serverOptions = [{ id: 'all', name: '全部服务器' }, ...serverIds.map(id => ({ id, name: nodeName(id) }))]
  function capable(loc: DatasetLocation) {
    if (!loc.enabled || loc.status !== 'ready') return false
    if (loc.node_id === 'local') return true
    const n = nodes.find(n => n.id === loc.node_id)
    return !!n?.enabled && !!n.training?.dataset_discovery && versionSupported(n.agent_version)
  }
  function chooseScan(id: string) {
    setIdentity(id); remember(id); setScan(null); setData(null); setBatches([]); setSelected([]); setPage(1); setState('all'); setServer('all'); setQuery(''); setSearch(''); setMode('results'); setError(''); setNotice('')
  }
  function filterServer(value: string) { setServer(value); setSelected([]); setPage(1) }
  async function start(e: React.FormEvent) {
    e.preventDefault(); if (busy) return
    const chosen = locations.filter(l => locationIds.includes(l.id))
    if (!chosen.length || chosen.some(l => !capable(l))) { setError('请选择已校验且支持扫描的保存位置'); return }
    const payload = { ...limits, location_ids: [...locationIds].sort() }
    if (!scanRequest.current || JSON.stringify({ ...scanRequest.current, request_id: undefined }) !== JSON.stringify({ ...payload, request_id: undefined })) scanRequest.current = { request_id: requestId(), ...payload }
    setBusy(true); setError('')
    try {
      const next = await api.createDatasetScan(scanRequest.current)
      setScans(current => [next, ...current.filter(s => s.id !== next.id)].slice(0, 50)); chooseScan(next.id); scanRequest.current = null; onChanged()
    } catch (e) { setError(e instanceof Error ? e.message : '扫描提交失败') }
    finally { setBusy(false) }
  }
  async function action(work: () => Promise<unknown>) {
    setBusy(true); setError('')
    try { await work(); setRefresh(v => v + 1); onChanged() }
    catch (e) { setError(e instanceof Error ? e.message : '操作失败') }
    finally { setBusy(false) }
  }
  function review() { batchRequest.current = null; setOwnership('external'); setAcknowledge(false); setMode('review'); setError('') }
  async function submit() {
    if (!selected.length || busy || (ownership === 'managed' && !acknowledge)) return
    const payload = { candidate_ids: selected.map(c => c.id).sort(), candidate_tokens: Object.fromEntries(selected.map(c => [c.id, c.selection_token])), ownership, acknowledge }
    if (!batchRequest.current || JSON.stringify({ ...batchRequest.current, request_id: undefined }) !== JSON.stringify({ ...payload, request_id: undefined })) batchRequest.current = { request_id: requestId(), ...payload }
    await action(async () => {
      const batch = await api.registerDatasetBatch(identity, batchRequest.current!)
      setBatches(current => [batch, ...current.filter(b => b.id !== batch.id)]); setSelected([]); setMode('results'); setNotice(`批次已受理：${batch.items.length} 项，具体结果见下方批量登记进度。`); batchRequest.current = null
    })
  }
  const pageEligible = data?.items.filter(c => c.state === 'ready') || []
  const selectedSet = new Set(selected.map(c => c.id))
  function selectPage(checked: boolean) {
    if (!checked) setSelected(current => current.filter(c => !pageEligible.some(p => p.id === c.id)))
    else setSelected(current => [...current, ...pageEligible.filter(c => !current.some(s => s.id === c.id))].slice(0, 500))
  }
  const allPageSelected = !!pageEligible.length && pageEligible.every(c => selectedSet.has(c.id))
  const scanActive = scan?.status === 'running'
  const canCancelScan = scan?.operations.some(o => o.can_cancel)
  return <div className="dataset-drawer-backdrop" onMouseDown={close}>
    <div ref={ref} className="dataset-discovery-drawer" role="dialog" aria-modal="true" aria-labelledby="dataset-discovery-title" tabIndex={-1} onMouseDown={e => e.stopPropagation()}>
      <header><div><h2 id="dataset-discovery-title">扫描已有数据集</h2><p>只读发现候选目录，完整校验通过后登记。关闭面板不会取消后台任务。</p></div><button type="button" className="icon-button" aria-label="关闭目录扫描" disabled={busy} onClick={close}><X size={20} /></button></header>
      <div className="dataset-discovery-navigation">
        {scans.length > 0 && <label>最近扫描<select aria-label="选择扫描记录" value={identity} disabled={busy || mode === 'review'} onChange={e => chooseScan(e.target.value)}>{!identity && <option value="">请选择扫描</option>}{scans.map(s => <option value={s.id} key={s.id}>{new Date(s.created_at).toLocaleString('zh-CN')} · {s.id.slice(0, 8)}</option>)}</select></label>}
        {mode !== 'setup' && <button type="button" className="button compact ghost" disabled={busy} onClick={() => { setMode('setup'); setError(''); setSelected([]); setServer('all') }}>新建扫描</button>}
        {mode === 'setup' && identity && <button type="button" className="button compact ghost" onClick={() => setMode('results')}><ChevronLeft size={14} />返回扫描结果</button>}
      </div>
      {error && <div className="alert error" role="alert">{error}</div>}
      {loading ? <div className="dataset-discovery-body"><p role="status">正在读取保存位置和扫描记录…</p></div> : mode === 'setup' ? <form className="dataset-discovery-body form-grid dataset-discovery-setup" onSubmit={e => void start(e)}>
        <div className="dataset-action-field"><span>扫描服务器</span><DatasetServerSelect label="选择扫描服务器" value={server} options={serverOptions} disabled={busy} onChange={setServer} /></div>
        <fieldset disabled={busy}><legend>扫描保存位置（已选 {locationIds.length}，最多 16 个）</legend><div className="dataset-discovery-locations">
          {locations.filter(l => server === 'all' || server === l.node_id).map(l => <label key={l.id}><input type="checkbox" checked={locationIds.includes(l.id)} disabled={!capable(l) || (!locationIds.includes(l.id) && locationIds.length >= 16)} onChange={e => setLocationIds(current => e.target.checked ? [...current, l.id] : current.filter(id => id !== l.id))} /><span><strong>{nodeName(l.node_id)} · {l.name}</strong><code>{l.resolved_path || l.path}</code>{!capable(l) && <small>{!l.enabled || l.status !== 'ready' ? '位置未启用或尚未校验通过' : '需要 Agent 2.2.0 并上报 dataset_discovery 能力'}</small>}{nodes.find(n => n.id === l.node_id)?.status === 'offline' && <small>远程离线，任务会等待节点上线</small>}</span></label>)}
          {!locations.length && <p>暂无保存位置，请关闭面板后在“保存位置”中添加并校验目录。</p>}
        </div></fieldset>
        <div className="dataset-discovery-limits">{([{ key: 'depth', label: '扫描深度', min: 0, max: 12 }, { key: 'max_candidates', label: '每个位置候选上限', min: 1, max: 2000 }, { key: 'max_directories', label: '每个位置目录上限', min: 1, max: 50000 }, { key: 'max_seconds', label: '每个位置时限（秒）', min: 5, max: 600 }] as const).map(f => <label key={f.key}>{f.label}<input type="number" required min={f.min} max={f.max} disabled={busy} value={limits[f.key]} onChange={e => setLimits(current => ({ ...current, [f.key]: Number(e.target.value) }))} /></label>)}</div>
        <aside className="dataset-action-help">深度 0 仅检查所选位置本身，深度 3 包含向下 3 层。跳过隐藏目录、缓存及图片／标签内部目录，不跟随符号链接；不会修改文件或训练配置。远程扫描等待训练空闲后执行。</aside>
        <div className="dataset-inline-actions"><button type="submit" className="button primary" disabled={busy || !locationIds.length}>{busy ? '正在提交…' : '开始后台扫描'}</button></div>
      </form> : mode === 'review' ? <div className="dataset-discovery-body dataset-discovery-review">
        <h3>确认批量登记 {selected.length} 个数据集</h3><p>正式登记将完整校验图片、标签与 SHA-256。远程数据会经过中台传输，需要临时空间；登记失败不影响其他项目，也不会自动整理或覆盖原文件。</p>
        <div className="dataset-discovery-selected">{selected.map(c => <div key={c.id}><strong>{c.node_name} · {c.name}</strong><code>{c.path}</code></div>)}</div>
        <label>文件管理方式<select aria-label="文件管理方式" value={ownership} disabled={busy} onChange={e => { setOwnership(e.target.value as 'external' | 'managed'); setAcknowledge(false) }}><option value="external">仅登记：保留外部文件所有权</option><option value="managed">接管：允许平台永久删除文件</option></select></label>
        {ownership === 'managed' && <label className="dataset-action-check"><span><input type="checkbox" checked={acknowledge} disabled={busy} onChange={e => setAcknowledge(e.target.checked)} />我确认所选目录可由平台管理和永久删除</span></label>}
        <div className="dataset-inline-actions"><button type="button" className="button ghost" disabled={busy} onClick={() => setMode('results')}>返回选择</button><button type="button" className="button primary" disabled={busy || (ownership === 'managed' && !acknowledge)} onClick={() => void submit()}>{busy ? '正在提交…' : '确认批量登记'}</button></div>
      </div> : <div className="dataset-discovery-body">
        {scan && <section className="dataset-discovery-progress"><div className="dataset-section-heading"><strong>{scanActive ? '扫描处理中' : scan.operations.some(o => o.progress.limited) ? '扫描达到限制，部分目录尚未检查' : scan.status === 'completed' ? '扫描范围内检查完成' : '扫描已结束，部分位置可能未完成'}</strong><button type="button" className="button compact ghost" disabled={busy || !canCancelScan} onClick={() => void action(() => api.cancelDatasetScan(identity))}>取消剩余扫描</button></div>
          {scan.operations.map(o => <div className="dataset-discovery-progress-row" key={o.id}><strong>{o.source_node_name || o.node_name} · {o.name} · {operationStates[o.status] || o.status}</strong><span>已检查 {o.progress.visited || 0} 个目录 · 发现 {o.progress.found || 0} 个候选{o.waiting_offline ? ' · 等待节点上线' : o.waiting_training ? ' · 等待训练空闲' : ''}</span>{o.progress.current_path && <code>{o.progress.current_path}</code>}{o.progress.limited && <p role="status">扫描未完整结束：{o.progress.limit_reason}</p>}{!!o.progress.skipped_depth && <small>有 {o.progress.skipped_depth} 个子目录超出深度范围，需调大深度另行扫描。</small>}{!!o.progress.skipped_links && <small>跳过 {o.progress.skipped_links} 个符号链接。</small>}{o.error && <p className="dataset-operation-error">{o.error}</p>}{o.status === 'failed' && <button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.controlDatasetOperation(o.id, 'retry'))}>重试此位置</button>}</div>)}
        </section>}
        {notice && <p className="dataset-history-notice" role="status">{notice}</p>}
        <div className="dataset-discovery-filters"><label className="dataset-search"><Search size={15} /><input aria-label="搜索候选目录" placeholder="搜索目录名称或路径" value={query} maxLength={160} onChange={e => { setQuery(e.target.value); setSelected([]) }} /></label><DatasetServerSelect label="按候选服务器筛选" value={server} options={serverOptions} onChange={filterServer} disabled={busy} /><select aria-label="候选状态" value={state} onChange={e => { setState(e.target.value); setPage(1); setSelected([]) }}><option value="all">全部状态</option>{Object.entries(states).map(([v, label]) => <option key={v} value={v}>{label}</option>)}</select><button type="button" className="button compact ghost" disabled={busy} onClick={() => setRefresh(v => v + 1)}><RefreshCw size={14} />刷新结果</button></div>
        <p className="dataset-discovery-summary">初检可登记 {data?.counts.ready || 0} · 已登记 {data?.counts.registered || 0} · 登记中／待处理 {data?.counts.registering || 0} · 其余候选 {(data ? Object.entries(data.counts).filter(([s]) => !['ready', 'registered', 'registering'].includes(s)).reduce((n, [, count]) => n + count, 0) : 0)}</p>
        {listError && <p className="alert error" role="alert">{listError}</p>}
        <div className="dataset-discovery-selection"><label><input type="checkbox" aria-label="选择本页可登记候选" checked={allPageSelected} disabled={!pageEligible.length || busy || listLoading || !!listError} onChange={e => selectPage(e.target.checked)} />选择本页可登记</label><span>已选 {selected.length} / 500 · 跨页保留，更换筛选清空</span><button type="button" className="button compact ghost" disabled={!selected.length || busy} onClick={() => setSelected([])}>清空选择</button></div>
        <div className="dataset-discovery-candidates" aria-busy={listLoading}>
          {data?.items.map(c => <article key={c.id}><input type="checkbox" aria-label={`选择候选：${c.path}`} checked={selectedSet.has(c.id)} disabled={c.state !== 'ready' || busy || listLoading || !!listError || (!selectedSet.has(c.id) && selected.length >= 500)} onChange={e => setSelected(current => e.target.checked ? [...current, c] : current.filter(item => item.id !== c.id))} /><div><div className="dataset-discovery-candidate-heading"><strong>{c.name}</strong><span className={`dataset-candidate-state ${c.state}`}>{states[c.state] || c.state}</span></div><small>{c.node_name}{c.task === 'detect' ? ' · 目标检测（初检）' : ''}</small><code>{c.path}</code><p>{c.reason}</p>{c.classes.length > 0 && <small>类别：{c.classes.join('、')}</small>}</div></article>)}
          {!data?.items.length && <p className="empty">{listLoading ? '正在读取候选目录…' : scanActive ? '扫描进行中，发现结果后会自动更新。' : '当前范围没有候选；可检查扫描位置、深度及筛选条件。'}</p>}
        </div>
        <div className="dataset-discovery-pagination"><span>共 {data?.total || 0} 项</span><button type="button" className="button compact ghost" disabled={busy || listLoading || page <= 1} onClick={() => setPage(p => p - 1)}>上一页</button><span>{page} / {Math.max(1, Math.ceil((data?.total || 0) / 20))}</span><button type="button" className="button compact ghost" disabled={busy || listLoading || !data || page * 20 >= data.total} onClick={() => setPage(p => p + 1)}>下一页</button></div>
        {batches.length > 0 && <section className="dataset-discovery-batches"><h3>批量登记进度</h3>{batches.map(batch => <details key={batch.id} open={batch.status === 'running' || undefined}><summary>批次 {batch.id.slice(0, 8)} · 成功 {batch.counts.completed || 0} · 失败 {batch.counts.failed || 0} · 等待／处理中 {(batch.counts.pending || 0) + (batch.counts.running || 0) + (batch.counts.committing || 0)} · 已取消 {batch.counts.cancelled || 0} · 跳过 {batch.counts.skipped || 0}</summary><div>{batch.items.some(i => i.can_cancel) && <button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.cancelDatasetBatch(batch.id))}>取消未完成登记</button>}{batch.items.map(item => <article key={item.candidate_id}><strong>{nodeName(item.node_id)} · {item.name} · {operationStates[item.status] || item.status}</strong><code>{item.path}</code>{item.phase && <small>{datasetPhaseLabel[item.phase] || item.phase}</small>}{item.error && <p className="dataset-operation-error">{item.error}</p>}{item.operation_id && item.status === 'failed' && <div className="dataset-inline-actions"><button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.controlDatasetOperation(item.operation_id!, 'retry'))}>重试此项</button>{item.can_cancel && <button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.controlDatasetOperation(item.operation_id!, 'cancel'))}>取消此项</button>}</div>}</article>)}</div></details>)}</section>}
      </div>}
      <footer><span>候选初检通过不代表正式登记成功。</span><button type="button" className="button ghost" disabled={busy} onClick={close}>{mode === 'review' ? '返回结果' : '关闭'}</button>{mode === 'results' && <button type="button" className="button primary" disabled={busy || !selected.length || listLoading || !!listError} onClick={review}>登记选中（{selected.length}）</button>}</footer>
    </div>
  </div>
}

function versionSupported(value: string | null) {
  const parts = (value || '').split('.').map(Number)
  return parts[0] > 2 || (parts[0] === 2 && parts[1] >= 2)
}
