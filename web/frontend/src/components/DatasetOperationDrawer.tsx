import { DatasetOperationProgress } from './DatasetOperationProgress'
import { useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, RefreshCw, Search, Trash2, X } from 'lucide-react'
import { api, formatBytes } from '../api'
import type { DatasetHistoryRemoval, DatasetHistoryRemovalPreview, DatasetOperationPage, DatasetOperationSummary } from '../types'
import { useModalFocus } from '../useModalFocus'
import { DatasetServerSelect, type DatasetServerOption } from './DatasetServerSelect'

const statuses: Record<string, string> = { pending: '等待处理', running: '正在处理', committing: '提交文件中', completed: '已完成', failed: '失败', cancelled: '已取消' }
const kinds: Record<string, string> = { discover: '目录扫描', check: '目录检查', upload: '上传', register: '登记', refresh: '重新加载', migrate: '复制 / 迁移', delete: '永久删除' }

export function DatasetOperationDrawer({ onClose, onChanged, summary, servers }: {
  onClose: () => void; onChanged: () => void; summary: DatasetOperationSummary | null; servers: DatasetServerOption[]
}) {
  const [removal, setRemoval] = useState<{ request: DatasetHistoryRemoval; label: string } | null>(null)
  const [removing, setRemoving] = useState(false)
  const close = () => { if (!removing) { if (removal) setRemoval(null); else onClose() } }
  const ref = useModalFocus<HTMLDivElement>(true, close, { canClose: !removing })
  const [status, setStatus] = useState(summary && summary.active + summary.failed > 0 ? 'attention' : 'all')
  const [kind, setKind] = useState('all')
  const [server, setServer] = useState('all')
  const [query, setQuery] = useState('')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const [snapshot, setSnapshot] = useState('')
  const [latest, setLatest] = useState(summary?.latest_created_at)
  const [data, setData] = useState<DatasetOperationPage | null>(null)
  const [detail, setDetail] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  const [loading, setLoading] = useState(true)
  const [refresh, setRefresh] = useState(0)
  const [selected, setSelected] = useState<string[]>([])
  const [selectionMode, setSelectionMode] = useState(false)
  const [after, setAfter] = useState('')
  const [through, setThrough] = useState('')
  const [notice, setNotice] = useState('')
  const selectedIds = data?.items.filter(j => selected.includes(j.id) && j.can_delete_history).map(j => j.id) || []
  const eligibleIds = data?.items.filter(j => j.can_delete_history).map(j => j.id) || []
  const snapshotRef = useRef(snapshot)
  snapshotRef.current = snapshot
  useEffect(() => { const timer = setTimeout(() => { setSearch(query); setPage(1) }, 250); return () => clearTimeout(timer) }, [query])
  useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = previous }
  }, [])
  useEffect(() => {
    let active = true
    let controller: AbortController | null = null
    let timer: ReturnType<typeof setTimeout> | undefined
    let running = false
    async function load() {
      if (!active || running || document.hidden) return
      running = true; controller = new AbortController()
      const params = new URLSearchParams({ page: String(page), page_size: '20', status, kind, node_id: server, q: search })
      if (snapshotRef.current) params.set('before', snapshotRef.current)
      if (after) params.set('after', dayBoundary(after))
      if (through) params.set('through', dayBoundary(through, true))
      try {
        const result = await api.datasetOperationPage(params, controller.signal)
        if (active) {
          if (!snapshotRef.current) { snapshotRef.current = result.snapshot; setSnapshot(result.snapshot) }
          setData(result); setError(''); setLoading(false)
          if (result.page !== page) setPage(result.page)
        }
      } catch (e) { if (active && !controller.signal.aborted) { setError(e instanceof Error ? e.message : '操作记录加载失败'); setLoading(false) } }
      finally { running = false; if (active) timer = setTimeout(load, 2000) }
    }
    function visible() { if (!document.hidden) { clearTimeout(timer); void load() } }
    setLoading(true); void load(); document.addEventListener('visibilitychange', visible)
    return () => { active = false; controller?.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible) }
  }, [page, status, kind, server, search, after, through, refresh])
  useEffect(() => {
    if (document.activeElement === document.body) ref.current?.focus()
  }, [data, ref])
  useEffect(() => { setSelected([]) }, [page, status, kind, server, search, after, through])
  function reset() { setAfter(''); setThrough(''); setStatus('all'); setKind('all'); setServer('all'); setQuery(''); setSearch(''); setPage(1); setDetail('') }
  function newest() {
    ref.current?.focus()
    snapshotRef.current = ''; setSnapshot(''); setLatest(summary?.latest_created_at); setPage(1); setLoading(true); setRefresh(v => v + 1)
  }
  async function control(id: string, action: 'retry' | 'cancel') {
    setBusy(id); setError('')
    try { await api.controlDatasetOperation(id, action); ref.current?.focus(); onChanged(); setRefresh(v => v + 1) }
    catch (e) { setError(e instanceof Error ? e.message : '操作失败') }
    finally { setBusy('') }
  }
  function askRemoval(ids?: string[]) {
    const cutoff = snapshotRef.current || summary?.snapshot || new Date().toISOString()
    const upper = through ? new Date(Math.min(new Date(cutoff).getTime(), new Date(dayBoundary(through, true)).getTime())).toISOString() : cutoff
    setRemoval({ request: ids ? { ids, before: cutoff } : { status, kind, node_id: server, q: search, before: upper, after: after ? dayBoundary(after) : undefined },
      label: ids ? ids.length === 1 ? '删除这条操作记录' : '删除选中的操作记录' : '清理当前筛选范围内的已结束记录' })
  }
  function removed(count: number) {
    setRemoval(null); setSelected([]); setNotice(`已删除 ${count} 条操作记录，数据集及实际文件保留。`)
    ref.current?.focus(); onChanged(); setRefresh(v => v + 1)
  }
  const hasNew = !!summary?.latest_created_at && summary.latest_created_at !== latest && !!snapshot && new Date(summary.latest_created_at).getTime() > new Date(snapshot).getTime()
  return <div className="dataset-drawer-backdrop" onMouseDown={close}>
    <div ref={ref} className="dataset-history-drawer" role="dialog" aria-modal="true" aria-labelledby="dataset-history-title" tabIndex={-1} onMouseDown={e => e.stopPropagation()}>
      <div className="dataset-history-content" inert={!!removal}><header><div><h2 id="dataset-history-title">数据集操作记录</h2><p>上传、校验与迁移在后台执行，关闭面板不会中断。</p></div><button type="button" className="icon-button" aria-label="关闭操作记录" onClick={close}><X size={20} /></button></header>
      <div className="dataset-history-summary"><span>全部 <b>{summary?.total ?? '—'}</b></span><span>进行中 <b>{summary?.active ?? '—'}</b></span><span>失败 <b>{summary?.failed ?? '—'}</b></span></div>
      <div className="dataset-history-filters">
        <label className="dataset-search"><Search size={16} /><input aria-label="搜索操作记录" placeholder="搜索数据集名称" maxLength={160} value={query} onChange={e => setQuery(e.target.value)} /></label>
        <DatasetServerSelect options={servers} value={server} onChange={v => { setServer(v); setPage(1) }} />
        <div className="dataset-history-filter-actions"><label>状态<select aria-label="按操作状态筛选" value={status} onChange={e => { setStatus(e.target.value); setPage(1) }}><option value="all">全部状态</option><option value="attention">进行中及失败</option><option value="active">进行中</option>{Object.entries(statuses).map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>
        <label>类型<select aria-label="按操作类型筛选" value={kind} onChange={e => { setKind(e.target.value); setPage(1) }}><option value="all">全部类型</option>{Object.entries(kinds).map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>
        <button type="button" className="button compact ghost" onClick={reset}>重置筛选</button>
        <button type="button" className="button compact ghost" onClick={newest}><RefreshCw size={14} />查看最新</button></div>
        <div className="dataset-history-dates"><label>开始日期<input type="date" aria-label="操作记录开始日期" value={after} max={through || undefined} onChange={e => { setAfter(e.target.value); setPage(1) }} /></label><label>结束日期<input type="date" aria-label="操作记录结束日期" value={through} min={after || undefined} onChange={e => { setThrough(e.target.value); setPage(1) }} /></label></div>
        <div className="dataset-history-cleanup"><button type="button" className="button compact ghost" aria-pressed={selectionMode} onClick={() => { setSelectionMode(!selectionMode); setSelected([]) }}>{selectionMode ? '退出多选' : '多选记录'}</button>
          {selectionMode && <><label><input type="checkbox" aria-label="选择本页可删除记录" checked={!!eligibleIds.length && selectedIds.length === eligibleIds.length} disabled={!eligibleIds.length || loading} onChange={e => setSelected(e.target.checked ? eligibleIds : [])} />本页已结束</label><button type="button" className="button compact danger" disabled={!selectedIds.length || loading || !!busy} onClick={() => askRemoval(selectedIds)}>删除选中（{selectedIds.length}）</button></>}
          <button type="button" className="button compact ghost" disabled={!data || !data.total || !!busy || loading || !!error || query !== search} onClick={() => askRemoval()}><Trash2 size={14} />批量清理</button><small>仅已完成、已取消的记录可删除</small>
        </div>
      </div>
      {notice && <div className="dataset-history-notice" role="status">{notice}</div>}
      {hasNew && <button className="dataset-new-records" type="button" onClick={newest}>有新操作记录，点击查看（当前页保持不变）</button>}
      {error && <div className="alert error" role="alert">{error}</div>}
      <div className="dataset-history-body" aria-busy={loading}>
        {loading && !data ? <p className="empty">正在加载操作记录…</p> : data?.items.length ? data.items.map(j => <article className="dataset-history-row" key={j.id}>
          <div className="dataset-history-row-main">{selectionMode && <input type="checkbox" className="dataset-history-checkbox" aria-label={`选择记录：${j.name || j.id}`} disabled={!j.can_delete_history || loading} checked={selectedIds.includes(j.id)} onChange={e => setSelected(current => e.target.checked ? [...current, j.id] : current.filter(id => id !== j.id))} />}
          <button type="button" className="dataset-history-row-toggle" aria-expanded={detail === j.id} aria-controls={`operation-${j.id}`} onClick={() => setDetail(detail === j.id ? '' : j.id)}>
            {detail === j.id ? <ChevronDown size={16} /> : <ChevronRight size={16} />}<span><strong title={j.name}>{j.name || '保存位置检查'}</strong><small>{kinds[j.kind] || j.kind} · {j.source_node_name || j.node_name || j.node_id}{j.target_node && j.target_node !== j.source_node ? ` → ${j.target_node_name || j.target_node}` : ''}</small></span>
            <span className={`dataset-job-state ${j.status}`}>{statuses[j.status] || j.status}</span><time>{new Date(j.created_at).toLocaleString('zh-CN')}</time>
          </button><button type="button" className="dataset-history-delete" aria-label={`删除记录：${j.name || j.id}`} title={j.can_delete_history ? '删除操作记录，保留数据集文件' : '仅已完成或已取消的记录可删除；失败操作请先重试或取消'} disabled={!j.can_delete_history || !!busy || loading} onClick={() => askRemoval([j.id])}><Trash2 size={15} /></button></div>
          <DatasetOperationProgress operation={j} />
          {detail === j.id && <div id={`operation-${j.id}`} className="dataset-history-detail">
            <DatasetOperationProgress operation={j} detailed />
            <small>更新于 {new Date(j.updated_at).toLocaleString('zh-CN')} · 编号 {j.id}</small>
            {(j.source_path || j.destination) && <code>{j.source_path || '浏览器上传'}{j.destination ? ` → ${j.destination}` : ''}</code>}
            {j.phase === 'discover' && <p>已检查 {j.result.visited || 0} 个目录 · 发现 {j.result.found || 0} 个候选{j.result.limited ? ` · 扫描达到限制：${j.result.limit_reason}` : ''}。候选与批次详情请打开“扫描已有数据集”。</p>}
            {j.transferred > 0 && <p>已传 {formatBytes(j.transferred)}{j.total_bytes ? ` / ${formatBytes(j.total_bytes)}` : ''}</p>}
            {j.error && <p role="alert" className="dataset-operation-error">{j.error}</p>}
            {j.result.directory && <code>目标已保存：{j.result.directory}</code>}
            <div className="dataset-inline-actions">{j.status === 'failed' && <button type="button" className="button compact" disabled={!!busy} onClick={() => void control(j.id, 'retry')}>{busy === j.id ? '正在提交…' : '重试当前阶段'}</button>}{j.can_cancel && <button type="button" className="button compact ghost" disabled={!!busy} onClick={() => void control(j.id, 'cancel')}>{busy === j.id ? '正在提交…' : '取消操作'}</button>}</div>
          </div>}
        </article>) : <div className="empty"><strong>没有符合条件的记录</strong><button type="button" className="button ghost" onClick={reset}>重置筛选</button></div>}
      </div>
      <footer><span>共 {data?.total ?? 0} 条 · 每页 20 条</span><button type="button" className="button compact ghost" disabled={page <= 1} onClick={() => { setLoading(true); setPage(v => v - 1) }}>上一页</button><span>{page} / {Math.max(1, Math.ceil((data?.total || 0) / 20))}</span><button type="button" className="button compact ghost" disabled={!data || page * 20 >= data.total} onClick={() => { setLoading(true); setPage(v => v + 1) }}>下一页</button></footer>
    </div>
    {removal && <HistoryRemovalConfirmation target={removal} onClose={() => setRemoval(null)} onRemoved={removed} onBusy={setRemoving} />}
    </div>
  </div>
}

function dayBoundary(value: string, end = false) {
  const date = new Date(`${value}T00:00:00`)
  if (end) { date.setDate(date.getDate() + 1); date.setMilliseconds(-1) }
  return date.toISOString()
}

function HistoryRemovalConfirmation({ target, onClose, onRemoved, onBusy }: {
  target: { request: DatasetHistoryRemoval; label: string }; onClose: () => void
  onRemoved: (count: number) => void; onBusy: (busy: boolean) => void
}) {
  const [preview, setPreview] = useState<DatasetHistoryRemovalPreview | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [refresh, setRefresh] = useState(0)
  const [notice, setNotice] = useState('')
  const dialog = useModalFocus<HTMLDivElement>(true, onClose, { canClose: !busy, initialFocusSelector: '[data-cancel]' })
  useEffect(() => {
    const controller = new AbortController()
    void api.previewDatasetHistoryRemoval(target.request, controller.signal).then(result => {
      if (!controller.signal.aborted) setPreview(result)
    }).catch((e: Error) => { if (!controller.signal.aborted) setError(e.message) })
    return () => controller.abort()
  }, [target, refresh])
  async function submit() {
    if (!preview?.count || busy) return
    setBusy(true); onBusy(true); setError('')
    try {
      const result = await api.removeDatasetHistory({ ...target.request, fingerprint: preview.fingerprint })
      onRemoved(result.removed)
    } catch (e) {
      setError(e instanceof Error ? e.message : '删除记录失败')
      setPreview(null)
    } finally { setBusy(false); onBusy(false) }
  }
  return <div className="dataset-history-confirm-backdrop" onMouseDown={e => { e.stopPropagation(); if (!busy) onClose() }}>
    <div ref={dialog} className="dataset-history-confirm" role="alertdialog" aria-modal="true" aria-labelledby="history-removal-title" aria-describedby="history-removal-description" tabIndex={-1} onMouseDown={e => e.stopPropagation()}>
      <h3 id="history-removal-title">{target.label}</h3>
      <p id="history-removal-description">仅从操作历史移除记录，不删除数据集、保存位置或训练文件。删除后这些记录不再显示。</p>
      <p role="status">{preview ? preview.count ? `本次将删除 ${preview.count} 条已完成或已取消的记录。` : '当前范围内没有可删除的已完成或已取消记录。' : error ? '请重新预览后确认。' : '正在核对可删除记录数量…'}</p>
      {notice && <p role="status">{notice}</p>}{error && <p className="dataset-operation-error" role="alert">{error}</p>}
      <footer><button data-cancel type="button" className="button ghost" disabled={busy} onClick={onClose}>取消</button>{!preview && error && <button type="button" className="button ghost" onClick={() => { setError(''); setNotice('已重新核对，请确认新的数量。'); setRefresh(v => v + 1) }}>重新预览</button>}<button type="button" className="button danger" disabled={!preview?.count || busy} onClick={() => void submit()}>{busy ? '正在删除记录…' : `确认删除${preview ? `（${preview.count}）` : ''}`}</button></footer>
    </div>
  </div>
}
