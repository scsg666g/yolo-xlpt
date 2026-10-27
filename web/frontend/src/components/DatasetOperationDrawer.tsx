import { useEffect, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, RefreshCw, Search, X } from 'lucide-react'
import { api, formatBytes } from '../api'
import type { DatasetOperationPage, DatasetOperationSummary } from '../types'
import { datasetPhaseLabel } from '../datasetUploadSession'
import { useModalFocus } from '../useModalFocus'
import { DatasetServerSelect, type DatasetServerOption } from './DatasetServerSelect'

const statuses: Record<string, string> = { pending: '等待处理', running: '正在处理', committing: '提交文件中', completed: '已完成', failed: '失败', cancelled: '已取消' }
const kinds: Record<string, string> = { check: '目录检查', upload: '上传', register: '登记', migrate: '复制 / 迁移', delete: '永久删除' }

export function DatasetOperationDrawer({ onClose, onChanged, summary, servers }: {
  onClose: () => void; onChanged: () => void; summary: DatasetOperationSummary | null; servers: DatasetServerOption[]
}) {
  const ref = useModalFocus<HTMLDivElement>(true, onClose)
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
      try {
        const result = await api.datasetOperationPage(params, controller.signal)
        if (active) {
          if (!snapshotRef.current) { snapshotRef.current = result.snapshot; setSnapshot(result.snapshot) }
          setData(result); setError(''); setLoading(false)
          if (result.page !== page) setPage(result.page)
        }
      } catch (e) { if (active && !controller.signal.aborted) { setError(e instanceof Error ? e.message : '操作记录加载失败'); setLoading(false) } }
      finally { running = false; if (active) timer = setTimeout(load, 5000) }
    }
    function visible() { if (!document.hidden) { clearTimeout(timer); void load() } }
    void load(); document.addEventListener('visibilitychange', visible)
    return () => { active = false; controller?.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible) }
  }, [page, status, kind, server, search, refresh])
  useEffect(() => {
    if (document.activeElement === document.body) ref.current?.focus()
  }, [data, ref])
  function reset() { setStatus('all'); setKind('all'); setServer('all'); setQuery(''); setSearch(''); setPage(1); setDetail('') }
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
  const hasNew = !!summary?.latest_created_at && summary.latest_created_at !== latest && !!snapshot && new Date(summary.latest_created_at).getTime() > new Date(snapshot).getTime()
  return <div className="dataset-drawer-backdrop" onMouseDown={onClose}>
    <div ref={ref} className="dataset-history-drawer" role="dialog" aria-modal="true" aria-labelledby="dataset-history-title" tabIndex={-1} onMouseDown={e => e.stopPropagation()}>
      <header><div><h2 id="dataset-history-title">数据集操作记录</h2><p>上传、校验与迁移在后台执行，关闭面板不会中断。</p></div><button type="button" className="icon-button" aria-label="关闭操作记录" onClick={onClose}><X size={20} /></button></header>
      <div className="dataset-history-summary"><span>全部 <b>{summary?.total ?? '—'}</b></span><span>进行中 <b>{summary?.active ?? '—'}</b></span><span>失败 <b>{summary?.failed ?? '—'}</b></span></div>
      <div className="dataset-history-filters">
        <label className="dataset-search"><Search size={16} /><input aria-label="搜索操作记录" placeholder="搜索数据集名称" maxLength={160} value={query} onChange={e => setQuery(e.target.value)} /></label>
        <DatasetServerSelect options={servers} value={server} onChange={v => { setServer(v); setPage(1) }} />
        <div className="dataset-history-filter-actions"><label>状态<select aria-label="按操作状态筛选" value={status} onChange={e => { setStatus(e.target.value); setPage(1) }}><option value="all">全部状态</option><option value="attention">进行中及失败</option><option value="active">进行中</option>{Object.entries(statuses).map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>
        <label>类型<select aria-label="按操作类型筛选" value={kind} onChange={e => { setKind(e.target.value); setPage(1) }}><option value="all">全部类型</option>{Object.entries(kinds).map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></label>
        <button type="button" className="button compact ghost" onClick={reset}>重置筛选</button>
        <button type="button" className="button compact ghost" onClick={newest}><RefreshCw size={14} />查看最新</button></div>
      </div>
      {hasNew && <button className="dataset-new-records" type="button" onClick={newest}>有新操作记录，点击查看（当前页保持不变）</button>}
      {error && <div className="alert error" role="alert">{error}</div>}
      <div className="dataset-history-body" aria-busy={loading}>
        {loading && !data ? <p className="empty">正在加载操作记录…</p> : data?.items.length ? data.items.map(j => <article className="dataset-history-row" key={j.id}>
          <button type="button" className="dataset-history-row-toggle" aria-expanded={detail === j.id} aria-controls={`operation-${j.id}`} onClick={() => setDetail(detail === j.id ? '' : j.id)}>
            {detail === j.id ? <ChevronDown size={16} /> : <ChevronRight size={16} />}<span><strong title={j.name}>{j.name || '保存位置检查'}</strong><small>{kinds[j.kind] || j.kind} · {j.source_node_name || j.node_name || j.node_id}{j.target_node && j.target_node !== j.source_node ? ` → ${j.target_node_name || j.target_node}` : ''}</small></span>
            <span className={`dataset-job-state ${j.status}`}>{statuses[j.status] || j.status}</span><time>{new Date(j.created_at).toLocaleString('zh-CN')}</time>
          </button>
          {detail === j.id && <div id={`operation-${j.id}`} className="dataset-history-detail">
            <p>{j.waiting_offline ? '等待远程节点上线' : j.waiting_training ? '等待训练任务结束' : datasetPhaseLabel[j.phase] || j.phase}</p>
            <small>更新于 {new Date(j.updated_at).toLocaleString('zh-CN')} · 编号 {j.id}</small>
            {(j.source_path || j.destination) && <code>{j.source_path || '浏览器上传'}{j.destination ? ` → ${j.destination}` : ''}</code>}
            {j.transferred > 0 && <p>已传 {formatBytes(j.transferred)}{j.total_bytes ? ` / ${formatBytes(j.total_bytes)}` : ''}</p>}
            {j.error && <p role="alert" className="dataset-operation-error">{j.error}</p>}
            {j.result.directory && <code>目标已保存：{j.result.directory}</code>}
            <div className="dataset-inline-actions">{j.status === 'failed' && <button type="button" className="button compact" disabled={!!busy} onClick={() => void control(j.id, 'retry')}>{busy === j.id ? '正在提交…' : '重试当前阶段'}</button>}{j.can_cancel && <button type="button" className="button compact ghost" disabled={!!busy} onClick={() => void control(j.id, 'cancel')}>{busy === j.id ? '正在提交…' : '取消操作'}</button>}</div>
          </div>}
        </article>) : <div className="empty"><strong>没有符合条件的记录</strong><button type="button" className="button ghost" onClick={reset}>重置筛选</button></div>}
      </div>
      <footer><span>共 {data?.total ?? 0} 条 · 每页 20 条</span><button type="button" className="button compact ghost" disabled={page <= 1} onClick={() => { setLoading(true); setPage(v => v - 1) }}>上一页</button><span>{page} / {Math.max(1, Math.ceil((data?.total || 0) / 20))}</span><button type="button" className="button compact ghost" disabled={!data || page * 20 >= data.total} onClick={() => { setLoading(true); setPage(v => v + 1) }}>下一页</button></footer>
    </div>
  </div>
}
