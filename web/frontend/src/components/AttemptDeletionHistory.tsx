import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronDown, History, LoaderCircle, RefreshCw, Search, X } from 'lucide-react'
import { api } from '../api'
import type { AttemptDeletionJob } from '../types'
import { useModalFocus } from '../useModalFocus'

const pageSize = 12
type HistoryFilter = 'all' | 'active' | 'attention' | 'completed'

function statusLabel(job: AttemptDeletionJob) {
  if (job.status === 'succeeded') return '已完成'
  if (job.phase === 'workspace_busy' && job.status === 'failed') return '工作区占用'
  if (job.status === 'failed') return '清理失败'
  if (job.status === 'cancelled') return '已取消'
  if (job.phase === 'waiting_agent') return '等待 Agent'
  return '清理中'
}

function statusTone(job: AttemptDeletionJob) {
  if (job.status === 'succeeded') return 'completed'
  if (job.phase === 'workspace_busy' && job.status === 'failed') return 'workspace-busy'
  if (job.status === 'failed') return 'failed'
  if (job.status === 'cancelled') return 'cancelled'
  return 'active'
}

function dateLabel(value: string) { return new Date(value).toLocaleString('zh-CN') }

export function AttemptDeletionHistory({ taskId }: { taskId?: string }) {
  const [open, setOpen] = useState(false)
  return <>
    <button type="button" className="button ghost compact" onClick={() => setOpen(true)}><History size={15} />训练清理记录</button>
    {open && <HistoryModal taskId={taskId} onClose={() => setOpen(false)} />}
  </>
}

function HistoryModal({ taskId, onClose }: { taskId?: string; onClose: () => void }) {
  const [jobs, setJobs] = useState<AttemptDeletionJob[]>([])
  const [total, setTotal] = useState(0)
  const [query, setQuery] = useState('')
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<HistoryFilter>('all')
  const [page, setPage] = useState(1)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [retrying, setRetrying] = useState<string | null>(null)
  const controller = useRef<AbortController | null>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const dialogRef = useModalFocus<HTMLDivElement>(true, onClose)

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (query.trim() !== search) { setLoading(true); setJobs([]); setSearch(query.trim()); setPage(1) }
    }, 250)
    return () => window.clearTimeout(timer)
  }, [query, search])

  const load = useCallback(async () => {
    controller.current?.abort()
    const request = new AbortController(); controller.current = request
    const params = new URLSearchParams({ limit: String(pageSize), offset: String((page - 1) * pageSize), status: filter })
    if (taskId) params.set('task_id', taskId)
    if (search) params.set('search', search)
    try {
      const result = await api.attemptDeletionPage(params, request.signal)
      if (!request.signal.aborted) {
        setJobs(result.items)
        setTotal(result.total)
        setError('')
        if (page > 1 && !result.items.length && result.total) setPage(Math.max(1, Math.ceil(result.total / pageSize)))
      }
      return result.items.some((job) => ['pending', 'running', 'finalizing'].includes(job.status))
    } catch (reason) {
      if (!request.signal.aborted) setError(reason instanceof Error ? reason.message : '清理记录读取失败')
      return false
    } finally { if (!request.signal.aborted) setLoading(false) }
  }, [filter, page, search, taskId])

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    async function poll() {
      if (document.hidden) return
      const active = await load()
      if (!cancelled && !document.hidden) timer = window.setTimeout(poll, active ? 3000 : 30_000)
    }
    function visibilityChanged() {
      window.clearTimeout(timer)
      if (document.hidden) controller.current?.abort()
      else void poll()
    }
    void poll()
    document.addEventListener('visibilitychange', visibilityChanged)
    return () => { cancelled = true; controller.current?.abort(); window.clearTimeout(timer); document.removeEventListener('visibilitychange', visibilityChanged) }
  }, [load])

  async function retry(job: AttemptDeletionJob) {
    setRetrying(job.id)
    try { await api.deleteRun(job.task_id, job.run_id); await load() }
    catch (reason) { setError(reason instanceof Error ? reason.message : '重试失败') }
    finally { setRetrying(null) }
  }

  function changePage(next: number) {
    setLoading(true)
    setJobs([])
    setPage(next)
    listRef.current?.scrollTo({ top: 0 })
  }
  const pageCount = Math.max(1, Math.ceil(total / pageSize))

  return <div className="modal-backdrop" onMouseDown={onClose}><div ref={dialogRef} className="modal trash-modal attempt-history-modal" role="dialog" aria-modal="true" aria-labelledby="attempt-cleanup-title" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
    <header><div><h2 id="attempt-cleanup-title">训练清理记录</h2><p>按任务、训练次数或服务器查找，点击记录查看原因与回执。</p></div><button type="button" className="icon-button" aria-label="关闭清理记录" onClick={onClose}><X size={18} /></button></header>
    <div className="attempt-history-tools">
      <div className="attempt-history-filters" role="group" aria-label="筛选训练清理状态">
        {([['all', '全部'], ['active', '清理中'], ['attention', '需关注'], ['completed', '已完成']] as const).map(([value, label]) => <button key={value} type="button" className={filter === value ? 'active' : ''} aria-pressed={filter === value} onClick={() => { if (filter !== value) { setFilter(value); changePage(1) } }}>{label}</button>)}
      </div>
      <div className="attempt-history-search-row"><label className="trash-browser-search"><Search size={15} aria-hidden="true" /><input type="search" aria-label="搜索训练清理记录" placeholder="搜索任务、Attempt 编号或服务器" maxLength={100} value={query} onChange={(event) => setQuery(event.target.value)} /></label><span>{loading ? '正在更新…' : `共 ${total} 条`}</span></div>
    </div>
    <div ref={listRef} className="trash-list attempt-history-list" aria-busy={loading}>
      {error && <p className="alert error" role="alert">{error}</p>}
      {loading && !jobs.length && <p role="status">正在加载清理记录…</p>}
      {!loading && !jobs.length && <div className="trash-browser-empty">{query.trim() || filter !== 'all' ? '没有匹配的清理记录，请调整搜索或筛选条件。' : '暂无单次训练清理记录'}</div>}
      {jobs.map((job) => <article className={`attempt-history-item ${statusTone(job)}`} key={job.id}>
        <details className="attempt-history-details"><summary>
          <span className="attempt-history-primary"><strong title={job.task_name}>{job.task_name}</strong><span>Attempt {job.attempt_number != null ? String(job.attempt_number).padStart(3, '0') : '—'}</span><span className={`deletion-status ${statusTone(job)}`}>{statusLabel(job)}</span></span>
          <span className="attempt-history-secondary"><span title={job.node_name || undefined}>{job.executor === 'remote' ? job.node_name || '远程服务器' : '本机服务器'}</span><time dateTime={job.updated_at} title={`更新于 ${dateLabel(job.updated_at)}`}>{dateLabel(job.updated_at)}</time></span><ChevronDown size={16} aria-hidden="true" />
        </summary><div className="attempt-history-expanded">
          <p>{job.message}</p>
          {job.error && job.error !== job.message && <p className="attempt-history-error">{job.error}</p>}
          <small>提交：{dateLabel(job.created_at)} · 更新：{dateLabel(job.updated_at)}</small>
          {job.receipt && <details className="deletion-receipt"><summary>查看清理回执</summary><pre className="cleanup-receipt">{JSON.stringify(job.receipt, null, 2)}</pre></details>}
          {job.retryable && <button type="button" className="button ghost compact" disabled={retrying !== null} onClick={() => void retry(job)}>{retrying === job.id ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}重试清理</button>}
        </div></details>
      </article>)}
    </div>
    <nav className="trash-pagination" aria-label="训练清理记录分页"><span>第 {Math.min(page, pageCount)} / {pageCount} 页 · 共 {total} 条</span><div><button type="button" className="button ghost compact" disabled={page <= 1 || loading} onClick={() => changePage(page - 1)}>上一页</button><button type="button" className="button ghost compact" disabled={page >= pageCount || loading} onClick={() => changePage(page + 1)}>下一页</button></div></nav>
    <footer><button type="button" className="button ghost" onClick={() => void load()}><RefreshCw size={15} />刷新</button><button type="button" className="button primary" onClick={onClose}>关闭</button></footer>
  </div></div>
}
