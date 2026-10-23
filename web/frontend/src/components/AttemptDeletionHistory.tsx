import { useCallback, useEffect, useRef, useState } from 'react'
import { History, LoaderCircle, RefreshCw, X } from 'lucide-react'
import { api } from '../api'
import type { AttemptDeletionJob } from '../types'
import { useModalFocus } from '../useModalFocus'

export function AttemptDeletionHistory({ taskId }: { taskId?: string }) {
  const [open, setOpen] = useState(false)
  return <>
    <button type="button" className="button ghost compact" onClick={() => setOpen(true)}><History size={15} />训练清理记录</button>
    {open && <HistoryModal taskId={taskId} onClose={() => setOpen(false)} />}
  </>
}

function HistoryModal({ taskId, onClose }: { taskId?: string; onClose: () => void }) {
  const [jobs, setJobs] = useState<AttemptDeletionJob[]>([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [retrying, setRetrying] = useState<string | null>(null)
  const controller = useRef<AbortController | null>(null)
  const dialogRef = useModalFocus<HTMLDivElement>(true, onClose)
  const load = useCallback(async () => {
    controller.current?.abort()
    const request = new AbortController(); controller.current = request
    try {
      const result = await api.attemptDeletionJobs(request.signal, taskId)
      if (!request.signal.aborted) { setJobs(result); setError('') }
      return result.some((job) => ['pending', 'running', 'finalizing'].includes(job.status))
    } catch (reason) {
      if (!request.signal.aborted) setError(reason instanceof Error ? reason.message : '清理记录读取失败')
      return false
    } finally { if (!request.signal.aborted) setLoading(false) }
  }, [taskId])
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
  const visible = jobs.filter((job) => !taskId || job.task_id === taskId)
  return <div className="modal-backdrop" onMouseDown={onClose}><div ref={dialogRef} className="modal trash-modal" role="dialog" aria-modal="true" aria-labelledby="attempt-cleanup-title" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
    <header><div><h2 id="attempt-cleanup-title">训练清理记录</h2><p>单次训练的本机与远程清理进度、失败原因和完成回执。</p></div><button type="button" className="icon-button" aria-label="关闭清理记录" onClick={onClose}><X size={18} /></button></header>
    <div className="trash-list">
      {error && <p className="alert error" role="alert">{error}</p>}
      {loading && <p role="status">正在加载清理记录…</p>}
      {!loading && !visible.length && <div className="empty">暂无单次训练清理记录</div>}
      {visible.map((job) => <article className={`trash-item deletion-history ${job.status === 'succeeded' ? 'completed' : job.status === 'failed' ? 'failed' : 'active'}`} key={job.id}>
        <div className="trash-item-copy"><strong>{job.task_name} · Attempt {String(job.attempt_number).padStart(3, '0')}</strong>
          <p>{job.executor === 'remote' ? job.node_name || '远程服务器' : '本机服务器'} · {job.message}</p>
          {job.error && <p className="alert error">{job.error}</p>}
          <small>提交：{new Date(job.created_at).toLocaleString('zh-CN')} · 更新：{new Date(job.updated_at).toLocaleString('zh-CN')}</small>
          {job.receipt && <details><summary>查看清理回执</summary><pre className="cleanup-receipt">{JSON.stringify(job.receipt, null, 2)}</pre></details>}
        </div>
        {job.retryable && <button type="button" className="button ghost compact" disabled={retrying !== null} onClick={() => void retry(job)}>{retrying === job.id ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}重试清理</button>}
      </article>)}
    </div><footer><button type="button" className="button ghost" onClick={() => void load()}><RefreshCw size={15} />刷新</button><button type="button" className="button primary" onClick={onClose}>关闭</button></footer>
  </div></div>
}
