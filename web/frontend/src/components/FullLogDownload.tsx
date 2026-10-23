import { useEffect, useRef, useState } from 'react'
import { Download, LoaderCircle } from 'lucide-react'
import { api } from '../api'

export function FullLogDownload({ taskId, runId }: { taskId: string; runId: string | null | undefined }) {
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const request = useRef<AbortController | null>(null)
  const timer = useRef<number | undefined>(undefined)
  const version = useRef(0)
  useEffect(() => {
    setBusy(false); setMessage('')
    return () => { request.current?.abort(); window.clearTimeout(timer.current) }
  }, [taskId, runId])

  async function prepare(refreshSnapshot = false) {
    if (!runId || busy) return
    const attempt = ++version.current
    const controller = new AbortController()
    request.current?.abort(); request.current = controller
    setBusy(true); setMessage('正在准备完整日志…')
    const started = Date.now()
    async function poll(refresh = false) {
      try {
        const result = await api.prepareRunLog(taskId, runId!, refresh, controller.signal)
        if (controller.signal.aborted || attempt !== version.current) return
        if (result.status === 'ready') {
          const link = document.createElement('a')
          link.href = `/api/tasks/${encodeURIComponent(taskId)}/runs/${encodeURIComponent(runId!)}/logs/download`
          link.download = ''; document.body.appendChild(link); link.click(); link.remove()
          setBusy(false); setMessage(result.snapshot_at
            ? `日志快照下载已开始（${new Date(result.snapshot_at).toLocaleString('zh-CN')}），需要最新内容时可重新获取`
            : '完整日志下载已开始；训练中的日志以本次获取时刻为准')
        } else if (result.status === 'failed') {
          setBusy(false); setMessage(result.message || '日志获取失败，请重试')
        } else if (Date.now() - started > 120_000) {
          setBusy(false); setMessage('远程日志仍在准备，可稍后再次点击查看；后台继续处理')
        } else {
          setMessage(result.message || '等待远程服务器传输日志…')
          timer.current = window.setTimeout(() => void poll(), 2000)
        }
      } catch (error) {
        if (!controller.signal.aborted && attempt === version.current) {
          setBusy(false); setMessage(error instanceof Error ? error.message : '日志下载失败')
        }
      }
    }
    await poll(refreshSnapshot)
  }

  return <span className="full-log-download">
    <button type="button" className="button ghost compact" disabled={!runId || busy} onClick={() => void prepare()}>
      {busy ? <LoaderCircle size={15} className="spin" /> : <Download size={15} />}{busy ? '准备日志中' : '下载完整日志'}
    </button>
    {message && !busy && <button type="button" className="button ghost compact" onClick={() => void prepare(true)}>重新获取</button>}
    {message && <small role="status">{message}</small>}
  </span>
}
