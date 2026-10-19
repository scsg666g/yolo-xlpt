import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, Clipboard, Download, Search, Terminal, X } from 'lucide-react'
import type { TrainingTask } from '../types'
import { useModalFocus } from '../useModalFocus'

interface Props {
  task: TrainingTask | null
  lines: string[]
  open: boolean
  onClose: () => void
}

function lineClass(line: string): string {
  const normalized = line.toLowerCase()
  if (normalized.includes('error') || normalized.includes('traceback') || normalized.includes('failed')) return 'error-line'
  if (normalized.includes('warning') || normalized.includes('warn')) return 'warning-line'
  if (normalized.includes('epoch') || normalized.includes('map50') || normalized.includes('train/')) return 'metric-line'
  return ''
}

export function LogViewerModal({ task, lines, open, onClose }: Props) {
  const [search, setSearch] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const [copied, setCopied] = useState(false)
  const [actionMessage, setActionMessage] = useState('')
  const outputRef = useRef<HTMLDivElement>(null)
  const dialogRef = useModalFocus<HTMLElement>(open && Boolean(task), onClose, {
    initialFocusSelector: '[data-modal-initial-focus]',
  })
  const visibleLines = useMemo(() => search ? lines.filter((line) => line.toLowerCase().includes(search.toLowerCase())) : lines, [lines, search])

  useEffect(() => {
    if (open && autoScroll && outputRef.current) outputRef.current.scrollTop = outputRef.current.scrollHeight
  }, [open, autoScroll, visibleLines])

  useEffect(() => { if (open) { setSearch(''); setCopied(false); setActionMessage('') } }, [open])
  if (!open || !task) return null

  async function copyLogs() {
    const text = lines.join('\n')
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text)
      } else {
        const textarea = document.createElement('textarea')
        textarea.value = text
        textarea.setAttribute('readonly', '')
        textarea.style.position = 'fixed'
        textarea.style.opacity = '0'
        document.body.appendChild(textarea)
        try {
          textarea.select()
          if (!document.execCommand('copy')) throw new Error('浏览器拒绝了复制操作')
        } finally {
          textarea.remove()
        }
      }
      setCopied(true); setActionMessage('日志已复制到剪贴板')
      window.setTimeout(() => { setCopied(false); setActionMessage('') }, 1600)
    } catch (reason) {
      setCopied(false)
      setActionMessage(reason instanceof Error ? `${reason.message}，请手动选择日志文本复制` : '复制失败，请手动选择日志文本复制')
    }
  }

  function downloadLogs() {
    if (!task) return
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${task.name.replace(/[\\/:*?"<>|]+/g, '_') || 'training'}-training.log`
    anchor.style.display = 'none'
    document.body.appendChild(anchor)
    anchor.click()
    window.setTimeout(() => { anchor.remove(); URL.revokeObjectURL(url) }, 1000)
    setActionMessage('日志下载已开始')
    window.setTimeout(() => setActionMessage(''), 1600)
  }

  return <div className="modal-backdrop log-backdrop" onMouseDown={onClose}>
    <section ref={dialogRef} className="log-modal" role="dialog" aria-modal="true" aria-labelledby="log-viewer-title" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
      <header>
        <div className="log-modal-title"><div className="terminal-mark"><Terminal size={18} /></div><div><span>LIVE TRAINING OUTPUT</span><h2 id="log-viewer-title">实时训练日志</h2><p>{task.name} · {task.log_path || '等待创建日志文件'}</p></div></div>
        <div className="log-header-actions"><span className={`log-live ${task.status === 'running' ? 'active' : ''}`}><i />{task.status === 'running' ? '实时更新' : '日志快照'}</span><button className="icon-button" data-modal-initial-focus aria-label="关闭日志查看器" onClick={onClose}><X size={20} /></button></div>
      </header>
      <div className="log-tool-row">
        <div className="log-search"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索日志内容" />{search && <span>{visibleLines.length} 条匹配</span>}</div>
        <label className="auto-scroll"><input type="checkbox" checked={autoScroll} onChange={(event) => setAutoScroll(event.target.checked)} />自动滚动</label>
        <button className="log-action-button" title="复制当前日志内容" aria-label="复制当前日志内容" onClick={copyLogs}>{copied ? <Check size={15} /> : <Clipboard size={15} />}{copied ? '已复制' : '复制全部'}</button>
        <button className="log-action-button" title="下载当前日志文件" aria-label="下载当前日志文件" onClick={downloadLogs}><Download size={15} />下载日志</button>
        {actionMessage && <span className="log-action-message" role="status">{actionMessage}</span>}
      </div>
      <div className="terminal-window">
        <div className="terminal-chrome"><span className="red" /><span className="yellow" /><span className="green" /><code>yolo://training/{task.id}</code><em>{visibleLines.length} lines</em></div>
        <div className="terminal-output" ref={outputRef}>
          {visibleLines.length ? visibleLines.map((line, index) => <div className={lineClass(line)} key={`${index}-${line.slice(0, 20)}`}><span>{String(index + 1).padStart(4, '0')}</span><code>{line || ' '}</code></div>) : <div className="terminal-placeholder"><Terminal size={25} /><strong>{search ? '没有匹配的日志' : '等待训练日志'}</strong><small>{search ? '请更换搜索关键词' : '任务启动后，输出将在这里实时显示'}</small></div>}
        </div>
      </div>
      <footer><span>日志每 3 秒自动同步 · 当前显示最近 {lines.length} 行</span><button className="button primary" onClick={onClose}>关闭查看器</button></footer>
    </section>
  </div>
}
