import { useEffect, useRef, useState } from 'react'
import { LoaderCircle, X } from 'lucide-react'
import { api } from '../api'
import type { TrainingTask } from '../types'

interface Props {
  task: TrainingTask
  onClose: () => void
  onSaved: (task: TrainingTask) => void
}

export function RenameTaskModal({ task, onClose, onSaved }: Props) {
  const [name, setName] = useState(task.name)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const input = useRef<HTMLInputElement>(null)
  const submitting = useRef(false)

  useEffect(() => {
    const previous = document.activeElement
    input.current?.focus()
    input.current?.select()
    return () => { if (previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [])

  async function save(event: React.FormEvent) {
    event.preventDefault()
    if (submitting.current) return
    const trimmed = name.trim()
    if (!trimmed) { setError('请输入任务名称'); return }
    if (Array.from(trimmed).length > 160) { setError('任务名称最多 160 个字符'); return }
    submitting.current = true
    setSaving(true); setError('')
    try {
      onSaved(await api.rename(task.id, trimmed))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '重命名失败，请重试')
    } finally { submitting.current = false; setSaving(false) }
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLFormElement>) {
    if (event.nativeEvent.isComposing) {
      if (event.key === 'Enter') event.preventDefault()
      return
    }
    if (event.key === 'Escape') {
      event.preventDefault()
      if (!submitting.current) onClose()
    }
    if (event.key === 'Tab') {
      const controls = event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled)')
      const first = controls[0], last = controls[controls.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
    }
  }

  return <div className="modal-backdrop" onMouseDown={() => { if (!submitting.current) onClose() }}>
    <form className="modal rename-task-modal" role="dialog" aria-modal="true" aria-labelledby="rename-task-title" onSubmit={save} onKeyDown={handleKeyDown} onMouseDown={(event) => event.stopPropagation()}>
      <header><h2 id="rename-task-title">重命名任务</h2><button type="button" className="icon-button" aria-label="关闭重命名窗口" disabled={saving} onClick={onClose}><X size={18} /></button></header>
      <div className="modal-body">
        <div className="form-grid"><label className="span-2" htmlFor="rename-task-name">任务名称<input ref={input} id="rename-task-name" value={name} disabled={saving} aria-invalid={Boolean(error)} aria-describedby={error ? 'rename-task-error' : undefined} onChange={(event) => { setName(event.target.value); setError('') }} /></label></div>
        {error && <div id="rename-task-error" className="alert error rename-task-error" role="alert">{error}</div>}
      </div>
      <footer><button type="button" className="button ghost" disabled={saving} onClick={onClose}>取消</button><button type="submit" className="button primary" disabled={saving || !name.trim()}>{saving && <LoaderCircle size={15} className="spin" />}{saving ? '保存中…' : '保存'}</button></footer>
    </form>
  </div>
}
