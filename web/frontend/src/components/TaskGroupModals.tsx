import { useEffect, useRef, useState } from 'react'
import { FolderInput, LoaderCircle, Trash2, X } from 'lucide-react'
import { api } from '../api'
import type { TaskGroup, TrainingTask } from '../types'
import { useModalFocus } from '../useModalFocus'

interface GroupEditorProps {
  group: TaskGroup | null
  taskIds?: string[]
  initialDelete?: boolean
  onClose: () => void
  onSaved: (group: TaskGroup, assignedTaskIds: string[]) => void
  onDeleted: (group: TaskGroup) => void
}

export function TaskGroupModal({ group, taskIds = [], initialDelete = false, onClose, onSaved, onDeleted }: GroupEditorProps) {
  const [name, setName] = useState(group?.name || '')
  const [saving, setSaving] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(initialDelete)
  const [error, setError] = useState('')
  const submitting = useRef(false)
  const dialogRef = useModalFocus<HTMLFormElement>(true, onClose, {
    canClose: !saving,
    initialFocusSelector: '[data-modal-initial-focus]',
  })

  useEffect(() => {
    setName(group?.name || '')
    setConfirmDelete(initialDelete)
    setError('')
  }, [group, initialDelete])

  async function save(event: React.FormEvent) {
    event.preventDefault()
    const trimmed = name.trim()
    if (!trimmed || submitting.current) return
    if (Array.from(trimmed).length > 160) { setError('分组名称最多 160 个字符'); return }
    submitting.current = true
    setSaving(true); setError('')
    try {
      const saved = group
        ? await api.renameTaskGroup(group.id, trimmed)
        : taskIds.length
          ? await api.createTaskGroupFromTasks(trimmed, taskIds)
          : await api.createTaskGroup(trimmed)
      onSaved(saved, taskIds)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保存分组失败')
    } finally {
      submitting.current = false
      setSaving(false)
    }
  }

  async function remove() {
    if (!group || submitting.current) return
    submitting.current = true
    setSaving(true); setError('')
    try {
      await api.deleteTaskGroup(group.id)
      onDeleted(group)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '删除分组失败')
    } finally {
      submitting.current = false
      setSaving(false)
    }
  }

  const converting = !group && taskIds.length > 0
  const title = confirmDelete ? '删除训练分组' : group ? '重命名训练分组' : converting ? '命名未分组任务' : '新建训练分组'

  return <div className="modal-backdrop" onMouseDown={() => !saving && onClose()}>
    <form ref={dialogRef} className="modal task-group-modal" role="dialog" aria-modal="true" aria-labelledby="task-group-modal-title" tabIndex={-1} onSubmit={save} onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="eyebrow">TRAINING GROUP</span><h2 id="task-group-modal-title">{title}</h2><p>{converting ? `将当前 ${taskIds.length} 个未分组任务整理到新分组。` : '用于整理同一模型的多轮实验。'}</p></div><button type="button" className="icon-button" aria-label="关闭分组窗口" disabled={saving} onClick={onClose}><X size={18} /></button></header>
      <div className="modal-body">
        {!confirmDelete && <div className="form-grid"><label className="span-2">分组名称<input data-modal-initial-focus required maxLength={160} value={name} disabled={saving} placeholder="例如：火灾检测" onChange={(event) => { setName(event.target.value); setError('') }} /></label></div>}
        {error && <div className="alert error" role="alert">{error}</div>}
        {confirmDelete && group && <div className="delete-confirm" role="alert"><strong>删除分组？</strong><span>组内任务将移到“未分组”，训练记录和文件都会保留。</span><div><button type="button" className="button ghost" disabled={saving} onClick={() => setConfirmDelete(false)}>返回</button><button type="button" className="button danger" disabled={saving} onClick={() => void remove()}><Trash2 size={14} />确认删除分组</button></div></div>}
      </div>
      <footer>{group && !confirmDelete ? <button type="button" className="button danger" disabled={saving} onClick={() => setConfirmDelete(true)}><Trash2 size={14} />删除分组</button> : <span />}<div className="modal-footer-actions"><button type="button" className="button ghost" disabled={saving} onClick={onClose}>{confirmDelete ? '关闭' : '取消'}</button>{!confirmDelete && <button type="submit" className="button primary" disabled={saving || !name.trim()}>{saving && <LoaderCircle size={15} className="spin" />}{saving ? '保存中…' : '保存'}</button>}</div></footer>
    </form>
  </div>
}

interface MoveTaskProps {
  task: TrainingTask
  groups: TaskGroup[]
  onClose: () => void
  onSaved: (task: TrainingTask) => void
}

export function MoveTaskGroupModal({ task, groups, onClose, onSaved }: MoveTaskProps) {
  const [groupId, setGroupId] = useState(task.group_id || '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const dialogRef = useModalFocus<HTMLFormElement>(true, onClose, {
    canClose: !saving,
    initialFocusSelector: 'select',
  })

  async function save(event: React.FormEvent) {
    event.preventDefault()
    if (saving) return
    setSaving(true); setError('')
    try { onSaved(await api.assignTaskGroup(task.id, groupId || null)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : '移动任务失败') }
    finally { setSaving(false) }
  }

  return <div className="modal-backdrop" onMouseDown={() => !saving && onClose()}>
    <form ref={dialogRef} className="modal move-task-modal" role="dialog" aria-modal="true" aria-labelledby="move-task-modal-title" tabIndex={-1} onSubmit={save} onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="eyebrow">MOVE TRAINING TASK</span><h2 id="move-task-modal-title">移动到分组</h2><p>{task.name}</p></div><button type="button" className="icon-button" aria-label="关闭移动任务窗口" disabled={saving} onClick={onClose}><X size={18} /></button></header>
      <div className="modal-body"><div className="form-grid"><label className="span-2">目标分组<select value={groupId} disabled={saving} onChange={(event) => setGroupId(event.target.value)}><option value="">未分组</option>{groups.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label></div>{error && <div className="alert error" role="alert">{error}</div>}</div>
      <footer><span>仅调整展示归属，不会移动训练文件。</span><div className="modal-footer-actions"><button type="button" className="button ghost" disabled={saving} onClick={onClose}>取消</button><button type="submit" className="button primary" disabled={saving || groupId === (task.group_id || '')}><FolderInput size={15} />{saving ? '移动中…' : '移动'}</button></div></footer>
    </form>
  </div>
}
