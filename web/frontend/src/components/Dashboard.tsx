import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, ArrowUpRight, CheckCircle2, Clock3, Copy, Cpu, GitCompareArrows, GripVertical, MoreHorizontal, Pencil, Plus, RefreshCw, RotateCcw, Search, Square, Trash2, Zap } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api, metricValue, statusLabel } from '../api'
import type { TrainingTask } from '../types'
import { CreateTaskModal } from './CreateTaskModal'
import { RenameTaskModal } from './RenameTaskModal'

const statusIcon = (status: string) => status === 'completed' ? <CheckCircle2 size={15} /> : status === 'failed' || status === 'interrupted' ? <AlertTriangle size={15} /> : status === 'running' ? <Zap size={15} /> : <Clock3 size={15} />

export function Dashboard() {
  const [tasks, setTasks] = useState<TrainingTask[]>([])
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [creating, setCreating] = useState(false)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')
  const [error, setError] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<TrainingTask | null>(null)
  const [deleteFromTrash, setDeleteFromTrash] = useState(false)
  const [confirmPermanent, setConfirmPermanent] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const [renameTarget, setRenameTarget] = useState<TrainingTask | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [trashOpen, setTrashOpen] = useState(false)
  const [trashTasks, setTrashTasks] = useState<TrainingTask[]>([])
  const [trashLoading, setTrashLoading] = useState(false)
  const [trashError, setTrashError] = useState('')
  const [operationNotice, setOperationNotice] = useState('')
  const [nodeNames, setNodeNames] = useState<Record<string, string>>({})
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const [dragOverId, setDragOverId] = useState<string | null>(null)
  const [ordering, setOrdering] = useState(false)
  const dragMoved = useRef(false)
  const orderRequest = useRef(false)
  const taskRequest = useRef<{ controller: AbortController; timer: number } | null>(null)
  const taskDataVersion = useRef(0)
  const trashRequest = useRef<AbortController | null>(null)
  const trashDataVersion = useRef(0)
  const deleteRequest = useRef(false)
  const navigate = useNavigate()

  useEffect(() => {
    const controller = new AbortController()
    void api.computeNodes(controller.signal).then((nodes) => {
      if (!controller.signal.aborted) setNodeNames(Object.fromEntries(nodes.map((node) => [node.id, node.name])))
    }).catch(() => undefined)
    return () => controller.abort()
  }, [])

  const refresh = useCallback(async () => {
    if (taskRequest.current) {
      window.clearTimeout(taskRequest.current.timer)
      taskRequest.current.controller.abort()
    }
    const controller = new AbortController()
    const dataVersion = taskDataVersion.current
    let timedOut = false
    const request = {
      controller,
      timer: window.setTimeout(() => { timedOut = true; controller.abort() }, 15_000),
    }
    taskRequest.current = request
    setLoadError('')
    if (dataVersion === 0) setLoading(true)
    try {
      const incoming = await api.tasks(controller.signal)
      if (taskRequest.current !== request) return
      // A newer poll response must not be replaced by an older HTTP response.
      if (taskDataVersion.current === dataVersion) {
        taskDataVersion.current += 1
        setTasks((current) => {
          if (orderRequest.current) return current
          // Keep the user's current order stable while still incorporating new,
          // removed, or updated tasks from the polling response.
          if (!current.length) return incoming
          const incomingById = new Map(incoming.map((task) => [task.id, task]))
          const currentIds = current.map((task) => task.id)
          const orderedIds = [
            ...currentIds.filter((id) => incomingById.has(id)),
            ...incoming.map((task) => task.id).filter((id) => !currentIds.includes(id)),
          ]
          return orderedIds.map((id) => incomingById.get(id)!).filter(Boolean)
        })
      }
      setLoadError('')
    } catch (reason) {
      if (taskRequest.current !== request || taskDataVersion.current !== dataVersion) return
      setLoadError(timedOut ? '任务加载超时，请检查网络后重试' : reason instanceof Error ? reason.message : '任务加载失败，请重试')
    } finally {
      window.clearTimeout(request.timer)
      if (taskRequest.current === request) {
        taskRequest.current = null
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof window.setTimeout> | undefined
    async function poll() {
      await refresh()
      if (!cancelled) timer = window.setTimeout(poll, 3000)
    }
    void poll()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
      if (taskRequest.current) {
        window.clearTimeout(taskRequest.current.timer)
        taskRequest.current.controller.abort()
        taskRequest.current = null
      }
    }
  }, [refresh])

  const filtered = useMemo(() => tasks.filter((task) => {
    const matches = task.name.toLowerCase().includes(search.toLowerCase()) || task.model_path.toLowerCase().includes(search.toLowerCase())
    const state = filter === 'all' || task.status === filter || (filter === 'active' && ['queued', 'starting', 'running', 'stopping'].includes(task.status))
    return matches && state
  }), [tasks, search, filter])

  const refreshTrash = useCallback(async () => {
    trashRequest.current?.abort()
    const controller = new AbortController()
    const version = trashDataVersion.current
    trashRequest.current = controller
    setTrashLoading(true)
    try {
      const incoming = await api.deletedTasks(controller.signal)
      if (trashRequest.current !== controller || trashDataVersion.current !== version) return
      setTrashTasks(incoming)
      setTrashError('')
    } catch (reason) {
      if (trashRequest.current !== controller || controller.signal.aborted) return
      setTrashError(reason instanceof Error ? reason.message : '回收站加载失败')
    } finally {
      if (trashRequest.current === controller) {
        trashRequest.current = null
        setTrashLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    if (!trashOpen) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    async function reload() {
      await refreshTrash()
      if (!cancelled) timer = setTimeout(reload, 3000)
    }
    void reload()
    return () => {
      cancelled = true
      clearTimeout(timer)
      trashRequest.current?.abort()
      trashRequest.current = null
    }
  }, [trashOpen, refreshTrash])

  const active = tasks.filter((task) => ['queued', 'starting', 'running', 'stopping'].includes(task.status)).length
  const completed = tasks.filter((task) => task.status === 'completed').length
  const attention = tasks.filter((task) => ['failed', 'interrupted'].includes(task.status)).length

  function reorderedTasks(draggedId: string, targetId: string, after = false): TrainingTask[] {
    if (draggedId === targetId) return tasks
    const taskById = new Map(tasks.map((task) => [task.id, task]))
    // A task can disappear while the pointer is held because SSE updates the list.
    // Keep the current order intact instead of creating an undefined card.
    if (!taskById.has(draggedId) || !taskById.has(targetId)) return tasks
    const visibleIds = new Set(filtered.map((task) => task.id))
    const visibleOrder = filtered.map((task) => task.id).filter((id) => id !== draggedId)
    const targetIndex = visibleOrder.indexOf(targetId)
    if (targetIndex < 0) return tasks
    visibleOrder.splice(targetIndex + (after ? 1 : 0), 0, draggedId)
    const replacement = visibleOrder
      .map((id) => taskById.get(id))
      .filter((task): task is TrainingTask => Boolean(task))
    if (replacement.length !== visibleIds.size) return tasks
    let cursor = 0
    return tasks.map((task) => visibleIds.has(task.id) ? replacement[cursor++] : task)
  }

  async function dropTask(targetId: string) {
    if (!draggingId || draggingId === targetId || ordering) return
    const previous = tasks
    const sourceIndex = filtered.findIndex((task) => task.id === draggingId)
    const targetIndex = filtered.findIndex((task) => task.id === targetId)
    if (sourceIndex < 0 || targetIndex < 0) {
      setDraggingId(null); setDragOverId(null)
      setError('任务列表刚刚发生变化，请重新拖动')
      return
    }
    // A card dragged toward a later position is placed after the target; a card
    // dragged toward an earlier position is placed before it.
    const next = reorderedTasks(draggingId, targetId, sourceIndex < targetIndex)
    if (next === previous) {
      setDraggingId(null); setDragOverId(null)
      setError('任务列表刚刚发生变化，请重新拖动')
      return
    }
    setTasks(next)
    setDraggingId(null); setDragOverId(null); dragMoved.current = true
    setOrdering(true); orderRequest.current = true; setError('')
    try {
      await api.reorderTasks(next.map((task) => task.id))
    } catch (reason) {
      setTasks(previous)
      setError(reason instanceof Error ? reason.message : '排序保存失败，请重试')
    } finally {
      orderRequest.current = false
      setOrdering(false)
    }
  }

  async function taskAction(event: React.MouseEvent, action: () => Promise<unknown>) {
    event.stopPropagation()
    try { await action(); await refresh() } catch (reason) { setError(reason instanceof Error ? reason.message : '操作失败') }
  }

  function openDeletion(task: TrainingTask, fromTrash = false) {
    if (deleteRequest.current) return
    setDeleteTarget(task)
    setDeleteFromTrash(fromTrash)
    setConfirmPermanent(fromTrash)
    setDeleteError('')
  }

  async function deleteTask(permanent: boolean) {
    if (!deleteTarget || deleteRequest.current) return
    const target = deleteTarget
    deleteRequest.current = true
    setDeleting(true); setDeleteError(''); setError('')
    let succeeded = false
    let pending = false
    try {
      if (permanent) {
        const result = await api.permanentRemove(target.id)
        pending = result?.status === 'pending'
        setOperationNotice(result?.message || '任务已永久删除')
      } else {
        await api.remove(target.id)
        setOperationNotice('任务已移入回收站，训练文件已保留')
      }
      // Acknowledgement releases the modal; refresh requests must not hold its lock.
      succeeded = true
      taskDataVersion.current += 1
      trashDataVersion.current += 1
      setTasks((current) => current.filter((task) => task.id !== target.id))
      setTrashTasks((current) => {
        const remaining = current.filter((task) => task.id !== target.id)
        return pending || !permanent ? [...remaining, {
          ...target, deleted_at: target.deleted_at || new Date().toISOString(),
          deleted_reason: pending ? (target.execution_node_id ? 'remote_delete_pending' : 'local_delete_pending') : 'soft_delete',
        }] : remaining
      })
      setDeleteTarget(null)
      if (pending) setTrashOpen(true)
    } catch (reason) {
      setDeleteError(reason instanceof Error ? reason.message : '删除失败，请检查任务状态后重试')
    } finally {
      deleteRequest.current = false
      setDeleting(false)
    }
    if (succeeded) {
      void refresh()
      if (trashOpen) void refreshTrash()
    }
  }

  function openTrash() {
    setTrashError('')
    setTrashLoading(true)
    setTrashOpen(true)
  }

  async function restoreTask(task: TrainingTask) {
    if (deleteRequest.current) return
    deleteRequest.current = true
    setDeleting(true); setTrashError('')
    let succeeded = false
    try {
      await api.restore(task.id)
      succeeded = true
      taskDataVersion.current += 1
      trashDataVersion.current += 1
      setTrashTasks((current) => current.filter((item) => item.id !== task.id))
      setOperationNotice('任务已恢复')
    } catch (reason) {
      setTrashError(reason instanceof Error ? reason.message : '恢复失败')
    } finally {
      deleteRequest.current = false
      setDeleting(false)
    }
    if (succeeded) { void refresh(); void refreshTrash() }
  }

  return <div className="page dashboard-page">
    <header className="page-header">
      <div><span className="eyebrow">TRAINING OPERATIONS</span><h1>训练任务中心</h1><p>创建、调度并观察每一次模型实验。</p></div>
      <div className="header-actions"><button className="button ghost" onClick={openTrash}><Trash2 size={16} />回收站</button><button className="button ghost" onClick={() => api.importRuns().then(refresh)}><RefreshCw size={16} />同步历史任务</button><button className="button primary" onClick={() => setCreating(true)}><Plus size={17} />新建训练任务</button></div>
    </header>

    <section className="stat-row">
      <div className="stat-card"><span>全部任务</span><strong>{tasks.length}</strong><small>训练记录总数</small></div>
      <div className="stat-card accent"><span>正在进行</span><strong>{active}</strong><small><i />实时运行与排队</small></div>
      <div className="stat-card"><span>已完成</span><strong>{completed}</strong><small>成功产出模型</small></div>
      <div className="stat-card warning"><span>需要关注</span><strong>{attention}</strong><small>失败或意外中断</small></div>
    </section>

    <section className="task-section">
      <div className="section-toolbar">
        <div><h2>所有训练</h2><span>{filtered.length} 条记录</span></div>
        <div className="toolbar-actions"><div className="search"><Search size={16} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="搜索任务或模型" /></div><select value={filter} onChange={(e) => setFilter(e.target.value)}><option value="all">全部状态</option><option value="active">进行中</option><option value="completed">已完成</option><option value="interrupted">已中断</option><option value="failed">失败</option><option value="draft">草稿</option></select></div>
      </div>
      {error && <div className="alert error">{error}</div>}
      {loadError && <div className="alert error" role="alert">{loadError}<button type="button" className="button ghost compact" onClick={() => void refresh()}>重新加载</button></div>}
      {loading ? <div className="empty">正在加载任务…</div> : loadError && !tasks.length ? <div className="empty">暂时无法获取任务列表，请重新加载。</div> : filtered.length === 0 ? <div className="empty"><Cpu size={28} /><strong>没有匹配的训练任务</strong><span>创建任务后，每一次训练都会拥有独立画布。</span></div> :
        <div className="task-grid">{filtered.map((task) => <article key={task.id} draggable={!ordering} className={`task-card ${draggingId === task.id ? 'is-dragging' : ''} ${dragOverId === task.id ? 'is-drag-over' : ''}`} onClick={() => { if (dragMoved.current) { dragMoved.current = false; return } navigate(`/tasks/${task.id}/canvas`) }} onDragStart={(event) => { setDraggingId(task.id); setDragOverId(null); dragMoved.current = true; event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', task.id) }} onDragOver={(event) => { event.preventDefault(); if (draggingId && draggingId !== task.id) setDragOverId(task.id); event.dataTransfer.dropEffect = 'move' }} onDrop={(event) => { event.preventDefault(); void dropTask(task.id) }} onDragEnd={() => { setDraggingId(null); setDragOverId(null) }}>
          <div className="task-card-top"><span className="task-drag-handle" title="按住拖动调整顺序" aria-label="按住拖动调整顺序"><GripVertical size={16} /></span><span className={`status ${task.status}`}>{statusIcon(task.status)}{statusLabel(task.status)}</span><button className="icon-button" onClick={(event) => event.stopPropagation()}><MoreHorizontal size={18} /></button></div>
          <div className="task-name-row"><h3 title={task.name}>{task.name}</h3><button type="button" className="icon-button task-rename-button" title="重命名" aria-label={`重命名任务：${task.name}`} onClick={(event) => { event.stopPropagation(); setRenameTarget(task) }}><Pencil size={14} /></button></div><p className="task-path">{task.model_path.split('/').pop()} · {task.task_type}</p>
          <p className="task-path" title={task.execution_node_id || 'local'}>执行服务器：{task.execution_node_id ? nodeNames[task.execution_node_id] || `远程 ${task.execution_node_id.slice(0, 8)}` : '本机服务器'}</p>
          <div className="progress-heading"><span>Epoch {task.completed_epochs} / {task.epochs}</span><strong>{Math.round(task.progress)}%</strong></div>
          <div className="progress"><i style={{ width: `${task.progress}%` }} /></div>
          <div className="metric-strip"><div><span>mAP50</span><strong>{metricValue(task, ['metrics/mAP50(B)', 'metrics/mAP50(M)'])}</strong></div><div><span>Box loss</span><strong>{metricValue(task, ['train/box_loss'])}</strong></div><div><span>设备</span><strong>GPU {task.device}</strong></div></div>
          {task.error_message && <div className="task-warning"><AlertTriangle size={14} />{task.error_message}</div>}
          <footer><span>{task.source === 'imported' ? '历史导入' : new Date(task.created_at).toLocaleString('zh-CN')}</span><div>
            {['draft', 'failed'].includes(task.status) && <button title="启动" onClick={(e) => taskAction(e, () => api.start(task.id))}><Zap size={15} /></button>}
            {['stopped', 'interrupted'].includes(task.status) && <button title="继续训练" onClick={(e) => taskAction(e, () => api.resume(task.id))}><RefreshCw size={15} /></button>}
            {['queued', 'starting', 'running'].includes(task.status) && <button title="停止" onClick={(e) => taskAction(e, () => api.stop(task.id))}><Square size={14} /></button>}
            <button title="克隆" onClick={(e) => taskAction(e, () => api.clone(task.id))}><Copy size={15} /></button>
            <button title="加入实验对比" onClick={(e) => { e.stopPropagation(); navigate(`/compare?tasks=${task.id}`) }}><GitCompareArrows size={15} /></button>
            {!['queued', 'starting', 'running', 'stopping'].includes(task.status) && <button title="删除" onClick={(e) => { e.stopPropagation(); openDeletion(task) }}><Trash2 size={15} /></button>}
            <button title="打开画布"><ArrowUpRight size={15} /></button>
          </div></footer>
        </article>)}</div>}
    </section>
    {ordering && <div className="order-saving" role="status"><RefreshCw size={13} className="spin" />正在保存任务顺序…</div>}
    {renameTarget && <RenameTaskModal task={renameTarget} onClose={() => setRenameTarget(null)} onSaved={(updated) => { setTasks((current) => current.map((task) => task.id === updated.id ? { ...task, name: updated.name } : task)); setRenameTarget(null) }} />}
    {deleteTarget && <div className="modal-backdrop delete-choice-backdrop" onMouseDown={() => !deleting && setDeleteTarget(null)}>
      <div className="modal delete-choice-modal" onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">DELETE TRAINING TASK</span><h2>{deleteFromTrash ? '确认永久删除' : '删除训练任务'}</h2><p>{deleteTarget.name}</p></div><button type="button" className="icon-button" disabled={deleting} onClick={() => setDeleteTarget(null)}>×</button></header>
        <div className="delete-choice-body">{deleteFromTrash ? <p>此操作将清理全部 Attempt、日志、权重和图表，且无法恢复。请确认继续。</p> : <><p>请选择删除方式。移入回收站保留文件，可恢复。永久删除将清理该任务的训练文件；远程任务等待 Agent 确认成功后才删除记录，提交后不能恢复。</p><div className="delete-choice-grid"><button type="button" className="delete-choice soft" disabled={deleting} onClick={() => void deleteTask(false)}><strong>移入回收站</strong><span>从平台隐藏，保留训练记录和文件，可恢复。</span></button><button type="button" className="delete-choice permanent" disabled={deleting} onClick={() => setConfirmPermanent(true)}><strong>永久删除</strong><span>清理全部 Attempt、日志、权重和图表，无法撤销。</span></button></div><small>基础模型和数据集文件不会被删除。</small></>}{deleteError && <div className="alert error" role="alert">{deleteError}</div>}{confirmPermanent && <div className="delete-confirm" role="alertdialog" aria-label="确认永久删除"><strong>最后确认</strong><span>永久删除后不能恢复，是否继续？</span><div><button type="button" className="button ghost" disabled={deleting} onClick={() => setConfirmPermanent(false)}>返回</button><button type="button" className="button danger" disabled={deleting} onClick={() => { setConfirmPermanent(false); void deleteTask(true) }}>{deleting ? '提交中…' : '确认永久删除'}</button></div></div>}</div>
        <footer><span>任务当前状态：{statusLabel(deleteTarget.status)}</span><button type="button" className="button ghost" disabled={deleting} onClick={() => setDeleteTarget(null)}>取消</button></footer>
      </div>
    </div>}
    {trashOpen && <div className="modal-backdrop" onMouseDown={() => !deleting && setTrashOpen(false)}><div className="modal trash-modal" onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="eyebrow">RECYCLE BIN</span><h2>训练任务回收站</h2><p>共 {trashTasks.length} 条已删除任务</p></div><button type="button" className="icon-button" disabled={deleting} onClick={() => setTrashOpen(false)}>×</button></header>
      <div className="trash-list">{operationNotice && <p role="status">{operationNotice}</p>}{trashError && <p className="alert error" role="alert">{trashError}</p>}{trashLoading && !trashTasks.length && <div className="empty">正在加载回收站…</div>}{trashTasks.map((task) => <article key={task.id}><div><strong>{task.name}</strong><span>{task.execution_node_id ? '远程服务器' : '本机服务器'} · {statusLabel(task.status)}</span><small>删除时间：{task.deleted_at ? new Date(task.deleted_at).toLocaleString('zh-CN') : '—'}</small>{task.deleted_reason === 'remote_delete_pending' && <small role="status">等待远程 Agent 清理；离线时保留记录，联网后执行</small>}{task.deleted_reason === 'local_delete_pending' && <small role="status">正在后台清理本机训练文件</small>}{['remote_delete_failed', 'local_delete_failed'].includes(task.deleted_reason || '') && <small role="alert">{task.error_message}</small>}</div><button type="button" className="button ghost compact" disabled={deleting || ['remote_delete_pending', 'remote_delete_failed', 'local_delete_pending', 'local_delete_failed'].includes(task.deleted_reason || '')} onClick={() => void restoreTask(task)}><RotateCcw size={14} />恢复</button><button type="button" className="button danger compact" disabled={deleting || ['remote_delete_pending', 'local_delete_pending'].includes(task.deleted_reason || '')} onClick={() => openDeletion(task, true)}><Trash2 size={14} />{['remote_delete_pending', 'local_delete_pending'].includes(task.deleted_reason || '') ? '清理中' : ['remote_delete_failed', 'local_delete_failed'].includes(task.deleted_reason || '') ? '重试清理' : '永久删除'}</button></article>)}{!trashLoading && !trashTasks.length && <div className="empty">回收站为空</div>}</div>
      <footer><span>移入回收站的任务不会被历史同步重新导入</span><button type="button" className="button ghost" disabled={deleting} onClick={() => setTrashOpen(false)}>关闭</button></footer>
    </div></div>}
    <CreateTaskModal open={creating} onClose={() => setCreating(false)} onCreated={(task) => { setCreating(false); navigate(`/tasks/${task.id}/canvas`) }} />
  </div>
}
