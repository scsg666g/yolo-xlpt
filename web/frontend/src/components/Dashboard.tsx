import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, ArrowUpRight, CheckCircle2, ChevronDown, Clock3, Copy, Cpu, Folder, FolderInput, FolderPlus, GitCompareArrows, GripVertical, Pencil, Plus, RefreshCw, RotateCcw, Search, Square, Trash2, Zap } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api, metricValue, statusLabel } from '../api'
import { earlyStoppingExplanation, trainingEpochProgress, trainingEpochTarget, trainingStatusLabel } from '../trainingCompletion'
import type { DeletionJob, TaskDeletionPreview, TaskGroup, TrainingTask } from '../types'
import { useModalFocus } from '../useModalFocus'
import { CreateTaskModal } from './CreateTaskModal'
import { RenameTaskModal } from './RenameTaskModal'
import { MoveTaskGroupModal, TaskGroupModal } from './TaskGroupModals'
import { AttemptDeletionHistory } from './AttemptDeletionHistory'

const statusIcon = (status: string) => status === 'completed' ? <CheckCircle2 size={15} /> : status === 'failed' || status === 'interrupted' ? <AlertTriangle size={15} /> : status === 'running' ? <Zap size={15} /> : <Clock3 size={15} />
const activeStatuses = ['queued', 'starting', 'running', 'stopping']
const activeDeletionStatuses = new Set<DeletionJob['status']>(['pending', 'running', 'finalizing'])
const ungroupedKey = '__ungrouped__'

const deletionPhaseLabels: Record<DeletionJob['phase'], string> = {
  stopping_tests: '正在停止模型测试',
  validating: '正在校验清理范围',
  queued: '已进入清理队列',
  waiting_agent: '等待远程 Agent 上线',
  agent_upgrade_required: '远程 Agent 需要升级',
  deleting_files: '正在删除训练文件',
  verifying: '正在验证清理结果',
  center_cleanup: '正在清理平台关联数据',
  completed: '清理完成',
  failed: '清理失败',
}

function deletionPhaseLabel(job: DeletionJob): string {
  if (job.status === 'succeeded') return '清理完成'
  if (job.status === 'failed') return '清理失败'
  if (job.status === 'cancelled') return '清理已取消'
  return deletionPhaseLabels[job.phase] || job.message || '正在处理'
}

function deletionTone(job: DeletionJob): 'active' | 'failed' | 'completed' | 'cancelled' {
  if (job.status === 'succeeded') return 'completed'
  if (job.status === 'failed') return 'failed'
  if (job.status === 'cancelled') return 'cancelled'
  return 'active'
}

function dateTimeLabel(value: string | null): string {
  return value ? new Date(value).toLocaleString('zh-CN') : '—'
}

function deletionExecutorLabel(job: Pick<DeletionJob, 'executor' | 'node_name' | 'node_id'>): string {
  if (job.executor === 'local') return '本机服务器'
  return job.node_name || (job.node_id ? `远程服务器 ${job.node_id.slice(0, 8)}` : '远程服务器')
}

function deletionNodeStatusLabel(status: DeletionJob['node_status']): string | null {
  if (status === 'online') return '在线'
  if (status === 'offline') return '离线'
  if (status === 'pending') return '待连接'
  if (status === 'disabled') return '已停用'
  return null
}

const deletionReceiptLabels: Record<string, string> = {
  version: '回执版本',
  plan_digest: '清理计划摘要',
  removed: '已删除',
  deleted_targets: '已删除目标',
  already_absent: '原本不存在',
  already_missing: '原本不存在',
  already_missing_targets: '原本不存在的目标',
  targets: '清理目标',
  verified_absent: '已验证不存在',
  verified_target_count: '已验证目标数',
  cleaned_test_ids: '已清理的模型测试',
  test_history_removed: '模型测试历史已清理',
  test_storage_cleanup: '模型测试存储',
  center_cleanup: '平台关联数据清理',
  completed_at: '完成时间',
  test_count: '模型测试数',
  run_count: '训练批次数',
  details: '详细信息',
}

function localizeDeletionReceipt(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(localizeDeletionReceipt)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, nested]) => [
      deletionReceiptLabels[key] || key,
      localizeDeletionReceipt(nested),
    ]))
  }
  return value
}

function deletionReceiptValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value === 'string' || typeof value === 'number') return String(value)
  return JSON.stringify(localizeDeletionReceipt(value), null, 2)
}

function taskMetric(task: TrainingTask, keys: string[]): { key: string; value: number } | null {
  for (const key of keys) {
    const raw = task.latest_metrics[key]
    if (typeof raw !== 'number' && typeof raw !== 'string') continue
    if (typeof raw === 'string' && !raw.trim()) continue
    const value = Number(raw)
    if (Number.isFinite(value) && value >= 0 && value <= 1) return { key, value }
  }
  return null
}

function groupMapSummary(tasks: TrainingTask[]) {
  const candidates = tasks.flatMap((task) => {
    const metric = taskMetric(task, ['metrics/mAP50(B)', 'metrics/mAP50(M)'])
    return metric ? [{ task, ...metric }] : []
  })
  const highest = candidates.reduce<(typeof candidates)[number] | null>(
    (current, candidate) => !current || candidate.value > current.value ? candidate : current,
    null,
  )
  // A source includes the node: equal paths on different servers are not
  // evidence of equal datasets. Missing metrics never affect the summary.
  const sources = new Set(candidates.map(({ task, key }) => JSON.stringify([
    task.execution_node_id || 'local', task.data_path, task.task_type, key,
  ])))
  return {
    highest,
    count: candidates.length,
    mixedSources: sources.size > 1 || candidates.some(({ task }) => !task.data_path?.trim()),
  }
}

interface TaskGroupView {
  key: string
  group: TaskGroup | null
  name: string
  allTasks: TrainingTask[]
  tasks: TrainingTask[]
}

interface GroupEditorState {
  group: TaskGroup | null
  taskIds?: string[]
  initialDelete?: boolean
}

export function Dashboard() {
  const [tasks, setTasks] = useState<TrainingTask[]>([])
  const [groups, setGroups] = useState<TaskGroup[]>([])
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
  const [deletePreview, setDeletePreview] = useState<TaskDeletionPreview | null>(null)
  const [deletePreviewLoading, setDeletePreviewLoading] = useState(false)
  const [deletePreviewError, setDeletePreviewError] = useState('')
  const [renameTarget, setRenameTarget] = useState<TrainingTask | null>(null)
  const [groupEditor, setGroupEditor] = useState<GroupEditorState | null>(null)
  const [moveTarget, setMoveTarget] = useState<TrainingTask | null>(null)
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(() => new Set())
  const [deleting, setDeleting] = useState(false)
  const [trashOpen, setTrashOpen] = useState(false)
  const [trashTasks, setTrashTasks] = useState<TrainingTask[]>([])
  const [deletionJobs, setDeletionJobs] = useState<DeletionJob[]>([])
  const [trashLoading, setTrashLoading] = useState(false)
  const [trashError, setTrashError] = useState('')
  const [trashLastUpdated, setTrashLastUpdated] = useState<Date | null>(null)
  const [submittedDeletionJobId, setSubmittedDeletionJobId] = useState<string | null>(null)
  const [operationNotice, setOperationNotice] = useState('')
  const [syncing, setSyncing] = useState(false)
  const [syncNotice, setSyncNotice] = useState('')
  const [taskActionIds, setTaskActionIds] = useState<Set<string>>(() => new Set())
  const [nodeNames, setNodeNames] = useState<Record<string, string>>({})
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const [dragOverId, setDragOverId] = useState<string | null>(null)
  const [ordering, setOrdering] = useState(false)
  const dragMoved = useRef(false)
  const orderRequest = useRef(false)
  const taskRequest = useRef<{ controller: AbortController; timer: number } | null>(null)
  const taskDataVersion = useRef(0)
  const trashRequest = useRef<AbortController | null>(null)
  const trashLoadingRequest = useRef<AbortController | null>(null)
  const deletePreviewRequest = useRef<AbortController | null>(null)
  const deletionJobsRef = useRef<DeletionJob[]>([])
  const trashDataVersion = useRef(0)
  const deleteRequest = useRef(false)
  const syncRequest = useRef(false)
  const taskActionRequests = useRef(new Set<string>())
  const navigate = useNavigate()
  const deleteDialogRef = useModalFocus<HTMLDivElement>(Boolean(deleteTarget), closeDeletion, {
    canClose: !deleting,
    initialFocusSelector: '[data-modal-initial-focus]',
  })
  const trashDialogRef = useModalFocus<HTMLDivElement>(trashOpen && !deleteTarget, () => setTrashOpen(false), {
    canClose: !deleting,
    initialFocusSelector: '[data-modal-initial-focus]',
  })

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
      const [incoming, incomingGroups] = await Promise.all([
        api.tasks(controller.signal),
        api.taskGroups(controller.signal),
      ])
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
        setGroups(incomingGroups)
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

  const filtered = useMemo(() => {
    const groupNames = new Map(groups.map((group) => [group.id, group.name.toLowerCase()]))
    return tasks.filter((task) => {
    const term = search.trim().toLowerCase()
    const matches = !term || task.name.toLowerCase().includes(term) || task.model_path.toLowerCase().includes(term) || Boolean(task.group_id && groupNames.get(task.group_id)?.includes(term))
    const state = filter === 'all' || task.status === filter || (filter === 'active' && activeStatuses.includes(task.status))
    return matches && state
  }) }, [tasks, groups, search, filter])

  const groupedTasks = useMemo(() => {
    const knownIds = new Set(groups.map((group) => group.id))
    const visibleIds = new Set(filtered.map((task) => task.id))
    const views: TaskGroupView[] = groups.map((group) => ({
      key: group.id,
      group,
      name: group.name,
      allTasks: tasks.filter((task) => task.group_id === group.id),
      tasks: tasks.filter((task) => task.group_id === group.id && visibleIds.has(task.id)),
    }))
    const ungrouped = {
      key: ungroupedKey,
      group: null,
      name: '未分组',
      allTasks: tasks.filter((task) => !task.group_id || !knownIds.has(task.group_id)),
      tasks: tasks.filter((task) => (!task.group_id || !knownIds.has(task.group_id)) && visibleIds.has(task.id)),
    }
    if (ungrouped.allTasks.length) views.push(ungrouped)
    if (search.trim() || filter !== 'all') return views.filter((view) => view.tasks.length)
    return views
  }, [filtered, filter, groups, search, tasks])

  const latestDeletionJobs = useMemo(() => {
    const latest = new Map<string, DeletionJob>()
    deletionJobs.forEach((job) => {
      const current = latest.get(job.task_id)
      if (!current || new Date(job.created_at).getTime() > new Date(current.created_at).getTime()) latest.set(job.task_id, job)
    })
    return latest
  }, [deletionJobs])

  const recentDeletionJobs = useMemo(() => {
    const taskIds = new Set(trashTasks.map((task) => task.id))
    return [...latestDeletionJobs.values()]
      .filter((job) => !taskIds.has(job.task_id))
      .sort((left, right) => new Date(right.updated_at).getTime() - new Date(left.updated_at).getTime())
  }, [latestDeletionJobs, trashTasks])

  const refreshTrash = useCallback(async (showLoading = false): Promise<DeletionJob[] | null> => {
    trashRequest.current?.abort()
    const controller = new AbortController()
    const version = trashDataVersion.current
    trashRequest.current = controller
    if (showLoading) {
      trashLoadingRequest.current = controller
      setTrashLoading(true)
    }
    try {
      const [incomingTasks, incomingJobs] = await Promise.all([
        api.deletedTasks(controller.signal),
        api.taskDeletionJobs(controller.signal),
      ])
      if (trashRequest.current !== controller || trashDataVersion.current !== version) return null
      setTrashTasks(incomingTasks)
      setDeletionJobs(incomingJobs)
      deletionJobsRef.current = incomingJobs
      setTrashLastUpdated(new Date())
      setTrashError('')
      return incomingJobs
    } catch (reason) {
      if (trashRequest.current !== controller || controller.signal.aborted) return null
      setTrashError(reason instanceof Error ? reason.message : '回收站加载失败')
      return null
    } finally {
      if (trashRequest.current === controller) {
        trashRequest.current = null
      }
      if (trashLoadingRequest.current === controller) {
        trashLoadingRequest.current = null
        setTrashLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    if (!trashOpen) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let revision = 0
    function schedule(jobs: DeletionJob[]) {
      if (cancelled) return
      const hasActiveJob = jobs.some((job) => activeDeletionStatuses.has(job.status))
      const delay = document.hidden ? 30_000 : hasActiveJob ? 3000 : 15_000
      timer = setTimeout(() => void reload(false), delay)
    }
    async function reload(showLoading: boolean) {
      const requestRevision = ++revision
      const incomingJobs = await refreshTrash(showLoading)
      if (cancelled || requestRevision !== revision) return
      schedule(incomingJobs || deletionJobsRef.current)
    }
    function handleVisibilityChange() {
      if (timer !== undefined) clearTimeout(timer)
      if (document.hidden) schedule(deletionJobsRef.current)
      else void reload(false)
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    void reload(true)
    return () => {
      cancelled = true
      if (timer !== undefined) clearTimeout(timer)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      trashRequest.current?.abort()
      trashRequest.current = null
      trashLoadingRequest.current = null
      setTrashLoading(false)
    }
  }, [trashOpen, refreshTrash])

  const active = tasks.filter((task) => ['queued', 'starting', 'running', 'stopping'].includes(task.status)).length
  const completed = tasks.filter((task) => task.status === 'completed').length
  const attention = tasks.filter((task) => ['failed', 'interrupted'].includes(task.status)).length
  const selectedDeletionJob = deleteTarget ? latestDeletionJobs.get(deleteTarget.id) || null : null

  function toggleGroup(key: string) {
    setExpandedGroups((current) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  function saveGroup(group: TaskGroup, assignedTaskIds: string[] = []) {
    setGroups((current) => {
      const existing = current.some((item) => item.id === group.id)
      return existing ? current.map((item) => item.id === group.id ? group : item) : [group, ...current]
    })
    if (assignedTaskIds.length) {
      const assigned = new Set(assignedTaskIds)
      setTasks((current) => current.map((task) => assigned.has(task.id) ? { ...task, group_id: group.id } : task))
      setOperationNotice(`已创建分组“${group.name}”并移入 ${assignedTaskIds.length} 个任务`)
    }
    setGroupEditor(null)
  }

  function removeGroup(group: TaskGroup) {
    setGroups((current) => current.filter((item) => item.id !== group.id))
    setTasks((current) => current.map((task) => task.group_id === group.id ? { ...task, group_id: null } : task))
    setExpandedGroups((current) => {
      const next = new Set(current)
      next.delete(group.id)
      return next
    })
    setOperationNotice(`分组“${group.name}”已删除，组内任务已移到未分组`)
    setGroupEditor(null)
  }

  function saveTaskGroup(task: TrainingTask) {
    setTasks((current) => current.map((item) => item.id === task.id ? task : item))
    setOperationNotice('任务分组已更新')
    setMoveTarget(null)
  }

  function reorderedTasks(draggedId: string, targetId: string, after = false): TrainingTask[] {
    if (draggedId === targetId) return tasks
    const taskById = new Map(tasks.map((task) => [task.id, task]))
    // A task can disappear while the pointer is held because polling updates the list.
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
    const sourceTask = tasks.find((task) => task.id === draggingId)
    const targetTask = tasks.find((task) => task.id === targetId)
    if (!sourceTask || !targetTask || sourceTask.group_id !== targetTask.group_id) {
      setDraggingId(null); setDragOverId(null)
      setError('请使用文件夹按钮将任务移动到其他分组')
      return
    }
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

  async function taskAction(event: React.MouseEvent, taskId: string, action: () => Promise<unknown>) {
    event.stopPropagation()
    if (taskActionRequests.current.has(taskId)) return
    taskActionRequests.current.add(taskId)
    setTaskActionIds((current) => new Set(current).add(taskId))
    setError('')
    try { await action(); await refresh() } catch (reason) { setError(reason instanceof Error ? reason.message : '操作失败') } finally {
      taskActionRequests.current.delete(taskId)
      setTaskActionIds((current) => {
        const next = new Set(current)
        next.delete(taskId)
        return next
      })
    }
  }

  async function cloneTask(source: TrainingTask) {
    const clone = await api.clone(source.id)
    setTasks((current) => {
      if (current.some((task) => task.id === clone.id)) return current
      const sourceIndex = current.findIndex((task) => task.id === source.id)
      if (sourceIndex < 0) return [...current, clone]
      return [...current.slice(0, sourceIndex), clone, ...current.slice(sourceIndex)]
    })
    return clone
  }

  async function syncHistoricalRuns() {
    if (syncRequest.current) return
    syncRequest.current = true
    setSyncing(true)
    setError('')
    setSyncNotice('')
    try {
      const result = await api.importRuns()
      await refresh()
      setSyncNotice(result.imported > 0 ? `同步完成，新增 ${result.imported} 条历史任务` : '同步完成，没有发现新的历史任务')
    } catch (reason) {
      setError(reason instanceof Error ? `同步历史任务失败：${reason.message}` : '同步历史任务失败')
    } finally {
      syncRequest.current = false
      setSyncing(false)
    }
  }

  function closeDeletion(force = false) {
    if (deleting && !force) return
    deletePreviewRequest.current?.abort()
    deletePreviewRequest.current = null
    setDeleteTarget(null)
    setDeletePreview(null)
    setDeletePreviewLoading(false)
    setDeletePreviewError('')
    setConfirmPermanent(false)
  }

  async function loadDeletionPreview(task: TrainingTask) {
    deletePreviewRequest.current?.abort()
    const controller = new AbortController()
    deletePreviewRequest.current = controller
    setDeletePreview(null)
    setDeletePreviewLoading(true)
    setDeletePreviewError('')
    try {
      const preview = await api.taskDeletionPreview(task.id, controller.signal)
      if (deletePreviewRequest.current !== controller) return
      setDeletePreview(preview)
    } catch (reason) {
      if (deletePreviewRequest.current !== controller || controller.signal.aborted) return
      setDeletePreviewError(reason instanceof Error ? reason.message : '无法读取删除范围')
    } finally {
      if (deletePreviewRequest.current === controller) {
        deletePreviewRequest.current = null
        setDeletePreviewLoading(false)
      }
    }
  }

  function openDeletion(task: TrainingTask, fromTrash = false) {
    if (deleteRequest.current) return
    setDeleteTarget(task)
    setDeleteFromTrash(fromTrash)
    setConfirmPermanent(fromTrash)
    setDeleteError('')
    void loadDeletionPreview(task)
  }

  async function deleteTask(permanent: boolean) {
    if (!deleteTarget || deleteRequest.current) return
    const target = deleteTarget
    if (permanent && (!deletePreview || deletePreview.task_id !== target.id || !deletePreview.can_delete)) {
      setDeleteError(deletePreview?.blocking_reason || '请先等待删除范围校验完成')
      return
    }
    deleteRequest.current = true
    setDeleting(true); setDeleteError(''); setError('')
    let succeeded = false
    try {
      if (permanent) {
        const result = await api.permanentRemove(target.id)
        setSubmittedDeletionJobId(result.operation_id)
        setOperationNotice(result.message || '已提交文件清理，可在回收站查看进度')
      } else {
        await api.remove(target.id)
        setOperationNotice('任务已移入回收站，训练文件已保留')
      }
      // Acknowledgement releases the modal; refresh requests must not hold its lock.
      succeeded = true
      taskDataVersion.current += 1
      trashDataVersion.current += 1
      setTasks((current) => current.filter((task) => task.id !== target.id))
      closeDeletion(true)
      if (permanent) setTrashOpen(true)
    } catch (reason) {
      setDeleteError(reason instanceof Error ? reason.message : '删除失败，请检查任务状态后重试')
    } finally {
      deleteRequest.current = false
      setDeleting(false)
    }
    if (succeeded) {
      void refresh()
      if (trashOpen) void refreshTrash(true)
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

  function renderDeletionReceipt(job: DeletionJob) {
    if (!job.receipt || !Object.keys(job.receipt).length) return null
    return <details className="deletion-receipt">
      <summary>查看清理回执</summary>
      <dl>{Object.entries(job.receipt).map(([key, value]) => <div key={key}>
        <dt>{deletionReceiptLabels[key] || key}</dt>
        <dd><pre>{deletionReceiptValue(value)}</pre></dd>
      </div>)}</dl>
    </details>
  }

  function renderDeletionJobStatus(job: DeletionJob) {
    const nodeStatus = deletionNodeStatusLabel(job.node_status)
    const agentStillRequired = job.executor === 'remote'
      && job.status !== 'succeeded'
      && job.phase !== 'center_cleanup'
      && !job.receipt
    return <>
      <div className="deletion-job-heading">
        <span className={`deletion-status ${deletionTone(job)}`} aria-live="polite">{deletionPhaseLabel(job)}</span>
        <span>{deletionExecutorLabel(job)}</span>
        {agentStillRequired && nodeStatus && <span className={`deletion-node-status ${job.node_status}`}>{nodeStatus}</span>}
      </div>
      {job.message && job.message !== job.error && <p className="deletion-job-message">{job.message}</p>}
      {agentStillRequired && <small>
        Agent {job.agent_version || '未上报'}
        {job.required_agent_version ? ` · 要求 ${job.required_agent_version} 或以上` : ''}
        {job.node_status === 'offline' ? ` · 最后在线 ${dateTimeLabel(job.last_seen_at)}` : ''}
      </small>}
      {job.error && <p className="deletion-job-error"><AlertTriangle size={14} />{job.error}</p>}
    </>
  }

  function renderTrashTask(task: TrainingTask) {
    const job = latestDeletionJobs.get(task.id) || null
    const legacyPending = ['remote_delete_pending', 'local_delete_pending'].includes(task.deleted_reason || '')
    const legacyFailed = ['remote_delete_failed', 'local_delete_failed'].includes(task.deleted_reason || '')
    const jobActive = Boolean(job && activeDeletionStatuses.has(job.status))
    const jobFailed = Boolean(job && ['failed', 'cancelled'].includes(job.status))
    const restoreBlocked = Boolean(job) || legacyPending || legacyFailed
    const canRetry = jobFailed ? Boolean(job?.retryable) : legacyFailed
    const permanentDisabled = deleting || jobActive || (jobFailed && !canRetry) || legacyPending
    const permanentLabel = jobActive || legacyPending ? '清理中' : jobFailed || legacyFailed ? canRetry ? '重试清理' : '无法重试' : '永久删除'
    const location = job
      ? deletionExecutorLabel(job)
      : task.execution_node_id ? nodeNames[task.execution_node_id] || `远程服务器 ${task.execution_node_id.slice(0, 8)}` : '本机服务器'
    const legacyTone = legacyFailed ? 'failed' : legacyPending ? 'active' : 'retained'
    const legacyStatus = legacyFailed ? '清理失败' : legacyPending ? '清理中' : '文件已保留'
    return <article className={`trash-item ${job ? deletionTone(job) : legacyTone}`} key={task.id}>
      <div className="trash-item-copy">
        <div className="trash-item-title"><strong>{task.name}</strong>{!job && <span className={`deletion-status ${legacyTone}`}>{legacyStatus}</span>}</div>
        {job ? renderDeletionJobStatus(job) : <>
          <span>{location} · 原任务状态：{statusLabel(task.status)}</span>
          <p className="deletion-job-message">{legacyPending
            ? '永久清理已提交，该任务不能恢复。'
            : legacyFailed
              ? '部分文件可能已被删除，该任务不能恢复。'
              : '任务可恢复，训练文件和模型测试历史尚未清理。'}</p>
          {legacyPending && <small>已提交文件清理，正在等待服务器处理</small>}
          {legacyFailed && <p className="deletion-job-error"><AlertTriangle size={14} />清理未完成，暂未读取到详细错误，请刷新后重试。</p>}
        </>}
        <small>移入回收站：{dateTimeLabel(task.deleted_at)}</small>
        {job && renderDeletionReceipt(job)}
      </div>
      <div className="trash-item-actions">
        <button type="button" className="button ghost compact" disabled={deleting || restoreBlocked} title={restoreBlocked ? '永久清理已提交，该任务不能恢复' : '恢复任务'} onClick={() => void restoreTask(task)}><RotateCcw size={14} />恢复</button>
        <button type="button" className="button danger compact" disabled={permanentDisabled} title={permanentDisabled ? (jobActive || legacyPending ? '文件清理正在进行' : '当前清理任务不可重试') : permanentLabel} onClick={() => openDeletion(task, true)}><Trash2 size={14} />{permanentLabel}</button>
      </div>
    </article>
  }

  function renderRecentDeletionJob(job: DeletionJob) {
    return <article className={`trash-item deletion-history ${deletionTone(job)} ${job.id === submittedDeletionJobId ? 'is-new' : ''}`} key={job.id}>
      <div className="trash-item-copy">
        <div className="trash-item-title"><strong>{job.task_name}</strong></div>
        {renderDeletionJobStatus(job)}
        <small>更新时间：{dateTimeLabel(job.updated_at)}</small>
        {renderDeletionReceipt(job)}
      </div>
    </article>
  }

  function renderPermanentPreview() {
    if (deletePreviewLoading) return <div className="delete-preview-state" role="status"><RefreshCw size={15} className="spin" />正在校验删除范围…</div>
    if (deletePreviewError) return <div className="alert error delete-preview-error" role="alert"><span>{deletePreviewError}</span><button type="button" className="button ghost compact" disabled={!deleteTarget} onClick={() => deleteTarget && void loadDeletionPreview(deleteTarget)}>重新读取</button></div>
    if (!deletePreview) return null
    const retryWarning = deleteFromTrash && selectedDeletionJob && ['failed', 'cancelled'].includes(selectedDeletionJob.status)
    return <div className="delete-preview">
      <div className="delete-preview-summary">
        <div><span>执行位置</span><strong>{deletePreview.executor === 'local' ? '本机服务器' : deletePreview.node_name || '远程服务器'}</strong></div>
        <div><span>训练批次</span><strong>{deletePreview.attempt_count} 个 Attempt</strong></div>
        <div><span>模型测试</span><strong>{deletePreview.test_count} 条历史</strong></div>
      </div>
      {retryWarning && <div className="delete-partial-warning"><AlertTriangle size={16} /><div><strong>上次清理未完成</strong><span>部分文件可能已被删除，重试会从当前状态继续校验和清理。</span>{selectedDeletionJob.error && <small>{selectedDeletionJob.error}</small>}</div></div>}
      <div className="delete-scope-grid">
        <section><h3>将永久删除</h3><ul>{deletePreview.delete_items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul></section>
        <section><h3>会继续保留</h3><ul>{deletePreview.keep_items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul></section>
      </div>
      {!deletePreview.can_delete && <div className="alert error" role="alert">{deletePreview.blocking_reason || '该任务当前不能永久删除'}</div>}
      <div className="delete-confirm">
        <strong>最后确认</strong>
        <span>训练产物和模型测试历史清理后无法恢复。远程任务会等待 Agent 返回验证结果。</span>
        <div><button type="button" className="button ghost" disabled={deleting} onClick={() => deleteFromTrash ? closeDeletion() : setConfirmPermanent(false)}>返回</button><button type="button" className="button danger" disabled={deleting || !deletePreview.can_delete} onClick={() => void deleteTask(true)}>{deleting ? '提交中…' : retryWarning ? '确认重试清理' : '确认永久删除'}</button></div>
      </div>
    </div>
  }

  function renderTaskCard(task: TrainingTask) {
    const completionNote = earlyStoppingExplanation(task)
    const epochProgress = trainingEpochProgress(task)
    return <article key={task.id} draggable={!ordering} className={`task-card ${draggingId === task.id ? 'is-dragging' : ''} ${dragOverId === task.id ? 'is-drag-over' : ''}`} onClick={() => { if (dragMoved.current) { dragMoved.current = false; return } navigate(`/tasks/${task.id}/canvas`) }} onDragStart={(event) => { setDraggingId(task.id); setDragOverId(null); dragMoved.current = true; event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', task.id) }} onDragOver={(event) => { const source = tasks.find((item) => item.id === draggingId); if (source?.group_id !== task.group_id) return; event.preventDefault(); if (draggingId && draggingId !== task.id) setDragOverId(task.id); event.dataTransfer.dropEffect = 'move' }} onDrop={(event) => { event.preventDefault(); void dropTask(task.id) }} onDragEnd={() => { setDraggingId(null); setDragOverId(null) }}>
      <div className="task-card-top"><span className="task-drag-handle" title="按住拖动调整组内顺序" aria-label="按住拖动调整组内顺序"><GripVertical size={16} /></span><span className={`status ${task.status}`} title={completionNote || undefined}>{statusIcon(task.status)}{trainingStatusLabel(task)}</span></div>
      <div className="task-name-row"><h3 title={task.name}>{task.name}</h3><button type="button" className="icon-button task-rename-button" disabled={taskActionIds.has(task.id)} title="重命名" aria-label={`重命名任务：${task.name}`} onClick={(event) => { event.stopPropagation(); setRenameTarget(task) }}><Pencil size={14} /></button></div><p className="task-path">{task.model_path.split('/').pop()} · {task.task_type}</p>
      <p className="task-path" title={task.execution_node_id || 'local'}>执行服务器：{task.execution_node_id ? nodeNames[task.execution_node_id] || `远程 ${task.execution_node_id.slice(0, 8)}` : '本机服务器'}</p>
      <div className="progress-heading"><span>Epoch {task.completed_epochs} / {trainingEpochTarget(task)}</span><strong title="实际完成轮数占目标轮数的比例">轮数进度 {Math.round(epochProgress)}%</strong></div>
      <div className="progress"><i style={{ width: `${epochProgress}%` }} /></div>
      {completionNote && <p className="training-completion-note">{completionNote}</p>}
      <div className="metric-strip" title={`最新 Epoch ${task.completed_epochs} 指标`}><div><span>mAP50</span><strong>{metricValue(task, ['metrics/mAP50(B)', 'metrics/mAP50(M)'])}</strong></div><div><span>Box loss</span><strong>{metricValue(task, ['train/box_loss'])}</strong></div><div><span>设备</span><strong>{task.device === 'cpu' ? 'CPU' : `GPU ${task.device}`}</strong></div></div>
      {task.error_message && <div className="task-warning"><AlertTriangle size={14} />{task.error_message}</div>}
      <footer><span>{task.source === 'imported' ? '历史导入' : new Date(task.created_at).toLocaleString('zh-CN')}</span><div>
        {task.status === 'draft' && <button type="button" disabled={taskActionIds.has(task.id)} aria-busy={taskActionIds.has(task.id)} title="启动训练" aria-label={`启动训练：${task.name}`} onClick={(e) => taskAction(e, task.id, () => api.start(task.id))}><Zap size={15} /></button>}
        {task.status === 'failed' && <button type="button" disabled={taskActionIds.has(task.id)} aria-busy={taskActionIds.has(task.id)} title="重新训练" aria-label={`重新训练：${task.name}`} onClick={(e) => taskAction(e, task.id, () => api.retry(task.id))}><RotateCcw size={15} /></button>}
        {['stopped', 'interrupted'].includes(task.status) && <button type="button" disabled={taskActionIds.has(task.id)} aria-busy={taskActionIds.has(task.id)} title="继续训练" aria-label={`继续训练：${task.name}`} onClick={(e) => taskAction(e, task.id, () => api.resume(task.id))}><RefreshCw size={15} /></button>}
        {['queued', 'starting', 'running'].includes(task.status) && <button type="button" disabled={taskActionIds.has(task.id)} aria-busy={taskActionIds.has(task.id)} title="停止训练" aria-label={`停止训练：${task.name}`} onClick={(e) => taskAction(e, task.id, () => api.stop(task.id))}><Square size={14} /></button>}
        <button type="button" title="移动到分组" aria-label={`移动任务到分组：${task.name}`} onClick={(event) => { event.stopPropagation(); setMoveTarget(task) }}><FolderInput size={15} /></button>
        <button type="button" disabled={taskActionIds.has(task.id)} aria-busy={taskActionIds.has(task.id)} title="克隆任务" aria-label={`克隆任务：${task.name}`} onClick={(e) => taskAction(e, task.id, () => cloneTask(task))}><Copy size={15} /></button>
        <button type="button" title="加入实验对比" aria-label={`将${task.name}加入实验对比`} onClick={(e) => { e.stopPropagation(); navigate(`/compare?tasks=${task.id}`) }}><GitCompareArrows size={15} /></button>
        {!activeStatuses.includes(task.status) && <button type="button" disabled={taskActionIds.has(task.id)} title="删除任务" aria-label={`删除任务：${task.name}`} onClick={(e) => { e.stopPropagation(); openDeletion(task) }}><Trash2 size={15} /></button>}
        <button type="button" title="打开画布" aria-label={`打开训练画布：${task.name}`} onClick={(e) => { e.stopPropagation(); navigate(`/tasks/${task.id}/canvas`) }}><ArrowUpRight size={15} /></button>
      </div></footer>
    </article>
  }

  return <div className="page dashboard-page">
    <header className="page-header">
      <div><span className="eyebrow">TRAINING OPERATIONS</span><h1>训练任务中心</h1><p>创建、调度并观察每一次模型实验。</p></div>
      <div className="header-actions"><AttemptDeletionHistory /><button type="button" className="button ghost" onClick={openTrash}><Trash2 size={16} />回收站</button><button type="button" className="button ghost" disabled={syncing} onClick={() => void syncHistoricalRuns()}><RefreshCw size={16} className={syncing ? 'spin' : ''} />{syncing ? '同步中…' : '同步历史任务'}</button><button type="button" className="button ghost" onClick={() => setGroupEditor({ group: null })}><FolderPlus size={16} />新建分组</button><button type="button" className="button primary" onClick={() => setCreating(true)}><Plus size={17} />新建训练任务</button></div>
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
      {syncNotice && <div className="alert" role="status">{syncNotice}</div>}
      {operationNotice && <div className="alert" role="status">{operationNotice}</div>}
      {loadError && <div className="alert error" role="alert">{loadError}<button type="button" className="button ghost compact" onClick={() => void refresh()}>重新加载</button></div>}
      {loading ? <div className="empty">正在加载任务…</div> : loadError && !tasks.length ? <div className="empty">暂时无法获取任务列表，请重新加载。</div> : filtered.length === 0 && (search.trim() || filter !== 'all') ? <div className="empty"><Cpu size={28} /><strong>没有匹配的训练任务</strong><span>请调整搜索关键词或状态筛选。</span></div> : tasks.length === 0 && groups.length === 0 ? <div className="empty"><Cpu size={28} /><strong>还没有训练任务</strong><span>可以先建立模型分组，再创建第一次训练。</span></div> :
        <div className="task-group-list">{groupedTasks.map((view) => {
          const isForcedOpen = Boolean(search.trim()) || filter !== 'all'
          const isCollapsed = !expandedGroups.has(view.key) && !isForcedOpen
          const activeCount = view.allTasks.filter((task) => activeStatuses.includes(task.status)).length
          const completedCount = view.allTasks.filter((task) => task.status === 'completed').length
          const attentionCount = view.allTasks.filter((task) => ['failed', 'interrupted'].includes(task.status)).length
          const mapSummary = groupMapSummary(view.allTasks)
          const mapDescription = mapSummary.highest
            ? `取本组 ${mapSummary.count} 个有有效 mAP50 的任务各自最新指标中的最大值，不代表历史最佳。来源：${mapSummary.highest.task.name}；${mapSummary.highest.task.execution_node_id ? nodeNames[mapSummary.highest.task.execution_node_id] || '远程服务器' : '本机服务器'}；${mapSummary.highest.key === 'metrics/mAP50(M)' ? '掩码 mAP50' : '框 mAP50'}。路径相同也不保证数据内容或版本相同。`
            : '组内尚无有效 mAP50；未产生验证指标的任务不参与统计，分类准确率不作为 mAP50。'
          return <section className={`task-group ${isCollapsed ? 'is-collapsed' : ''}`} key={view.key}>
            <header className="task-group-header">
              <div className="task-group-heading">
                <button type="button" className="task-group-toggle" aria-expanded={!isCollapsed} aria-controls={`task-group-${view.key}`} onClick={() => toggleGroup(view.key)}>
                  <ChevronDown size={18} /><span className="task-group-icon"><Folder size={17} /></span><span className="task-group-title"><strong>{view.name}</strong><small>{view.tasks.length === view.allTasks.length ? `${view.allTasks.length} 个任务` : `${view.tasks.length}/${view.allTasks.length} 个任务`}</small></span>
                </button>
                <div className="task-group-actions">
                  <button type="button" className="icon-button rename" title={view.group ? '重命名分组' : '命名未分组任务'} aria-label={view.group ? `重命名分组：${view.name}` : '命名未分组任务'} onClick={() => setGroupEditor(view.group ? { group: view.group } : { group: null, taskIds: view.allTasks.map((task) => task.id) })}><Pencil size={14} /></button>
                  {view.group && <button type="button" className="icon-button delete" title="删除分组" aria-label={`删除分组：${view.name}`} onClick={() => setGroupEditor({ group: view.group, initialDelete: true })}><Trash2 size={14} /></button>}
                </div>
              </div>
              <div className="task-group-summary"><span className={activeCount ? 'active' : ''}>进行中 <strong>{activeCount}</strong></span><span>已完成 <strong>{completedCount}</strong></span>{attentionCount > 0 && <span className="attention">需关注 <strong>{attentionCount}</strong></span>}<span title={mapDescription}>当前最高 mAP50 <strong>{mapSummary.highest ? mapSummary.highest.value.toFixed(4) : '—'}</strong></span>{mapSummary.mixedSources && <span title="有指标的任务来自不同服务器、数据集路径或指标类型，数值汇总仅供参考，不作为模型优劣排名。">多来源，仅供参考</span>}{!mapSummary.highest && <span>暂无有效指标</span>}</div>
            </header>
            {!isCollapsed && <div id={`task-group-${view.key}`} className="task-group-body">{view.tasks.length ? <div className="task-grid">{view.tasks.map(renderTaskCard)}</div> : <div className="task-group-empty">分组中还没有任务，可在新建任务时选择该分组。</div>}</div>}
          </section>
        })}</div>}
    </section>
    {ordering && <div className="order-saving" role="status"><RefreshCw size={13} className="spin" />正在保存任务顺序…</div>}
    {renameTarget && <RenameTaskModal task={renameTarget} onClose={() => setRenameTarget(null)} onSaved={(updated) => { setTasks((current) => current.map((task) => task.id === updated.id ? { ...task, name: updated.name } : task)); setRenameTarget(null) }} />}
    {deleteTarget && <div className="modal-backdrop delete-choice-backdrop" onMouseDown={() => closeDeletion()}>
      <div ref={deleteDialogRef} className="modal delete-choice-modal" role="dialog" aria-modal="true" aria-labelledby="delete-task-modal-title" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">DELETE TRAINING TASK</span><h2 id="delete-task-modal-title">{deleteFromTrash ? '确认永久删除' : '删除训练任务'}</h2><p>{deleteTarget.name}</p></div><button type="button" className="icon-button" data-modal-initial-focus title="关闭" aria-label="关闭删除任务弹窗" disabled={deleting} onClick={() => closeDeletion()}>×</button></header>
        <div className="delete-choice-body">
          {confirmPermanent ? <><p>请核对执行位置和清理范围。提交后任务不能恢复，可在回收站持续查看进度。</p>{renderPermanentPreview()}</> : <>
            <p>移入回收站仅从任务中心隐藏，文件会保留。永久删除会先校验范围，再由本机清理服务或远程 Agent 执行。</p>
            <div className="delete-choice-grid">
              <button type="button" className="delete-choice soft" disabled={deleting} onClick={() => void deleteTask(false)}><strong>移入回收站</strong><span>保留训练记录、权重、日志和模型测试历史，以后可恢复。</span></button>
              <button type="button" className="delete-choice permanent" disabled={deleting || deletePreviewLoading || Boolean(deletePreviewError) || !deletePreview?.can_delete} onClick={() => setConfirmPermanent(true)}><strong>永久删除</strong><span>{deletePreview ? `清理 ${deletePreview.attempt_count} 个 Attempt 和 ${deletePreview.test_count} 条模型测试历史，无法撤销。` : '正在校验将要清理的训练产物和历史记录。'}</span></button>
            </div>
            <div className="delete-preview-readiness">{deletePreviewLoading && <span><RefreshCw size={13} className="spin" />正在读取删除范围…</span>}{deletePreviewError && <span className="error"><AlertTriangle size={13} />{deletePreviewError}<button type="button" className="button ghost compact" onClick={() => void loadDeletionPreview(deleteTarget)}>重试</button></span>}{deletePreview && !deletePreview.can_delete && <span className="error"><AlertTriangle size={13} />{deletePreview.blocking_reason || '该任务当前不能永久删除'}</span>}{deletePreview?.can_delete && <span>删除范围已校验，点击“永久删除”查看明细。</span>}</div>
          </>}
          {deleteError && <div className="alert error" role="alert">{deleteError}</div>}
        </div>
        <footer><span>原任务状态：{statusLabel(deleteTarget.status)}</span><button type="button" className="button ghost" disabled={deleting} onClick={() => closeDeletion()}>取消</button></footer>
      </div>
    </div>}
    {trashOpen && !deleteTarget && <div className="modal-backdrop" onMouseDown={() => !deleting && setTrashOpen(false)}><div ref={trashDialogRef} className="modal trash-modal" role="dialog" aria-modal="true" aria-labelledby="trash-modal-title" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="eyebrow">RECYCLE BIN</span><h2 id="trash-modal-title">训练任务回收站</h2><p>{trashTasks.length} 个回收站任务 · {recentDeletionJobs.length} 条近期清理记录</p></div><div className="trash-header-actions"><button type="button" className="icon-button" title="刷新清理状态" aria-label="刷新回收站和清理状态" disabled={trashLoading} onClick={() => void refreshTrash(true)}><RefreshCw size={16} className={trashLoading ? 'spin' : ''} /></button><button type="button" className="icon-button" data-modal-initial-focus title="关闭" aria-label="关闭回收站" disabled={deleting} onClick={() => setTrashOpen(false)}>×</button></div></header>
      <div className="trash-list">{operationNotice && <p className="trash-operation-notice" role="status">{operationNotice}</p>}{trashError && <p className="alert error" role="alert">{trashError}</p>}{trashLoading && !trashTasks.length && !recentDeletionJobs.length && <div className="empty">正在加载回收站…</div>}{trashTasks.map(renderTrashTask)}{recentDeletionJobs.length > 0 && <section className="trash-history" aria-labelledby="trash-history-title"><div className="trash-history-heading"><strong id="trash-history-title">近期清理记录</strong><span>不再出现在回收站任务列表的清理状态与回执。</span></div>{recentDeletionJobs.map(renderRecentDeletionJob)}</section>}{!trashLoading && !trashTasks.length && !recentDeletionJobs.length && <div className="empty">回收站为空</div>}</div>
      <footer><span>上次检查：{trashLastUpdated ? trashLastUpdated.toLocaleTimeString('zh-CN') : '尚未完成'} · 移入回收站的任务不会被历史同步重新导入</span><button type="button" className="button ghost" disabled={deleting} onClick={() => setTrashOpen(false)}>关闭</button></footer>
    </div></div>}
    {groupEditor && <TaskGroupModal group={groupEditor.group} taskIds={groupEditor.taskIds} initialDelete={groupEditor.initialDelete} onClose={() => setGroupEditor(null)} onSaved={saveGroup} onDeleted={removeGroup} />}
    {moveTarget && <MoveTaskGroupModal task={moveTarget} groups={groups} onClose={() => setMoveTarget(null)} onSaved={saveTaskGroup} />}
    <CreateTaskModal open={creating} groups={groups} onClose={() => setCreating(false)} onCreated={(task) => { setCreating(false); navigate(`/tasks/${task.id}/canvas`) }} />
  </div>
}
