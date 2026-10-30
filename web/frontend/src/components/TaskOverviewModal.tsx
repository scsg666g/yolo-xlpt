import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  AlertTriangle, ArrowUpRight, Check, CheckCircle2, Clock3, Grid2X2, GripHorizontal,
  Link2, LoaderCircle, LocateFixed, Maximize2, Monitor, Save, Search, Server, Trash2, X, Zap,
} from 'lucide-react'
import {
  addEdge, Background, BackgroundVariant, Controls, Handle, MarkerType, MiniMap,
  Position, ReactFlow, ReactFlowProvider, useEdgesState, useNodesInitialized,
  useNodesState, useReactFlow, type Connection, type Edge, type Node, type NodeProps,
  type Viewport,
} from '@xyflow/react'
import { api, statusLabel } from '../api'
import { taskMetric } from '../groupMetrics'
import { primaryMetric } from '../metricProfiles'
import { useTaskCenterPreferences } from '../preferences'
import { earlyStoppingExplanation, trainingEpochProgress, trainingEpochTarget, trainingStatusLabel } from '../trainingCompletion'
import type { TaskWorkspace, TaskWorkspaceModeLayout, TrainingTask } from '../types'
import { useModalFocus } from '../useModalFocus'
import '../taskOverview.css'

export type TaskOverviewMode = 'all' | 'active' | 'completed' | 'attention'

type Props = {
  mode: TaskOverviewMode
  tasks: TrainingTask[]
  nodeNames: Record<string, string>
  onClose: () => void
  onNavigate: () => void
}

type OverviewData = { task: TrainingTask; serverName: string; onNavigate: () => boolean }
type OverviewNode = Node<OverviewData, 'taskOverview'>
type OverviewEdge = Edge
type LayoutFeedback = { kind: 'loading' | 'saving' | 'success' | 'error'; message: string }

const CARD_WIDTH = 306
const CARD_HEIGHT = 204
const GRID_GAP_X = 36
const GRID_GAP_Y = 38
const activeStatuses = new Set<TrainingTask['status']>(['queued', 'starting', 'running', 'stopping'])
const attentionStatuses = new Set<TrainingTask['status']>(['failed', 'interrupted'])
const modeTitles: Record<TaskOverviewMode, string> = { all: '全部任务', active: '正在进行', completed: '已完成', attention: '需要关注' }
const modeDescriptions: Record<TaskOverviewMode, string> = {
  all: '自由整理全部任务，关联线仅用于表达关系，不影响训练调度。',
  active: '集中查看正在运行、排队或停止中的任务及其实时进度。',
  completed: '整理已完成任务，并保留轮数、时间和最新验证指标。',
  attention: '集中排查失败与意外中断任务，可用关联线标记排查关系。',
}

function matchesMode(task: TrainingTask, mode: TaskOverviewMode): boolean {
  if (mode === 'active') return activeStatuses.has(task.status)
  if (mode === 'completed') return task.status === 'completed'
  if (mode === 'attention') return attentionStatuses.has(task.status)
  return true
}

function formatTime(value: string | null | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false })
}

function metricSummary(task: TrainingTask): string | null {
  const metric = primaryMetric(task.task_type)
  if (!metric) return null
  const value = taskMetric(task, metric.key)
  return value === null ? null : metric.name + ' ' + value.toFixed(4)
}

function taskServerName(task: TrainingTask, nodeNames: Record<string, string>): string {
  if (!task.execution_node_id) return '本机服务器'
  return nodeNames[task.execution_node_id] || '远程服务器 ' + task.execution_node_id.slice(0, 8)
}

function cardTone(status: TrainingTask['status']): string {
  if (status === 'failed' || status === 'interrupted') return 'attention'
  if (status === 'completed') return 'completed'
  if (activeStatuses.has(status)) return 'active'
  return 'neutral'
}

function cardSignature(task: TrainingTask): string {
  return JSON.stringify([
    task.name, task.status, task.epochs, task.completed_epochs, task.task_type,
    task.queue_reason, task.error_message, task.finished_at,
    task.status === 'draft' || task.status === 'stopped' || attentionStatuses.has(task.status) ? task.updated_at : null,
    task.completion_info?.reason, task.completion_info?.expected_epochs,
    task.completion_info?.best_epoch, task.completion_info?.patience, task.latest_metrics,
  ])
}

function cardAccessibilityLabel(task: TrainingTask, current: number, target: number, percent: number | null): string {
  const label = '打开训练画布：' + task.name + '，' + trainingStatusLabel(task)
  if (task.status === 'running') return label + '，已完成 ' + current + ' / ' + (target > 0 ? target : '目标未知') + ' Epoch' + (percent === null ? '' : '，' + percent + '%')
  if (task.status === 'queued') return label + '，' + (task.queue_reason || '等待调度')
  if (task.status === 'starting') return label + '，准备训练'
  if (task.status === 'stopping') return label + '，正在结束'
  if (attentionStatuses.has(task.status)) return label + '，' + (task.error_message || '请查看任务日志').slice(0, 120)
  return label
}

function defaultPosition(index: number, columns: number) {
  return { x: index % columns * (CARD_WIDTH + GRID_GAP_X), y: Math.floor(index / columns) * (CARD_HEIGHT + GRID_GAP_Y) }
}

function overviewColumns(width: number): number {
  return Math.max(1, Math.min(5, Math.floor((Math.max(360, width) - 80) / (CARD_WIDTH + GRID_GAP_X))))
}

function edgeId(): string {
  return 'overview-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10)
}

const TaskOverviewNode = memo(function TaskOverviewNode({ data, dragging }: NodeProps<OverviewNode>) {
  const navigate = useNavigate()
  const suppressClick = useRef(false)
  const { task, serverName, onNavigate } = data
  const tone = cardTone(task.status)
  const target = trainingEpochTarget(task)
  const current = Math.max(0, task.completed_epochs || 0)
  const percent = task.status === 'running' && target > 0 ? Math.round(trainingEpochProgress(task)) : null
  const metric = metricSummary(task)
  const earlyStop = earlyStoppingExplanation(task)
  const time = task.finished_at || task.updated_at
  useEffect(() => {
    if (dragging) {
      suppressClick.current = true
      return
    }
    if (!suppressClick.current) return
    const timer = window.setTimeout(() => { suppressClick.current = false }, 120)
    return () => window.clearTimeout(timer)
  }, [dragging])
  const openTask = () => {
    if (suppressClick.current) return
    if (!onNavigate()) return
    navigate('/tasks/' + encodeURIComponent(task.id) + '/canvas')
  }

  return <article className={'task-overview-card tone-' + tone} role="link" tabIndex={0} aria-label={cardAccessibilityLabel(task, current, target, percent)} onClick={openTask} onKeyDown={(event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openTask() }
  }}>
    <Handle type="target" position={Position.Left} className="task-overview-handle task-overview-handle-target" aria-label={'连接到 ' + task.name} onClick={(event) => event.stopPropagation()} />
    <Handle type="source" position={Position.Right} className="task-overview-handle task-overview-handle-source" aria-label={'从 ' + task.name + ' 创建关联'} onClick={(event) => event.stopPropagation()} />
    <span className="task-overview-card-top">
      <span className="task-overview-card-drag" title={'拖动任务卡片：' + task.name} aria-hidden="true" onClick={(event) => event.stopPropagation()}><GripHorizontal size={16} /></span>
      <span className="task-overview-card-icon" aria-hidden="true">{tone === 'attention' ? <AlertTriangle size={17} /> : tone === 'completed' ? <CheckCircle2 size={17} /> : task.status === 'running' ? <Zap size={17} /> : <Clock3 size={17} />}</span>
      <span className="task-overview-card-title" title={task.name}>{task.name}</span>
      <span className="task-overview-card-open" title={'打开训练画布：' + task.name} aria-hidden="true"><span>打开画布</span><ArrowUpRight size={16} /></span>
    </span>
    <span className="task-overview-card-context"><span className={'task-overview-status status-' + task.status}>{trainingStatusLabel(task)}</span><span className="task-overview-server" title={serverName}>{task.execution_node_id ? <Server size={13} /> : <Monitor size={13} />}{serverName}</span></span>
    {task.status === 'running' && <span className="task-overview-card-main">
      <span className="task-overview-card-row"><span>训练轮数</span><strong>{current} / {target > 0 ? target : '—'} Epoch</strong></span>
      <span className="task-overview-progress" role="progressbar" aria-label={task.name + ' 训练轮数进度'} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} aria-valuetext={percent === null ? '轮数目标未知' : current + ' / ' + target + ' Epoch'}><span style={{ width: (percent ?? 0) + '%' }} /></span>
      <span className="task-overview-card-foot"><span>{metric || '暂无验证指标'}</span><strong>{percent === null ? '—' : percent + '%'}</strong></span>
    </span>}
    {['queued', 'starting', 'stopping'].includes(task.status) && <span className="task-overview-card-main">
      <span className="task-overview-card-row"><span>{task.status === 'queued' ? '等待调度' : task.status === 'starting' ? '准备训练' : '正在结束'}</span><strong>{statusLabel(task.status)}</strong></span>
      <span className="task-overview-progress indeterminate" aria-hidden="true"><span /></span>
      <span className="task-overview-card-foot" title={task.queue_reason || ''}><span>{task.status === 'queued' ? task.queue_reason || '等待运行资源' : current > 0 ? '已记录 ' + current + ' Epoch' : '正在同步状态'}</span></span>
    </span>}
    {tone === 'attention' && <span className="task-overview-card-main attention-main">
      <span className="task-overview-error-label">{task.status === 'failed' ? '失败原因' : '中断说明'}</span>
      <span className="task-overview-error" title={task.error_message || ''}>{task.error_message || (task.status === 'failed' ? '平台未记录失败原因，请查看任务日志。' : '任务意外中断，请查看任务日志。')}</span>
      <span className="task-overview-card-foot"><span>最后记录 {current} / {target > 0 ? target : '—'} Epoch</span><span>{formatTime(time)}</span></span>
    </span>}
    {task.status === 'completed' && <span className="task-overview-card-main completed-main">
      <span className="task-overview-card-row"><span>完成轮数</span><strong>{current} / {target > 0 ? target : '—'} Epoch</strong></span>
      <span className="task-overview-card-result" title={earlyStop || metric || ''}>{earlyStop || metric || '暂无验证指标'}</span>
      <span className="task-overview-card-foot"><span>{earlyStop && metric ? metric : '完成时间'}</span><span>{formatTime(task.finished_at)}</span></span>
    </span>}
    {(task.status === 'draft' || task.status === 'stopped') && <span className="task-overview-card-main neutral-main">
      <span className="task-overview-card-row"><span>{task.status === 'draft' ? '尚未启动' : '已手动停止'}</span><strong>{current} / {target > 0 ? target : '—'} Epoch</strong></span>
      <span className="task-overview-card-foot"><span>{metric || '暂无验证指标'}</span><span>{formatTime(time)}</span></span>
    </span>}
  </article>
}, (before, after) => before.dragging === after.dragging && before.data.task.id === after.data.task.id && before.data.serverName === after.data.serverName && before.data.onNavigate === after.data.onNavigate && cardSignature(before.data.task) === cardSignature(after.data.task))

const nodeTypes = { taskOverview: TaskOverviewNode }

function normalizeEdges(layout: TaskWorkspaceModeLayout | undefined, validIds: Set<string>): OverviewEdge[] {
  return (layout?.edges || []).filter((edge) => edge.source !== edge.target && validIds.has(edge.source) && validIds.has(edge.target)).map((edge) => ({
    ...edge, type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed }, className: 'task-overview-edge',
  }))
}

function TaskOverviewCanvas({ mode, tasks, nodeNames, onNavigate, query, onDirtyChange, onFeedback }: Omit<Pick<Props, 'mode' | 'tasks' | 'nodeNames'>, never> & {
  onNavigate: () => boolean
  query: string
  onDirtyChange: (dirty: boolean) => void
  onFeedback: (feedback: LayoutFeedback | null) => void
}) {
  const preferences = useTaskCenterPreferences()
  const { fitView, getNodes, getViewport, setViewport } = useReactFlow<OverviewNode, OverviewEdge>()
  const nodesInitialized = useNodesInitialized()
  const [nodes, setNodes, onNodesChange] = useNodesState<OverviewNode>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<OverviewEdge>([])
  const [workspace, setWorkspace] = useState<TaskWorkspace | null>(null)
  const [saving, setSaving] = useState(false)
  const [selectedEdges, setSelectedEdges] = useState<string[]>([])
  const [layoutMode, setLayoutMode] = useState<'free' | 'grid'>(preferences.layoutMode)
  const loadedModeRef = useRef<string | null>(null)
  const restoringViewportRef = useRef(false)
  const dirtyRef = useRef(false)
  const normalizedQuery = query.trim().toLocaleLowerCase()
  const modeTasks = useMemo(() => tasks.filter((task) => matchesMode(task, mode)), [tasks, mode])
  const layoutTasks = useMemo(() => {
    const taskById = new Map(tasks.map((task) => [task.id, task]))
    const persistedOrder = (workspace?.modes[mode]?.nodes || []).map((node) => node.id)
    const persistedIds = new Set(persistedOrder)
    const persisted = persistedOrder.map((id) => taskById.get(id)).filter((task): task is TrainingTask => !!task)
    const appended = modeTasks.filter((task) => !persistedIds.has(task.id))
    return [...persisted, ...appended]
  }, [mode, modeTasks, tasks, workspace])
  const layoutTaskIds = useMemo(() => new Set(layoutTasks.map((task) => task.id)), [layoutTasks])
  const columns = useMemo(() => overviewColumns(window.innerWidth), [])

  const markDirty = useCallback(() => {
    if (dirtyRef.current) return
    dirtyRef.current = true
    onDirtyChange(true)
  }, [onDirtyChange])

  useEffect(() => {
    let active = true
    const controller = new AbortController()
    onFeedback({ kind: 'loading', message: '正在读取已保存布局…' })
    api.taskWorkspace(controller.signal).then((value) => {
      if (!active) return
      setWorkspace(value)
      onFeedback(null)
    }).catch((reason) => {
      if (!active || controller.signal.aborted) return
      setWorkspace({ version: 1, revision: 0, modes: {}, updated_at: null })
      onFeedback({ kind: 'error', message: reason instanceof Error ? reason.message : '布局读取失败，当前可继续使用临时布局' })
    })
    return () => { active = false; controller.abort() }
  }, [onFeedback])

  useEffect(() => {
    if (!workspace) return
    const savedLayout = workspace.modes[mode]
    const savedPositions = new Map((savedLayout?.nodes || []).map((item) => [item.id, item.position]))
    setNodes((current) => {
      const currentPositions = new Map(current.map((node) => [node.id, node.position]))
      return layoutTasks.map((task, index) => ({
        id: task.id,
        type: 'taskOverview',
        position: layoutMode === 'grid' ? defaultPosition(index, columns) : currentPositions.get(task.id) || savedPositions.get(task.id) || defaultPosition(index, columns),
        data: { task, serverName: taskServerName(task, nodeNames), onNavigate },
        draggable: layoutMode === 'free', selectable: false, dragHandle: '.task-overview-card-drag',
        style: { pointerEvents: 'all' },
      }))
    })
    if (loadedModeRef.current !== mode) {
      setEdges(normalizeEdges(savedLayout, layoutTaskIds))
      loadedModeRef.current = mode
      const viewport = savedLayout?.viewport || { x: 32, y: 32, zoom: preferences.defaultZoom }
      restoringViewportRef.current = true
      window.requestAnimationFrame(() => { void setViewport(viewport, { duration: 0 }).finally(() => { restoringViewportRef.current = false }) })
    } else setEdges((current) => current.filter((edge) => layoutTaskIds.has(edge.source) && layoutTaskIds.has(edge.target)))
  }, [columns, layoutMode, layoutTaskIds, layoutTasks, mode, nodeNames, onNavigate, preferences.defaultZoom, setEdges, setNodes, setViewport, workspace])

  useEffect(() => { setLayoutMode(preferences.layoutMode) }, [preferences.layoutMode])

  const visibleNodeIds = useMemo(() => new Set(modeTasks.filter((task) => !normalizedQuery || task.name.toLocaleLowerCase().includes(normalizedQuery) || taskServerName(task, nodeNames).toLocaleLowerCase().includes(normalizedQuery)).map((task) => task.id)), [modeTasks, nodeNames, normalizedQuery])
  const displayedNodes = useMemo(() => nodes.map((node) => ({ ...node, hidden: !visibleNodeIds.has(node.id), draggable: layoutMode === 'free' })), [layoutMode, nodes, visibleNodeIds])
  const displayedEdges = useMemo(() => edges.map((edge) => ({ ...edge, hidden: !preferences.showConnections || !visibleNodeIds.has(edge.source) || !visibleNodeIds.has(edge.target) })), [edges, preferences.showConnections, visibleNodeIds])

  const arrangeNodes = useCallback(() => {
    const arrangedIds = nodes.filter((node) => visibleNodeIds.has(node.id)).map((node) => node.id)
    setNodes((current) => current.map((node) => {
      const index = arrangedIds.indexOf(node.id)
      return index < 0 ? node : { ...node, position: defaultPosition(index, columns) }
    }))
    markDirty()
    window.requestAnimationFrame(() => {
      const visible = getNodes().filter((node) => visibleNodeIds.has(node.id))
      if (visible.length) void fitView({ nodes: visible, padding: 0.12, minZoom: 0.68, maxZoom: 1, duration: 260 })
    })
  }, [columns, fitView, getNodes, markDirty, nodes, setNodes, visibleNodeIds])

  const focusVisible = useCallback(() => {
    const visible = displayedNodes.filter((node) => visibleNodeIds.has(node.id))
    if (!visible.length) return
    markDirty()
    void fitView({ nodes: visible, padding: 0.12, minZoom: 0.68, maxZoom: 1, duration: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 260 })
  }, [displayedNodes, fitView, markDirty, visibleNodeIds])

  const resetViewport = useCallback(() => {
    markDirty()
    void setViewport({ x: 32, y: 32, zoom: preferences.defaultZoom }, { duration: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 220 })
  }, [markDirty, preferences.defaultZoom, setViewport])

  const onConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target || connection.source === connection.target) return
    if (edges.some((edge) => (edge.source === connection.source && edge.target === connection.target)
      || (edge.source === connection.target && edge.target === connection.source))) return
    setEdges((current) => addEdge({ ...connection, id: edgeId(), type: 'smoothstep', markerEnd: { type: MarkerType.ArrowClosed }, className: 'task-overview-edge' }, current))
    markDirty()
  }, [edges, markDirty, setEdges])

  const removeSelectedEdges = useCallback(() => {
    if (!selectedEdges.length) return
    setEdges((current) => current.filter((edge) => !selectedEdges.includes(edge.id)))
    setSelectedEdges([])
    markDirty()
  }, [markDirty, selectedEdges, setEdges])

  const toggleLayoutMode = useCallback(() => {
    setLayoutMode((current) => {
      if (current === 'free') {
        setNodes((items) => items.map((node, index) => ({ ...node, position: defaultPosition(index, columns) })))
        markDirty()
        return 'grid'
      }
      return 'free'
    })
  }, [columns, markDirty, setNodes])

  const saveLayout = useCallback(async () => {
    if (!workspace || saving) return
    setSaving(true)
    onFeedback({ kind: 'saving', message: '正在保存工作区布局…' })
    const modeLayout: TaskWorkspaceModeLayout = {
      nodes: nodes.map(({ id, position }) => ({ id, position })),
      edges: edges.map(({ id, source, target, label }) => ({ id, source, target, ...(typeof label === 'string' ? { label } : {}) })),
      viewport: getViewport(),
    }
    try {
      const saved = await api.saveTaskWorkspace({ revision: workspace.revision, modes: { ...workspace.modes, [mode]: modeLayout } })
      setWorkspace(saved)
      dirtyRef.current = false
      onDirtyChange(false)
      onFeedback({ kind: 'success', message: '工作区布局已保存' })
      window.setTimeout(() => onFeedback(null), 1800)
    } catch (reason) {
      const error = reason as Error & { status?: number }
      if (error.status === 409) {
        try {
          const latest = await api.taskWorkspace()
          setNodes([])
          setEdges([])
          setSelectedEdges([])
          setWorkspace(latest)
          loadedModeRef.current = null
          dirtyRef.current = false
          onDirtyChange(false)
          onFeedback({ kind: 'error', message: '其他页面已更新布局，已载入最新版本，请重新调整后保存' })
          return
        } catch { /* keep the original conflict message */ }
      }
      onFeedback({ kind: 'error', message: error.message || '布局保存失败，请重试' })
    } finally {
      setSaving(false)
    }
  }, [edges, getViewport, mode, nodes, onDirtyChange, onFeedback, saving, setEdges, setNodes, workspace])

  const handleNodesChange = useCallback((changes: Parameters<typeof onNodesChange>[0]) => {
    onNodesChange(changes)
    if (changes.some((change) => change.type === 'position' && change.dragging === false)) markDirty()
  }, [markDirty, onNodesChange])

  const handleEdgesChange = useCallback((changes: Parameters<typeof onEdgesChange>[0]) => {
    onEdgesChange(changes)
    const selected = new Set(selectedEdges)
    for (const change of changes) {
      if (change.type === 'select') {
        if (change.selected) selected.add(change.id)
        else selected.delete(change.id)
      }
      if (change.type === 'remove') { selected.delete(change.id); markDirty() }
    }
    setSelectedEdges([...selected])
  }, [markDirty, onEdgesChange, selectedEdges])

  const handleMoveEnd = useCallback((event: MouseEvent | TouchEvent | null, viewport: Viewport) => {
    if (!event || restoringViewportRef.current || !workspace) return
    const saved = workspace.modes[mode]?.viewport
    if (!saved || Math.abs(saved.x - viewport.x) > 0.5 || Math.abs(saved.y - viewport.y) > 0.5 || Math.abs(saved.zoom - viewport.zoom) > 0.002) markDirty()
  }, [markDirty, mode, workspace])

  if (!workspace) return <div className="task-overview-empty" role="status"><LoaderCircle className="spin" size={20} />正在装载任务工作区…</div>
  if (modeTasks.length === 0) return <div className="task-overview-empty" role="status">当前没有{modeTitles[mode]}的任务。</div>
  if (visibleNodeIds.size === 0) return <div className="task-overview-empty" role="status">没有找到匹配的任务。</div>

  return <div className={'task-overview-flow density-' + preferences.cardDensity} aria-label={modeTitles[mode] + '任务画布'}>
    <ReactFlow nodes={displayedNodes} edges={displayedEdges} nodeTypes={nodeTypes} onNodesChange={handleNodesChange} onEdgesChange={handleEdgesChange} onConnect={onConnect} onMoveEnd={handleMoveEnd} nodesDraggable={layoutMode === 'free'} nodesConnectable={layoutMode === 'free' && preferences.showConnections} nodesFocusable elementsSelectable panOnDrag selectionOnDrag={false} deleteKeyCode={null} zoomOnScroll zoomOnPinch zoomOnDoubleClick={false} minZoom={0.4} maxZoom={1.45} defaultViewport={{ x: 32, y: 32, zoom: preferences.defaultZoom }} onlyRenderVisibleElements={nodes.length > 20} attributionPosition="bottom-left">
      <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} />
      <Controls showInteractive={false} showFitView={false} position="bottom-right" />
      {preferences.showMiniMap && <MiniMap position="bottom-right" pannable zoomable nodeStrokeWidth={2} nodeColor={(node) => {
        const task = (node.data as OverviewData).task
        const tone = cardTone(task.status)
        return tone === 'attention' ? '#e98585' : tone === 'completed' ? '#6ea9df' : tone === 'active' ? '#62d8aa' : '#8e9aa3'
      }} />}
    </ReactFlow>
    <div className="task-overview-canvas-actions" aria-label="画布布局工具">
      <button type="button" onClick={toggleLayoutMode} aria-pressed={layoutMode === 'free'} title={layoutMode === 'free' ? '切换为固定网格模式' : '切换为自由编辑模式'}>{layoutMode === 'free' ? <Link2 size={14} /> : <Grid2X2 size={14} />}{layoutMode === 'free' ? '自由编辑' : '固定网格'}</button>
      <button type="button" onClick={arrangeNodes}><Grid2X2 size={14} />自动排列</button>
      <button type="button" onClick={focusVisible}><Maximize2 size={14} />适配可见任务</button>
      <button type="button" onClick={resetViewport}><LocateFixed size={14} />回到首屏</button>
      <button type="button" disabled={!selectedEdges.length} onClick={removeSelectedEdges}><Trash2 size={14} />删除连线</button>
      <button type="button" className="primary" disabled={saving} onClick={() => void saveLayout()}>{saving ? <LoaderCircle className="spin" size={14} /> : <Save size={14} />}{saving ? '正在保存' : '保存布局'}</button>
    </div>
    <span className="task-overview-canvas-tip"><GripHorizontal size={13} />拖动卡片顶部把手 · 从卡片两侧连接点创建关联 · 点击卡片进入训练画布</span>
    {!nodesInitialized && <span className="task-overview-initializing"><LoaderCircle className="spin" size={14} />正在绘制画布</span>}
  </div>
}

export function TaskOverviewModal({ mode, tasks, nodeNames, onClose, onNavigate }: Props) {
  const [query, setQuery] = useState('')
  const [dirty, setDirty] = useState(false)
  const [feedback, setFeedback] = useState<LayoutFeedback | null>(null)
  const requestClose = useCallback(() => {
    if (dirty && !window.confirm('任务工作区布局尚未保存，确定关闭并放弃本次调整吗？')) return
    onClose()
  }, [dirty, onClose])
  const requestNavigate = useCallback(() => {
    if (dirty && !window.confirm('任务工作区布局尚未保存，确定离开并放弃本次调整吗？')) return false
    onNavigate()
    return true
  }, [dirty, onNavigate])
  const dialogRef = useModalFocus<HTMLDivElement>(true, requestClose, { initialFocusSelector: '[data-modal-initial-focus]' })
  const count = tasks.filter((task) => matchesMode(task, mode)).length

  return <div className="modal-backdrop task-overview-backdrop" onMouseDown={requestClose}>
    <div ref={dialogRef} className="modal task-overview-dialog" role="dialog" aria-modal="true" aria-labelledby="task-overview-title" aria-describedby="task-overview-description" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
      <header className="task-overview-header"><div><span className="eyebrow">TRAINING WORKSPACE</span><h2 id="task-overview-title">{modeTitles[mode]} <span>{count}</span>{dirty && <em>未保存</em>}</h2><p id="task-overview-description">{modeDescriptions[mode]}</p></div><div className="task-overview-header-state">{feedback && <span className={'task-overview-feedback ' + feedback.kind} role={feedback.kind === 'error' ? 'alert' : 'status'}>{feedback.kind === 'loading' || feedback.kind === 'saving' ? <LoaderCircle className="spin" size={14} /> : feedback.kind === 'success' ? <Check size={14} /> : <AlertTriangle size={14} />}{feedback.message}</span>}<button type="button" className="task-overview-close" data-modal-initial-focus onClick={requestClose} aria-label="关闭任务总览"><X size={19} /></button></div></header>
      <div className="task-overview-toolbar"><label><Search size={15} aria-hidden="true" /><input type="search" aria-label="搜索任务名称或执行服务器" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索任务或服务器" /></label><span>{query.trim() ? '筛选仅影响显示，不会删除已保存位置' : '布局按当前分类独立保存'}</span></div>
      <ReactFlowProvider><TaskOverviewCanvas mode={mode} tasks={tasks} nodeNames={nodeNames} onNavigate={requestNavigate} query={query} onDirtyChange={setDirty} onFeedback={setFeedback} /></ReactFlowProvider>
    </div>
  </div>
}
