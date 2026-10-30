import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, ArrowUpRight, CheckCircle2, Clock3, Maximize2, Monitor, Search, Server, X, Zap } from 'lucide-react'
import { Background, BackgroundVariant, Controls, ReactFlow, ReactFlowProvider, useNodesInitialized, useReactFlow, type Node, type NodeProps } from '@xyflow/react'
import { statusLabel } from '../api'
import { taskMetric } from '../groupMetrics'
import { primaryMetric } from '../metricProfiles'
import { earlyStoppingExplanation, trainingEpochProgress, trainingEpochTarget, trainingStatusLabel } from '../trainingCompletion'
import type { TrainingTask } from '../types'
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

type OverviewData = {
  task: TrainingTask
  serverName: string
  onNavigate: () => void
}

type OverviewNode = Node<OverviewData, 'taskOverview'>

const activeStatuses = new Set<TrainingTask['status']>(['queued', 'starting', 'running', 'stopping'])
const attentionStatuses = new Set<TrainingTask['status']>(['failed', 'interrupted'])
const modeTitles: Record<TaskOverviewMode, string> = {
  all: '全部任务',
  active: '正在进行',
  completed: '已完成',
  attention: '需要关注',
}
const modeDescriptions: Record<TaskOverviewMode, string> = {
  all: '查看全部任务状态，点击卡片进入任务详情。',
  active: '训练进度按已完成轮数计算；排队等阶段不显示虚构的百分比。',
  completed: '查看已结束任务的轮数、完成时间和最新验证指标。',
  attention: '失败与意外中断分别标记，可进入详情排查。',
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
    task.completion_info?.best_epoch, task.completion_info?.patience,
    task.latest_metrics,
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

const TaskOverviewNode = memo(function TaskOverviewNode({ data }: NodeProps<OverviewNode>) {
  const { task, serverName, onNavigate } = data
  const tone = cardTone(task.status)
  const target = trainingEpochTarget(task)
  const current = Math.max(0, task.completed_epochs || 0)
  const percent = task.status === 'running' && target > 0 ? Math.round(trainingEpochProgress(task)) : null
  const metric = metricSummary(task)
  const earlyStop = earlyStoppingExplanation(task)
  const time = task.finished_at || task.updated_at

  return <Link to={'/tasks/' + encodeURIComponent(task.id) + '/canvas'} draggable={false} className={'task-overview-card nodrag nopan tone-' + tone} onClick={(event) => {
    if (event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) onNavigate()
  }} aria-label={cardAccessibilityLabel(task, current, target, percent)}>
    <span className="task-overview-card-top">
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
  </Link>
}, (before, after) => before.data.task.id === after.data.task.id && before.data.serverName === after.data.serverName && before.data.onNavigate === after.data.onNavigate && cardSignature(before.data.task) === cardSignature(after.data.task))

const nodeTypes = { taskOverview: TaskOverviewNode }

function TaskOverviewCanvas({ mode, tasks, nodeNames, onNavigate, query }: Pick<Props, 'mode' | 'tasks' | 'nodeNames' | 'onNavigate'> & { query: string }) {
  const { fitView, setViewport } = useReactFlow<OverviewNode>()
  const nodesInitialized = useNodesInitialized()
  const lastCentered = useRef<string | null>(null)
  const columns = useRef(Math.max(1, Math.min(3, Math.floor((Math.min(window.innerWidth - 32, 1120) - 20) / 320))))
  const normalizedQuery = query.trim().toLocaleLowerCase()
  const visibleTasks = useMemo(() => tasks.filter((task) => matchesMode(task, mode) && (!normalizedQuery || task.name.toLocaleLowerCase().includes(normalizedQuery) || taskServerName(task, nodeNames).toLocaleLowerCase().includes(normalizedQuery))), [tasks, mode, nodeNames, normalizedQuery])
  // Positions depend on membership, not live status values. Removing a task
  // compacts the grid; ordinary polling keeps the same layout and viewport.
  const membershipKey = visibleTasks.map((task) => task.id).join('\u001f')
  const nodes = useMemo<OverviewNode[]>(() => visibleTasks.map((task, index) => ({
    id: task.id,
    type: 'taskOverview',
    position: { x: index % columns.current * 328, y: Math.floor(index / columns.current) * 234 },
    data: { task, serverName: taskServerName(task, nodeNames), onNavigate },
    draggable: false,
    selectable: false,
    // React Flow disables pointer events on static nodes without node handlers.
    // Keep the card link interactive while the canvas remains pannable.
    style: { pointerEvents: 'all' },
  })), [visibleTasks, nodeNames, onNavigate])

  useEffect(() => {
    if (membershipKey === lastCentered.current || (visibleTasks.length <= 6 && !nodesInitialized)) return
    lastCentered.current = membershipKey
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (visibleTasks.length <= 6 && visibleTasks.length > 0) {
      void fitView({ padding: 0.14, maxZoom: 1, duration: reduceMotion ? 0 : 180 })
    } else {
      // A long historical list opens at readable size instead of shrinking
      // hundreds of cards into an unusable whole-canvas fit.
      void setViewport({ x: 24, y: 24, zoom: 1 }, { duration: reduceMotion ? 0 : 180 })
    }
  }, [fitView, membershipKey, nodesInitialized, setViewport, visibleTasks.length])

  const resetViewport = () => {
    const duration = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 180
    if (visibleTasks.length <= 6) void fitView({ padding: 0.14, maxZoom: 1, duration })
    else void setViewport({ x: 24, y: 24, zoom: 1 }, { duration })
  }

  if (nodes.length === 0) return <div className="task-overview-empty" role="status">{normalizedQuery ? '没有找到匹配的任务。' : '当前没有' + modeTitles[mode] + '的任务。'}</div>

  return <div className="task-overview-flow" aria-label={modeTitles[mode] + '任务画布'}>
    <ReactFlow nodes={nodes} edges={[]} nodeTypes={nodeTypes} nodesDraggable={false} nodesConnectable={false} nodesFocusable={false} elementsSelectable={false} panOnDrag zoomOnScroll zoomOnPinch zoomOnDoubleClick={false} minZoom={0.25} maxZoom={1.5} defaultViewport={{ x: 24, y: 24, zoom: 1 }} onlyRenderVisibleElements={nodes.length > 12} attributionPosition="bottom-left">
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} />
      <Controls showInteractive={false} showFitView={false} position="bottom-right" />
    </ReactFlow>
    <span className="task-overview-canvas-tip"><Maximize2 size={13} />拖动画布 · 滚轮缩放 · 点击卡片或箭头打开训练画布</span>
    <button type="button" className="task-overview-reset" onClick={resetViewport} aria-label={visibleTasks.length <= 6 ? '重新适配所有任务' : '回到任务列表首屏'}>{visibleTasks.length <= 6 ? '适配画布' : '回到首屏'}</button>
  </div>
}

export function TaskOverviewModal({ mode, tasks, nodeNames, onClose, onNavigate }: Props) {
  const [query, setQuery] = useState('')
  const dialogRef = useModalFocus<HTMLDivElement>(true, onClose, { initialFocusSelector: '[data-modal-initial-focus]' })
  const count = tasks.filter((task) => matchesMode(task, mode)).length

  return <div className="modal-backdrop task-overview-backdrop" onMouseDown={onClose}>
    <div ref={dialogRef} className="modal task-overview-dialog" role="dialog" aria-modal="true" aria-labelledby="task-overview-title" aria-describedby="task-overview-description" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
      <header className="task-overview-header"><div><span className="eyebrow">TRAINING OVERVIEW</span><h2 id="task-overview-title">{modeTitles[mode]} <span>{count}</span></h2><p id="task-overview-description">{modeDescriptions[mode]}</p></div><button type="button" className="task-overview-close" data-modal-initial-focus onClick={onClose} aria-label="关闭任务总览"><X size={19} /></button></header>
      <div className="task-overview-toolbar"><label><Search size={15} aria-hidden="true" /><input type="search" aria-label="搜索任务名称或执行服务器" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索任务或服务器" /></label><span>{query.trim() ? '已筛选' : '拖动浏览更多任务'}</span></div>
      <ReactFlowProvider><TaskOverviewCanvas mode={mode} tasks={tasks} nodeNames={nodeNames} onNavigate={onNavigate} query={query} /></ReactFlowProvider>
    </div>
  </div>
}
