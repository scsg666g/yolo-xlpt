import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ArtifactDownload } from './ArtifactDownload'
import { Background, Controls, MarkerType, MiniMap, ReactFlow, ReactFlowProvider, addEdge, useEdgesState, useNodesState, useReactFlow, type Connection, type Edge, type Node } from '@xyflow/react'
import { Activity, AlertCircle, ArrowLeft, Box, Braces, Check, Clock3, Cpu, Database, Download, FileArchive, FileText, Gauge, History, LoaderCircle, Maximize, Play, Radio, RefreshCw, RotateCcw, Save, ScanEye, SlidersHorizontal, Square, Terminal, Trophy, Workflow, Zap } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'
import { api, formatBytes, metricValue } from '../api'
import { earlyStoppingExplanation, trainingEpochProgress, trainingEpochTarget, trainingStatusLabel } from '../trainingCompletion'
import { useAppearance } from '../appearance'
import type { Artifact, EffectiveBestWeight, MetricPoint, TaskEvent, TrainingRun, TrainingTask } from '../types'
import { CanvasNode, type PanelNodeData } from './CanvasNode'
import { CanvasDetailModal, type CanvasDetailKind } from './CanvasDetailModal'
import { LineChartView } from './LineChart'
import { EditParametersModal } from './EditParametersModal'
import { EditResourceModal } from './EditResourceModal'
import { LogViewerModal } from './LogViewerModal'
import { ModelTestModal } from './ModelTestModal'
import { isSupportedModelTestTask, modelTestMetrics } from './ModelTestShared'
import { TrainingTraceModal } from './TrainingTraceModal'

const nodeTypes = { panel: CanvasNode }
const initialEdges: Edge[] = [
  ['dataset', 'config'], ['model', 'config'], ['config', 'status'], ['runtime', 'status'],
  ['status', 'metrics'], ['status', 'logs'], ['status', 'weights'], ['status', 'trace'], ['metrics', 'artifacts'], ['weights', 'inference'],
].map(([source, target], index) => ({
  id: `edge-${index}`, source, target, markerEnd: { type: MarkerType.ArrowClosed },
}))
const positions: Record<string, { x: number; y: number }> = {
  dataset: { x: 20, y: 80 }, model: { x: 20, y: 420 }, config: { x: 400, y: 60 }, runtime: { x: 400, y: 430 },
  weights: { x: 800, y: -150 }, status: { x: 800, y: 170 }, trace: { x: 800, y: 650 }, metrics: { x: 1180, y: 20 }, logs: { x: 1180, y: 430 }, artifacts: { x: 1580, y: 180 },
  inference: { x: 1180, y: -250 },
}

function row(label: string, value: React.ReactNode) { return <div className="node-row"><span>{label}</span><strong>{value}</strong></div> }

function edgeIsActive(source: string, target: string, status?: string) {
  if (status === 'running') return true
  if (status === 'queued') return source === 'runtime' && target === 'status'
  if (status === 'starting') return target === 'config' || target === 'status'
  if (status === 'stopping') return source === 'status' && target === 'logs'
  return false
}

function chartOption(metrics: MetricPoint[], animate: boolean) {
  const values = (key: string) => metrics.map((point) => typeof point.values[key] === 'number' ? point.values[key] : null)
  return {
    animation: animate,
    animationDuration: 280,
    animationDurationUpdate: 280,
    grid: { left: 36, right: 12, top: 30, bottom: 28 },
    tooltip: { trigger: 'axis', backgroundColor: '#111a24', borderColor: '#2b3a49', textStyle: { color: '#d9e4ed' } },
    legend: { data: ['mAP50', 'mAP50-95'], textStyle: { color: '#8493a2', fontSize: 10 }, top: 2 },
    xAxis: { type: 'category', data: metrics.map((point) => point.epoch), axisLabel: { color: '#667685' }, axisLine: { lineStyle: { color: '#273542' } } },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#667685' }, splitLine: { lineStyle: { color: '#1b2732' } } },
    series: [
      { name: 'mAP50', type: 'line', showSymbol: false, smooth: true, data: values('metrics/mAP50(B)'), lineStyle: { color: '#70f0bd', width: 2 }, areaStyle: { color: 'rgba(112,240,189,.08)' } },
      { name: 'mAP50-95', type: 'line', showSymbol: false, smooth: true, data: values('metrics/mAP50-95(B)'), lineStyle: { color: '#6da8ff', width: 2 } },
    ],
  }
}

function TaskCanvasInner() {
  const { taskId = '' } = useParams()
  const navigate = useNavigate()
  const { resolvedTheme } = useAppearance()
  const [task, setTask] = useState<TrainingTask | null>(null)
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [metrics, setMetrics] = useState<MetricPoint[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [effectiveBestWeight, setEffectiveBestWeight] = useState<EffectiveBestWeight | null>(null)
  const [events, setEvents] = useState<TaskEvent[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [initialLoadError, setInitialLoadError] = useState('')
  const [working, setWorking] = useState(false)
  const [notice, setNotice] = useState('')
  const [syncError, setSyncError] = useState('')
  const [layoutFeedback, setLayoutFeedback] = useState<{ kind: 'saving' | 'success' | 'error'; message: string } | null>(null)
  const [editingParameters, setEditingParameters] = useState(false)
  const [editingResource, setEditingResource] = useState<'dataset' | 'model' | null>(null)
  const [viewingLogs, setViewingLogs] = useState(false)
  const [detailKind, setDetailKind] = useState<CanvasDetailKind | null>(null)
  const [testingArtifact, setTestingArtifact] = useState<Artifact | null>(null)
  const [modelTestOpen, setModelTestOpen] = useState(false)
  const [viewingTrace, setViewingTrace] = useState(false)
  const [pageVisible, setPageVisible] = useState(() => document.visibilityState !== 'hidden')
  const [reduceMotion, setReduceMotion] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [lastSyncedAt, setLastSyncedAt] = useState(Date.now())
  const [clock, setClock] = useState(Date.now())
  const [effectsEnabled, setEffectsEnabled] = useState(() => {
    try { return localStorage.getItem('yolo-canvas-effects') !== 'off' } catch { return true }
  })
  const [epochPulse, setEpochPulse] = useState<'a' | 'b' | undefined>()
  const [logPulse, setLogPulse] = useState<'a' | 'b' | undefined>()
  const [artifactPulse, setArtifactPulse] = useState<'a' | 'b' | undefined>()
  const [newArtifactIds, setNewArtifactIds] = useState<number[]>([])
  const [lastOutputAt, setLastOutputAt] = useState(Date.now())
  const previousUpdate = useRef<{ runId: string; epoch: number; logs: string; status: string; artifacts: Map<number, string> } | null>(null)
  const requestVersion = useRef(0)
  const pendingRequest = useRef<{ version: number; controller: AbortController } | null>(null)
  const mutationPending = useRef(false)
  const hasLoadedTask = useRef(false)
  const layoutSavePending = useRef(false)
  const layoutSaveController = useRef<AbortController | null>(null)
  const layoutFeedbackTimer = useRef<number | null>(null)
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(initialEdges)
  const { fitView, getViewport } = useReactFlow()

  const cancelPendingRefresh = useCallback(() => {
    requestVersion.current += 1
    pendingRequest.current?.controller.abort()
    pendingRequest.current = null
  }, [])

  const refresh = useCallback(async (options: { allowDuringMutation?: boolean; preferredRunId?: string } = {}) => {
    if ((mutationPending.current && !options.allowDuringMutation) || pendingRequest.current !== null) return
    const version = ++requestVersion.current
    const controller = new AbortController()
    pendingRequest.current = { version, controller }
    let resolvedSelectedId: string | undefined
    try {
      // Validate the task first so a stale canvas URL does not fire several
      // dependent 404 requests on every polling cycle.
      const [taskData, runData] = await Promise.all([api.task(taskId, controller.signal), api.runs(taskId, controller.signal)])
      if (version !== requestVersion.current) return
      hasLoadedTask.current = true
      setInitialLoadError('')
      const requestedRunId = options.preferredRunId ?? selectedRunId
      const selectedId = runData.some((run) => run.id === requestedRunId)
        ? requestedRunId
        : (taskData.current_run_id || runData[0]?.id || '')
      resolvedSelectedId = selectedId
      const selectedRun = runData.find((run) => run.id === selectedId)
      setTask(taskData); setRuns(runData)
      if (!selectedRun) {
        if (selectedId !== selectedRunId) setSelectedRunId(selectedId)
        setMetrics([]); setArtifacts([]); setEvents([]); setLogs([])
        setEffectiveBestWeight(null)
        setSyncError('')
        setLastSyncedAt(Date.now())
        return
      }
      const [metricData, artifactData, effectiveBestWeightData, eventData, logData] = await Promise.all([
        api.runMetrics(taskId, selectedRun.id, controller.signal), api.runArtifacts(taskId, selectedRun.id, controller.signal),
        api.runEffectiveBestWeight(taskId, selectedRun.id, controller.signal), api.runEvents(taskId, selectedRun.id, controller.signal),
        api.runLogs(taskId, selectedRun.id, controller.signal),
      ])
      if (version !== requestVersion.current) return
      const previous = previousUpdate.current
      const currentEpoch = metricData.at(-1)?.epoch || selectedRun.completed_epochs
      const isLiveRun = selectedRun.id === taskData.current_run_id
      const logSignature = logData.lines.join('\n')
      const artifactSignatures = new Map(artifactData.map((artifact) => [artifact.id, `${artifact.modified_at}:${artifact.size}`]))
      const sameRun = previous?.runId === selectedRun.id
      const changedArtifacts = sameRun ? artifactData.filter((artifact) => previous.artifacts.get(artifact.id) !== artifactSignatures.get(artifact.id)).map((artifact) => artifact.id) : []
      if (!sameRun || currentEpoch !== previous.epoch || logSignature !== previous.logs || selectedRun.status !== previous.status || changedArtifacts.length) {
        setLastOutputAt(Date.now())
      }
      if (previous?.runId === selectedRun.id && isLiveRun) {
        if (currentEpoch > previous.epoch) setEpochPulse((value) => value === 'a' ? 'b' : 'a')
        if (logSignature !== previous.logs) setLogPulse((value) => value === 'a' ? 'b' : 'a')
        if (changedArtifacts.length) {
          setArtifactPulse((value) => value === 'a' ? 'b' : 'a')
          setNewArtifactIds(changedArtifacts)
        }
      }
      previousUpdate.current = {
        runId: selectedRun.id,
        epoch: currentEpoch,
        logs: logSignature,
        status: selectedRun.status,
        artifacts: artifactSignatures,
      }
      if (selectedId !== selectedRunId) setSelectedRunId(selectedId)
      setMetrics(metricData); setArtifacts(artifactData); setEvents(eventData); setLogs(logData.lines)
      setEffectiveBestWeight(effectiveBestWeightData)
      setSyncError('')
      setLastSyncedAt(Date.now())
    } catch (reason) {
      if (controller.signal.aborted || version !== requestVersion.current) return
      if (resolvedSelectedId !== undefined && resolvedSelectedId !== selectedRunId) setSelectedRunId(resolvedSelectedId)
      const status = reason instanceof Error ? (reason as Error & { status?: number }).status : undefined
      if (status === 404) {
        navigate('/', { replace: true })
        return
      }
      const message = reason instanceof Error ? reason.message : '加载失败'
      if (hasLoadedTask.current) setSyncError(message)
      else setInitialLoadError(message)
    } finally {
      if (pendingRequest.current?.version === version) pendingRequest.current = null
      if (version === requestVersion.current) setLoading(false)
    }
  }, [navigate, selectedRunId, taskId])

  useEffect(() => {
    const onVisibility = () => setPageVisible(document.visibilityState !== 'hidden')
    document.addEventListener('visibilitychange', onVisibility)
    return () => document.removeEventListener('visibilitychange', onVisibility)
  }, [])

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const onChange = () => setReduceMotion(media.matches)
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  useEffect(() => {
    if (!pageVisible) return
    refresh()
    const timer = window.setInterval(refresh, 3000)
    return () => {
      window.clearInterval(timer)
      cancelPendingRefresh()
    }
  }, [cancelPendingRefresh, pageVisible, refresh])

  useEffect(() => {
    previousUpdate.current = null
    setEpochPulse(undefined); setLogPulse(undefined); setArtifactPulse(undefined)
    setNewArtifactIds([])
    setMetrics([]); setArtifacts([]); setEffectiveBestWeight(null); setEvents([]); setLogs([])
  }, [taskId, selectedRunId])

  useEffect(() => {
    if (!newArtifactIds.length) return
    const timer = window.setTimeout(() => setNewArtifactIds([]), 4000)
    return () => window.clearTimeout(timer)
  }, [newArtifactIds])

  useEffect(() => () => {
    if (layoutFeedbackTimer.current !== null) window.clearTimeout(layoutFeedbackTimer.current)
    layoutSaveController.current?.abort()
  }, [])

  useEffect(() => {
    try { localStorage.setItem('yolo-canvas-effects', effectsEnabled ? 'on' : 'off') } catch { /* Storage may be disabled. */ }
  }, [effectsEnabled])

  useEffect(() => {
    if (!pageVisible) return
    const timer = window.setInterval(() => setClock(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [pageVisible])

  const selectedRun = useMemo(() => runs.find((run) => run.id === selectedRunId) || null, [runs, selectedRunId])
  const isCurrentRun = !selectedRun || selectedRun.id === task?.current_run_id
  const viewTask = useMemo(() => {
    if (!task || !selectedRun) return task
    const snapshot = selectedRun.parameter_snapshot || {}
    const snapshotConfig = (snapshot.config as Record<string, unknown> | undefined) || task.config || {}
    return {
      ...task,
      status: selectedRun.status,
      model_path: selectedRun.model_path,
      data_path: selectedRun.data_path,
      run_dir: selectedRun.run_dir,
      log_path: selectedRun.log_path,
      epochs: Number(snapshot.epochs ?? task.epochs),
      batch: Number(snapshot.batch ?? task.batch),
      imgsz: Number(snapshot.imgsz ?? task.imgsz),
      device: String(snapshot.device ?? task.device),
      workers: Number(snapshot.workers ?? task.workers),
      optimizer: String(snapshot.optimizer ?? task.optimizer),
      config: snapshotConfig,
      completed_epochs: selectedRun.completed_epochs,
      progress: selectedRun.progress,
      pid: selectedRun.pid,
      exit_code: selectedRun.exit_code,
      error_message: selectedRun.error_message,
      completion_info: selectedRun.completion_info,
      latest_metrics: selectedRun.latest_metrics,
      queued_at: selectedRun.queued_at,
      started_at: selectedRun.started_at,
      finished_at: selectedRun.finished_at,
    } satisfies TrainingTask
  }, [selectedRun, task])
  const latest = useMemo(() => viewTask?.latest_metrics || {}, [viewTask])
  const weightArtifacts = useMemo(() => artifacts.filter((artifact) => artifact.kind === 'weight'), [artifacts])
  const regularArtifacts = useMemo(() => artifacts.filter((artifact) => artifact.kind !== 'weight'), [artifacts])
  const directBestWeight = weightArtifacts.find((artifact) => artifact.name.split('/').pop() === 'best.pt')
  const bestWeight = directBestWeight || effectiveBestWeight?.artifact || undefined
  const inheritedBestWeight = !directBestWeight && !!effectiveBestWeight?.inherited
  const bestWeightOrigin = inheritedBestWeight && effectiveBestWeight?.source_attempt_number
    ? `继承自 Attempt ${String(effectiveBestWeight.source_attempt_number).padStart(3, '0')}` : ''
  const modelTestAttempt = inheritedBestWeight ? effectiveBestWeight?.source_attempt_number : selectedRun?.attempt_number
  const modelTestBestMetric = useMemo(() => {
    if (inheritedBestWeight || !isSupportedModelTestTask(viewTask?.task_type)) return null
    const metricKey = modelTestMetrics[viewTask.task_type].key
    let best: { epoch: number; value: number } | null = null
    for (const point of metrics) {
      const value = point.values[metricKey]
      if (typeof value === 'number' && Number.isFinite(value) && (!best || value > best.value)) best = { epoch: point.epoch, value }
    }
    return best
  }, [inheritedBestWeight, metrics, viewTask?.task_type])
  const lastWeight = weightArtifacts.find((artifact) => artifact.name.split('/').pop() === 'last.pt')
  const testDisabledReason = !isSupportedModelTestTask(viewTask?.task_type) ? '当前模型任务类型暂不支持图片测试'
    : ['queued', 'starting', 'running', 'stopping'].includes(viewTask?.status || '') ? '训练结束或停止后可测试' : null
  const syncAge = Math.max(0, Math.floor((clock - lastSyncedAt) / 1000))
  const outputAge = Math.max(0, Math.floor((clock - lastOutputAt) / 1000))
  const syncDelayed = pageVisible && syncAge > 15
  const currentIsActive = isCurrentRun && ['queued', 'starting', 'running', 'stopping'].includes(viewTask?.status || '')
  const outputWaiting = isCurrentRun && viewTask?.status === 'running' && outputAge > 120
  const motionAllowed = isCurrentRun && effectsEnabled && pageVisible && !reduceMotion && !syncDelayed && !outputWaiting
  const nodeActivity = useMemo<PanelNodeData['activity']>(() => {
    if (!viewTask || !isCurrentRun) return undefined
    if (['queued', 'starting', 'running', 'stopping', 'stopped', 'interrupted', 'failed', 'completed'].includes(viewTask.status)) return viewTask.status as PanelNodeData['activity']
    return undefined
  }, [isCurrentRun, viewTask])
  const displayedDataPath = isCurrentRun && task ? task.data_path : viewTask?.data_path || ''
  const displayedModelPath = isCurrentRun && task ? task.model_path : viewTask?.model_path || ''
  const dataPathChanged = Boolean(isCurrentRun && selectedRun && task && selectedRun.data_path !== task.data_path)
  const modelPathChanged = Boolean(isCurrentRun && selectedRun && task && selectedRun.model_path !== task.model_path)
  const resourceConfig = useMemo(
    () => isCurrentRun && task ? task.config : viewTask?.config || {},
    [isCurrentRun, task, viewTask],
  )
  const remoteResource = Boolean(task?.execution_node_id)
  const nodeData = useMemo<Record<string, PanelNodeData> | null>(() => viewTask ? {
    dataset: { title: displayedDataPath.split('/').pop() || '数据集', kicker: `DATASET${!isCurrentRun ? ' · 历史快照' : dataPathChanged ? ' · 重新训练使用' : ' · 双击编辑'}`, icon: <Database size={17} />, content: <><p className="mono path-value">{displayedDataPath}</p>{dataPathChanged && selectedRun && <div className="resource-snapshot-note"><span>当前 Attempt 使用</span><code>{selectedRun.data_path}</code><small>重新训练将使用上方新路径</small></div>}{remoteResource ? row('路径校验', <span className="check-pending">Agent 启动时检查</span>) : row('配置状态', resourceConfig.path_issues && (resourceConfig.path_issues as { data_missing?: boolean }).data_missing ? <span className="bad">路径失效</span> : <span className="good">可用</span>)}{isCurrentRun && <div className="node-edit-hint"><SlidersHorizontal size={12} />双击更换训练数据集</div>}</> },
    model: { title: displayedModelPath.split('/').pop() || '基础模型', kicker: `MODEL${!isCurrentRun ? ' · 历史快照' : modelPathChanged ? ' · 重新训练使用' : ' · 双击编辑'}`, icon: <Box size={17} />, accent: '#6da8ff', content: <><p className="mono path-value">{displayedModelPath}</p>{modelPathChanged && selectedRun && <div className="resource-snapshot-note"><span>当前 Attempt 使用</span><code>{selectedRun.model_path}</code><small>重新训练将使用上方新路径</small></div>}{remoteResource && row('路径校验', <span className="check-pending">Agent 启动时检查</span>)}{row('任务类型', viewTask.task_type)}{row('预训练', viewTask.config.pretrained === false ? '关闭' : '启用')}{isCurrentRun && <div className="node-edit-hint"><SlidersHorizontal size={12} />双击更换基础模型</div>}</> },
    config: { title: '超参数配置', kicker: `CONFIGURATION${isCurrentRun ? ' · 双击编辑' : ' · 历史快照'}`, icon: <Braces size={17} />, accent: '#d99cff', content: <><div className="node-grid">{row('Epochs', viewTask.epochs)}{row('Batch', viewTask.batch)}{row('Image', viewTask.imgsz)}{row('Optimizer', viewTask.optimizer)}{row('Workers', viewTask.workers)}{row('Patience', String(viewTask.config.patience ?? '—'))}</div>{isCurrentRun && <div className="node-edit-hint"><SlidersHorizontal size={12} />双击添加或修改训练参数</div>}</> },
    runtime: { title: `GPU ${viewTask.device}`, kicker: 'RUNTIME · 双击查看', icon: <Cpu size={17} />, accent: '#ffba6b', activity: nodeActivity, content: <>{row('进程 PID', viewTask.pid || '—')}{row('输出目录', viewTask.run_dir ? viewTask.run_dir.split('/').slice(-4).join('/') : '等待创建')}{row('训练环境', 'PyTorch · CUDA')}</> },
    weights: { title: '模型权重', kicker: 'CHECKPOINTS · 双击查看', icon: <Save size={17} />, accent: '#7da8ff', pulse: isCurrentRun && motionAllowed ? artifactPulse : undefined, content: <div className="weight-node-list">{[{ artifact: bestWeight, label: '最佳模型', name: 'best.pt', Icon: Trophy }, { artifact: lastWeight, label: '最新模型', name: 'last.pt', Icon: History }].map(({ artifact, label, name, Icon }) => <div className={`weight-mini-card ${artifact ? '' : 'pending'}`} key={name}><div className="weight-mini-icon"><Icon size={15} /></div><div><span>{label}</span><strong>{artifact?.name.split('/').pop() || name}</strong><small title={name === 'best.pt' ? bestWeightOrigin : ''}>{artifact ? `${formatBytes(artifact.size)}${name === 'best.pt' && bestWeightOrigin ? ` · ${bestWeightOrigin}` : ''}` : '等待训练生成'}</small></div>{artifact && <ArtifactDownload artifact={artifact} className="nodrag nowheel" title={`下载 ${name}`}><Download size={15} /></ArtifactDownload>}{artifact && name === 'best.pt' && <button className="weight-test-button nodrag nowheel" disabled={!!testDisabledReason} title={testDisabledReason || '测试模型'} onDoubleClick={(event) => event.stopPropagation()} onClick={() => { setTestingArtifact(artifact); setModelTestOpen(true) }}><ScanEye size={12} />测试模型</button>}</div>)}</div> },
    status: { title: trainingStatusLabel(viewTask), kicker: selectedRun ? `ATTEMPT ${String(selectedRun.attempt_number).padStart(3, '0')} · 双击查看` : 'TRAINING STATUS · 双击查看', icon: viewTask.status === 'running' ? <Zap size={17} /> : <Gauge size={17} />, activity: nodeActivity, pulse: isCurrentRun && motionAllowed ? epochPulse : undefined, content: <><div className="big-progress"><strong title="实际完成轮数占目标轮数的比例">{Math.round(trainingEpochProgress(viewTask))}%</strong><span>轮数进度 · Epoch {viewTask.completed_epochs} / {trainingEpochTarget(viewTask)}</span></div><div className="progress"><i style={{ width: `${trainingEpochProgress(viewTask)}%` }} /></div>{earlyStoppingExplanation(viewTask) && <p className="training-completion-note">{earlyStoppingExplanation(viewTask)}</p>}<div className="node-grid metrics-mini" title={`最新 Epoch ${viewTask.completed_epochs} 指标`}>{row('mAP50', metricValue(viewTask, ['metrics/mAP50(B)']))}{row('Precision', metricValue(viewTask, ['metrics/precision(B)']))}{row('Recall', metricValue(viewTask, ['metrics/recall(B)']))}{row('Box loss', typeof latest['train/box_loss'] === 'number' ? latest['train/box_loss'].toFixed(4) : '—')}</div>{viewTask.error_message && <p className="node-error">{viewTask.error_message}</p>}</> },
    metrics: { title: '验证指标趋势', kicker: 'METRICS · 双击查看', icon: <Activity size={17} />, accent: '#53d6e8', pulse: isCurrentRun && motionAllowed ? epochPulse : undefined, content: metrics.length ? <LineChartView option={chartOption(metrics, motionAllowed && isCurrentRun && viewTask.status === 'running')} style={{ height: 220, width: 380 }} /> : <div className="node-empty">训练产生首个 Epoch 后显示曲线</div> },
    logs: { title: '实时训练日志', kicker: 'LIVE OUTPUT · 双击放大', icon: <Terminal size={17} />, accent: '#95a3b3', pulse: isCurrentRun && motionAllowed ? logPulse : undefined, content: <><pre className="log-view">{logs.length ? logs.slice(-18).map((line, index, visibleLines) => <span key={index} className={index === visibleLines.length - 1 && isCurrentRun && logPulse ? `log-new-line log-flash-${logPulse}` : ''}>{line}{index < visibleLines.length - 1 ? '\n' : ''}</span>) : '等待训练日志…'}</pre><div className="node-edit-hint"><Maximize size={12} />双击打开大屏日志</div></> },
    trace: { title: '训练过程解析', kicker: 'TRAINING TRACE · 双击打开', icon: <Workflow size={17} />, accent: '#f08bba', content: <><div className="trace-node-summary"><div><strong>真实 Batch</strong><span>输入 · 损失 · 梯度</span></div><div><strong>本机检测</strong><span>按需采样</span></div></div><div className="node-edit-hint"><Maximize size={12} />双击打开逐步解析</div></> },
    artifacts: { title: '训练产物', kicker: 'ARTIFACTS · 双击查看', icon: <FileArchive size={17} />, accent: '#ffd56d', pulse: isCurrentRun && motionAllowed ? artifactPulse : undefined, content: regularArtifacts.length ? <div className="artifact-list">{regularArtifacts.slice(0, 12).map((item) => <div key={item.id} className={isCurrentRun && newArtifactIds.includes(item.id) ? `artifact-updated artifact-flash-${artifactPulse}` : ''}><FileText size={14} /><span>{item.name}{isCurrentRun && newArtifactIds.includes(item.id) && <em>已更新</em>}</span><small>{formatBytes(item.size)}</small></div>)}</div> : <div className="node-empty">暂无训练产物</div> },
    inference: { title: '模型测试', kicker: 'MODEL TEST · 双击打开', icon: <ScanEye size={17} />, accent: '#b3a0ff', content: <>{row('测试权重', bestWeight ? `best.pt${bestWeightOrigin ? ` · ${bestWeightOrigin}` : ''}` : '等待训练生成')}{row('测试方式', '图片 · 置信度筛选')}<div className="node-edit-hint"><Maximize size={12} />原图对比 · 双击放大 · 下载结果</div><button className="button ghost nodrag nowheel model-test-node-button" title={bestWeight && !testDisabledReason ? '打开模型测试' : '查看历史记录'} onDoubleClick={(event) => event.stopPropagation()} onClick={() => { setTestingArtifact(bestWeight || null); setModelTestOpen(true) }}><ScanEye size={14} />{bestWeight && !testDisabledReason ? '测试模型' : '历史记录'}</button></> },
  } : null, [viewTask, isCurrentRun, selectedRun, metrics, regularArtifacts, logs, latest, nodeActivity, epochPulse, logPulse, artifactPulse, motionAllowed, newArtifactIds, bestWeight, bestWeightOrigin, lastWeight, testDisabledReason, displayedDataPath, displayedModelPath, dataPathChanged, modelPathChanged, remoteResource, resourceConfig])

  useEffect(() => {
    if (!task || !nodeData) return
    setNodes((current) => Object.entries(nodeData).map(([id, data]) => {
      const existing = current.find((node) => node.id === id)
      const saved = task.canvas_layout.nodes?.find((node) => node.id === id)
      return { ...existing, id, type: 'panel', data, position: existing?.position || saved?.position || positions[id], className: `node-${id}` }
    }))
  }, [task, nodeData, setNodes])

  useEffect(() => {
    const flowStatus = isCurrentRun ? viewTask?.status : undefined
    const activeFlow = motionAllowed && ['queued', 'starting', 'running', 'stopping'].includes(flowStatus || '')
    setEdges((current) => current.map((edge) => ({
      ...edge,
      animated: activeFlow && edgeIsActive(edge.source, edge.target, flowStatus),
      className: `flow-edge flow-${flowStatus || 'idle'}`,
    })))
  }, [viewTask?.status, isCurrentRun, motionAllowed, setEdges])

  const onConnect = useCallback((connection: Connection) => setEdges((items) => addEdge({
    ...connection, markerEnd: { type: MarkerType.ArrowClosed },
    animated: motionAllowed && currentIsActive && edgeIsActive(connection.source, connection.target, viewTask?.status),
    className: `flow-edge flow-${isCurrentRun ? viewTask?.status || 'idle' : 'idle'}`,
  }, items)), [setEdges, motionAllowed, currentIsActive, isCurrentRun, viewTask?.status])

  async function action(kind: 'start' | 'stop' | 'resume' | 'retry') {
    if (mutationPending.current) return
    mutationPending.current = true
    cancelPendingRefresh()
    setWorking(true); setNotice('')
    let preferredRunId = selectedRunId
    try {
      const updated = await api[kind](taskId)
      cancelPendingRefresh()
      setTask(updated)
      preferredRunId = updated.current_run_id || selectedRunId
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '操作失败')
    } finally {
      try {
        await refresh({ allowDuringMutation: true, preferredRunId })
      } finally {
        mutationPending.current = false
        setWorking(false)
      }
    }
  }

  async function saveResource(mode: 'dataset' | 'model', path: string) {
    if (!task || mutationPending.current) throw new Error('其他任务操作正在进行，请稍后重试')
    mutationPending.current = true
    cancelPendingRefresh()
    setNotice('')
    try {
      const updated = await api.updateResources(task.id, mode === 'dataset' ? { data_path: path } : { model_path: path })
      cancelPendingRefresh()
      setTask(updated)
      setEditingResource(null)
      setNotice(`${mode === 'dataset' ? '数据集' : '模型'}路径已更新，重新训练将使用新配置`)
      window.setTimeout(() => setNotice(''), 2600)
    } finally {
      try {
        await refresh({ allowDuringMutation: true, preferredRunId: selectedRunId })
      } finally {
        mutationPending.current = false
      }
    }
  }

  async function forceInterrupt() {
    const runId = task?.current_run_id
    if (!runId || mutationPending.current) return
    const confirmed = window.confirm('仅在已登录远程服务器并确认该 Attempt 的 YOLO 进程已经结束时使用。错误确认可能让新任务与旧进程同时占用 GPU。\n\n确认将任务标记为已中断并解除节点占用吗？')
    if (!confirmed) return
    mutationPending.current = true
    cancelPendingRefresh()
    setWorking(true); setNotice('')
    try {
      const updated = await api.forceInterrupt(taskId, runId)
      cancelPendingRefresh()
      setTask(updated)
      setNotice('已解除远程节点占用，任务标记为中断')
    } catch (reason) {
      setNotice(reason instanceof Error ? reason.message : '解除节点占用失败')
    } finally {
      try {
        await refresh({ allowDuringMutation: true, preferredRunId: runId })
      } finally {
        mutationPending.current = false
        setWorking(false)
      }
    }
  }

  async function saveLayout() {
    if (!task || layoutSavePending.current) return
    layoutSavePending.current = true
    const controller = new AbortController()
    layoutSaveController.current = controller
    if (layoutFeedbackTimer.current !== null) window.clearTimeout(layoutFeedbackTimer.current)
    setLayoutFeedback({ kind: 'saving', message: '正在保存布局…' })
    try {
      await api.saveLayout(task.id, { nodes: nodes.map(({ id, position }) => ({ id, position })), viewport: getViewport() }, controller.signal)
      setLayoutFeedback({ kind: 'success', message: '画布布局已保存' })
      layoutFeedbackTimer.current = window.setTimeout(() => {
        setLayoutFeedback(null)
        layoutFeedbackTimer.current = null
      }, 1800)
    } catch (reason) {
      if (!controller.signal.aborted) setLayoutFeedback({ kind: 'error', message: reason instanceof Error ? reason.message : '布局保存失败，请重试' })
    } finally {
      if (layoutSaveController.current === controller) layoutSaveController.current = null
      layoutSavePending.current = false
    }
  }

  if (loading && !task) return <div className="canvas-loading"><LoaderCircle className="spin" />正在装载训练画布…</div>
  if (!task || !viewTask) return <div className="canvas-loading canvas-load-failure" role="alert">
    <AlertCircle size={28} />
    <strong>训练画布加载失败</strong>
    <p>{initialLoadError || '暂时无法获取任务数据，请重试。'}</p>
    <div className="canvas-load-failure-actions"><button className="button ghost" onClick={() => navigate('/')}><ArrowLeft size={15} />返回任务中心</button><button className="button primary" onClick={() => { cancelPendingRefresh(); setInitialLoadError(''); setLoading(true); void refresh() }}><RefreshCw size={15} />重新加载</button></div>
  </div>
  const savingLayout = layoutFeedback?.kind === 'saving'
  const canStart = task.status === 'draft'
  const canResume = ['stopped', 'interrupted'].includes(task.status)
  const canStop = ['queued', 'starting', 'running'].includes(task.status)
  const stopRequestedAt = new Date(events.find((event) => event.event_type === 'stop_requested')?.created_at
    || selectedRun?.updated_at || task.updated_at).getTime()
  const canForceInterrupt = Boolean(task.execution_node_id && task.status === 'stopping'
    && Number.isFinite(stopRequestedAt) && clock - stopRequestedAt >= 30_000)
  const canRetry = ['failed', 'stopped', 'interrupted', 'completed'].includes(task.status)
  const launchModeLabel = { fresh: '首次训练', resume: '继续训练', retry: '重新训练' }[selectedRun?.launch_mode || 'fresh']
  const heartbeatLabel = !isCurrentRun ? '历史快照' : syncDelayed ? '同步延迟' : outputWaiting ? '等待新输出' : currentIsActive ? '实时同步' : '数据已同步'
  const flowLabels: Partial<Record<TrainingTask['status'], string>> = {
    queued: '排队等待算力', starting: '正在启动训练', running: '训练流程运行中', stopping: '正在停止训练',
    stopped: '训练已停止', interrupted: '训练已中断', failed: '训练失败 · 请查看日志', completed: trainingStatusLabel(viewTask),
  }
  const flowLabel = !isCurrentRun ? '历史运行快照' : syncDelayed ? '同步延迟 · 动效暂停' : outputWaiting ? '暂未收到新输出' : flowLabels[viewTask.status] || '当前训练流程'

  return <div className={`canvas-page ${motionAllowed ? '' : 'motion-paused'}`}>
    <header className="canvas-toolbar">
      <button className="icon-button" onClick={() => navigate('/')}><ArrowLeft size={19} /></button>
      <div className="canvas-title"><span className={`status-dot ${viewTask.status}`} /><div><strong>{task.name}</strong><span>{task.task_type} · {viewTask.model_path.split('/').pop()}</span></div></div>
      {runs.length > 0 && <label className="attempt-picker"><History size={15} /><select value={selectedRunId} onChange={(event) => setSelectedRunId(event.target.value)}>{runs.map((run) => <option key={run.id} value={run.id}>Attempt {String(run.attempt_number).padStart(3, '0')} · {{ fresh: '首次训练', resume: '继续训练', retry: '重新训练' }[run.launch_mode]} · {trainingStatusLabel(run)}</option>)}</select></label>}
      <div className="canvas-state" title={earlyStoppingExplanation(viewTask) || undefined}><Clock3 size={14} />{selectedRun ? `Attempt ${String(selectedRun.attempt_number).padStart(3, '0')} · ${launchModeLabel} · ` : ''}{trainingStatusLabel(viewTask)} · Epoch {viewTask.completed_epochs}/{trainingEpochTarget(viewTask)}</div>
      <div className={`canvas-heartbeat ${currentIsActive ? 'active' : ''} ${syncDelayed || outputWaiting ? 'delayed' : ''}`} title={syncError || `接口响应：${syncAge} 秒前；本次查看中最近观察到输出变化：${outputAge} 秒前。无新输出不代表训练失败。`}><Radio size={13} /><i /><span>{heartbeatLabel}</span><strong>{syncAge}秒前</strong></div>
      <div className="canvas-actions">
        {notice && <span className="toast"><Check size={14} />{notice}</span>}
        {layoutFeedback && <span className={`toast ${layoutFeedback.kind}`} role={layoutFeedback.kind === 'error' ? 'alert' : 'status'} title={layoutFeedback.message}>{layoutFeedback.kind === 'saving' ? <LoaderCircle className="spin" size={14} /> : layoutFeedback.kind === 'error' ? <AlertCircle size={14} /> : <Check size={14} />}{layoutFeedback.message}</span>}
        {syncError && <span className="canvas-sync-error" role="alert" title={syncError}>同步失败：{syncError}</span>}
        <button className="button ghost compact effects-toggle" aria-pressed={effectsEnabled} title={reduceMotion ? '系统已启用减少动态效果，动画保持关闭' : '仅控制画布动画，不影响训练'} onClick={() => setEffectsEnabled((enabled) => !enabled)}><Activity size={15} />动效{effectsEnabled ? '开' : '关'}</button>
        <button className="button ghost compact" onClick={() => fitView({ padding: 0.16, duration: reduceMotion ? 0 : 500 })}><Maximize size={15} />适应画布</button>
        <button className="button ghost compact" disabled={savingLayout} aria-busy={savingLayout} onClick={() => void saveLayout()}>{savingLayout ? <LoaderCircle className="spin" size={15} /> : <Save size={15} />}{savingLayout ? '保存中' : '保存布局'}</button>
        {isCurrentRun && canStart && <button className="button primary compact" disabled={working} onClick={() => action('start')}><Play size={15} />启动训练</button>}
        {isCurrentRun && canResume && <button className="button primary compact" disabled={working} onClick={() => action('resume')}><RefreshCw size={15} />继续训练</button>}
        {isCurrentRun && canRetry && <button className="button ghost compact" disabled={working} onClick={() => action('retry')}><RotateCcw size={15} />重新训练</button>}
        {isCurrentRun && canStop && <button className="button danger compact" disabled={working} onClick={() => action('stop')}><Square size={14} />停止</button>}
        {isCurrentRun && canForceInterrupt && <button className="button danger compact" disabled={working} title="仅在远程 YOLO 进程已确认结束时解除占用" onClick={() => void forceInterrupt()}><AlertCircle size={14} />解除占用</button>}
      </div>
    </header>
    <div className={`canvas-wrap flow-${isCurrentRun ? viewTask.status : 'history'}`}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeDoubleClick={(_, node) => { if (isCurrentRun && node.id === 'config') setEditingParameters(true); if (isCurrentRun && (node.id === 'dataset' || node.id === 'model')) setEditingResource(node.id); if (node.id === 'logs') setViewingLogs(true); if (node.id === 'trace') setViewingTrace(true); if (node.id === 'inference') { setTestingArtifact(bestWeight || null); setModelTestOpen(true); } if (['runtime', 'status', 'metrics', 'artifacts', 'weights'].includes(node.id)) setDetailKind(node.id as CanvasDetailKind) }} fitView fitViewOptions={{ padding: 0.15 }} minZoom={0.25} maxZoom={1.6}>
        <Background color={resolvedTheme === 'light' ? '#c8d8df' : '#24313d'} gap={24} size={1} />
        <MiniMap pannable zoomable nodeColor={(node) => node.id === 'status' ? (resolvedTheme === 'light' ? '#159a78' : '#70f0bd') : (resolvedTheme === 'light' ? '#b7cad6' : '#344657')} maskColor={resolvedTheme === 'light' ? 'rgba(225,235,242,.72)' : 'rgba(5,9,13,.72)'} />
        <Controls showInteractive={false} />
      </ReactFlow>
      <div className="canvas-legend"><span><i className={currentIsActive && !syncDelayed && !outputWaiting ? 'green live' : 'green'} />{flowLabel}</span><span>{events.length} 条任务事件</span><span>拖拽节点 · 滚轮缩放 · 空白处平移</span></div>
    </div>
    <EditParametersModal task={task} open={editingParameters} onClose={() => setEditingParameters(false)} onSaved={(updated) => { setTask(updated); setEditingParameters(false); setNotice('训练参数已保存'); window.setTimeout(() => setNotice(''), 1800) }} />
    <EditResourceModal task={task} mode={editingResource} onClose={() => setEditingResource(null)} onSave={saveResource} />
    <LogViewerModal task={viewTask} lines={logs} open={viewingLogs} onClose={() => setViewingLogs(false)} />
    <CanvasDetailModal kind={detailKind} task={viewTask} run={selectedRun} metrics={metrics} artifacts={artifacts} effectiveBestWeight={effectiveBestWeight} events={events} onClose={() => setDetailKind(null)} onTest={(artifact) => { setDetailKind(null); setTestingArtifact(artifact); setModelTestOpen(true) }} testDisabledReason={testDisabledReason} />
    <TrainingTraceModal task={viewTask} run={selectedRun} open={viewingTrace} onClose={() => setViewingTrace(false)} />
    {modelTestOpen && <ModelTestModal key={testingArtifact?.id || 'history'} artifact={testingArtifact} taskId={task.id} taskName={task.name} taskType={viewTask.task_type} runId={selectedRun?.id || null} attemptNumber={modelTestAttempt} bestMetric={modelTestBestMetric} newTestDisabledReason={testDisabledReason} onClose={() => setModelTestOpen(false)} />}
  </div>
}

export function TaskCanvas() {
  const { taskId } = useParams()
  return <ReactFlowProvider key={taskId}><TaskCanvasInner /></ReactFlowProvider>
}
