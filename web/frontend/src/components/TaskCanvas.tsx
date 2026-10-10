import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Background, Controls, MarkerType, MiniMap, ReactFlow, ReactFlowProvider, addEdge, useEdgesState, useNodesState, useReactFlow, type Connection, type Edge, type Node } from '@xyflow/react'
import { Activity, ArrowLeft, Box, Braces, Check, Clock3, Cpu, Database, FileArchive, FileText, Gauge, History, LoaderCircle, Maximize, Play, Radio, RefreshCw, RotateCcw, Save, SlidersHorizontal, Square, Terminal, Zap } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'
import { api, formatBytes, metricValue, statusLabel } from '../api'
import type { Artifact, MetricPoint, TaskEvent, TrainingRun, TrainingTask } from '../types'
import { CanvasNode, type PanelNodeData } from './CanvasNode'
import { CanvasDetailModal, type CanvasDetailKind } from './CanvasDetailModal'
import { EditParametersModal } from './EditParametersModal'
import { EditResourceModal } from './EditResourceModal'
import { LogViewerModal } from './LogViewerModal'

const nodeTypes = { panel: CanvasNode }
const initialEdges: Edge[] = [
  ['dataset', 'config'], ['model', 'config'], ['config', 'status'], ['runtime', 'status'],
  ['status', 'metrics'], ['status', 'logs'], ['metrics', 'artifacts'],
].map(([source, target], index) => ({
  id: `edge-${index}`, source, target, markerEnd: { type: MarkerType.ArrowClosed },
}))
const positions: Record<string, { x: number; y: number }> = {
  dataset: { x: 20, y: 80 }, model: { x: 20, y: 420 }, config: { x: 400, y: 60 }, runtime: { x: 400, y: 430 },
  status: { x: 800, y: 170 }, metrics: { x: 1180, y: 20 }, logs: { x: 1180, y: 430 }, artifacts: { x: 1580, y: 180 },
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
  const [task, setTask] = useState<TrainingTask | null>(null)
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [metrics, setMetrics] = useState<MetricPoint[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [events, setEvents] = useState<TaskEvent[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [working, setWorking] = useState(false)
  const [notice, setNotice] = useState('')
  const [syncError, setSyncError] = useState('')
  const [editingParameters, setEditingParameters] = useState(false)
  const [editingResource, setEditingResource] = useState<'dataset' | 'model' | null>(null)
  const [viewingLogs, setViewingLogs] = useState(false)
  const [detailKind, setDetailKind] = useState<CanvasDetailKind | null>(null)
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
  const pendingRequest = useRef<number | null>(null)
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(initialEdges)
  const { fitView, getViewport } = useReactFlow()

  const refresh = useCallback(async () => {
    if (pendingRequest.current !== null) return
    const version = ++requestVersion.current
    pendingRequest.current = version
    try {
      // Validate the task first so a stale canvas URL does not fire several
      // dependent 404 requests on every polling cycle.
      const [taskData, runData] = await Promise.all([api.task(taskId), api.runs(taskId)])
      if (version !== requestVersion.current) return
      const selectedId = runData.some((run) => run.id === selectedRunId)
        ? selectedRunId
        : (taskData.current_run_id || runData[0]?.id || '')
      const selectedRun = runData.find((run) => run.id === selectedId)
      setTask(taskData); setRuns(runData)
      if (selectedId !== selectedRunId) setSelectedRunId(selectedId)
      if (!selectedRun) {
        setMetrics([]); setArtifacts([]); setEvents([]); setLogs([])
        setSyncError('')
        setLastSyncedAt(Date.now())
        return
      }
      const [metricData, artifactData, eventData, logData] = await Promise.all([
        api.runMetrics(taskId, selectedRun.id), api.runArtifacts(taskId, selectedRun.id),
        api.runEvents(taskId, selectedRun.id), api.runLogs(taskId, selectedRun.id),
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
      setMetrics(metricData); setArtifacts(artifactData); setEvents(eventData); setLogs(logData.lines)
      setSyncError('')
      setLastSyncedAt(Date.now())
    } catch (reason) {
      if (version !== requestVersion.current) return
      const status = reason instanceof Error ? (reason as Error & { status?: number }).status : undefined
      if (status === 404) {
        navigate('/', { replace: true })
        return
      }
      setSyncError(reason instanceof Error ? reason.message : '加载失败')
    } finally {
      if (pendingRequest.current === version) pendingRequest.current = null
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
      requestVersion.current += 1
      pendingRequest.current = null
    }
  }, [pageVisible, refresh])

  useEffect(() => {
    previousUpdate.current = null
    setEpochPulse(undefined); setLogPulse(undefined); setArtifactPulse(undefined)
    setNewArtifactIds([])
    setMetrics([]); setArtifacts([]); setEvents([]); setLogs([])
  }, [taskId, selectedRunId])

  useEffect(() => {
    if (!newArtifactIds.length) return
    const timer = window.setTimeout(() => setNewArtifactIds([]), 4000)
    return () => window.clearTimeout(timer)
  }, [newArtifactIds])

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
      latest_metrics: selectedRun.latest_metrics,
      queued_at: selectedRun.queued_at,
      started_at: selectedRun.started_at,
      finished_at: selectedRun.finished_at,
    } satisfies TrainingTask
  }, [selectedRun, task])
  const latest = viewTask?.latest_metrics || {}
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
  const nodeData = useMemo<Record<string, PanelNodeData> | null>(() => viewTask ? {
    dataset: { title: viewTask.data_path.split('/').pop() || '数据集', kicker: `DATASET${isCurrentRun ? ' · 双击编辑' : ' · 历史快照'}`, icon: <Database size={17} />, content: <><p className="mono path-value">{viewTask.data_path}</p>{row('配置状态', viewTask.config.path_issues && (viewTask.config.path_issues as { data_missing?: boolean }).data_missing ? <span className="bad">路径失效</span> : <span className="good">可用</span>)}{isCurrentRun && <div className="node-edit-hint"><SlidersHorizontal size={12} />双击更换训练数据集</div>}</> },
    model: { title: viewTask.model_path.split('/').pop() || '基础模型', kicker: `MODEL${isCurrentRun ? ' · 双击编辑' : ' · 历史快照'}`, icon: <Box size={17} />, accent: '#6da8ff', content: <><p className="mono path-value">{viewTask.model_path}</p>{row('任务类型', viewTask.task_type)}{row('预训练', viewTask.config.pretrained === false ? '关闭' : '启用')}{isCurrentRun && <div className="node-edit-hint"><SlidersHorizontal size={12} />双击更换基础模型</div>}</> },
    config: { title: '超参数配置', kicker: `CONFIGURATION${isCurrentRun ? ' · 双击编辑' : ' · 历史快照'}`, icon: <Braces size={17} />, accent: '#d99cff', content: <><div className="node-grid">{row('Epochs', viewTask.epochs)}{row('Batch', viewTask.batch)}{row('Image', viewTask.imgsz)}{row('Optimizer', viewTask.optimizer)}{row('Workers', viewTask.workers)}{row('Patience', String(viewTask.config.patience ?? '—'))}</div>{isCurrentRun && <div className="node-edit-hint"><SlidersHorizontal size={12} />双击添加或修改训练参数</div>}</> },
    runtime: { title: `GPU ${viewTask.device}`, kicker: 'RUNTIME · 双击查看', icon: <Cpu size={17} />, accent: '#ffba6b', activity: nodeActivity, content: <>{row('进程 PID', viewTask.pid || '—')}{row('输出目录', viewTask.run_dir ? viewTask.run_dir.split('/').slice(-4).join('/') : '等待创建')}{row('训练环境', 'PyTorch · CUDA')}</> },
    status: { title: statusLabel(viewTask.status), kicker: selectedRun ? `ATTEMPT ${String(selectedRun.attempt_number).padStart(3, '0')} · 双击查看` : 'TRAINING STATUS · 双击查看', icon: viewTask.status === 'running' ? <Zap size={17} /> : <Gauge size={17} />, activity: nodeActivity, pulse: isCurrentRun && motionAllowed ? epochPulse : undefined, content: <><div className="big-progress"><strong>{Math.round(viewTask.progress)}%</strong><span>Epoch {viewTask.completed_epochs} / {viewTask.epochs}</span></div><div className="progress"><i style={{ width: `${viewTask.progress}%` }} /></div><div className="node-grid metrics-mini">{row('mAP50', metricValue(viewTask, ['metrics/mAP50(B)']))}{row('Precision', metricValue(viewTask, ['metrics/precision(B)']))}{row('Recall', metricValue(viewTask, ['metrics/recall(B)']))}{row('Box loss', typeof latest['train/box_loss'] === 'number' ? latest['train/box_loss'].toFixed(4) : '—')}</div>{viewTask.error_message && <p className="node-error">{viewTask.error_message}</p>}</> },
    metrics: { title: '验证指标趋势', kicker: 'METRICS · 双击查看', icon: <Activity size={17} />, accent: '#53d6e8', pulse: isCurrentRun && motionAllowed ? epochPulse : undefined, content: metrics.length ? <ReactECharts option={chartOption(metrics, motionAllowed && isCurrentRun && viewTask.status === 'running')} style={{ height: 220, width: 380 }} /> : <div className="node-empty">训练产生首个 Epoch 后显示曲线</div> },
    logs: { title: '实时训练日志', kicker: 'LIVE OUTPUT · 双击放大', icon: <Terminal size={17} />, accent: '#95a3b3', pulse: isCurrentRun && motionAllowed ? logPulse : undefined, content: <><pre className="log-view">{logs.length ? logs.slice(-18).map((line, index, visibleLines) => <span key={index} className={index === visibleLines.length - 1 && isCurrentRun && logPulse ? `log-new-line log-flash-${logPulse}` : ''}>{line}{index < visibleLines.length - 1 ? '\n' : ''}</span>) : '等待训练日志…'}</pre><div className="node-edit-hint"><Maximize size={12} />双击打开大屏日志</div></> },
    artifacts: { title: '训练产物', kicker: 'ARTIFACTS · 双击查看', icon: <FileArchive size={17} />, accent: '#ffd56d', pulse: isCurrentRun && motionAllowed ? artifactPulse : undefined, content: artifacts.length ? <div className="artifact-list">{artifacts.slice(0, 12).map((item) => <div key={item.id} className={isCurrentRun && newArtifactIds.includes(item.id) ? `artifact-updated artifact-flash-${artifactPulse}` : ''}><FileText size={14} /><span>{item.name}{isCurrentRun && newArtifactIds.includes(item.id) && <em>已更新</em>}</span><small>{formatBytes(item.size)}</small></div>)}</div> : <div className="node-empty">暂无训练产物</div> },
  } : null, [viewTask, isCurrentRun, selectedRun, metrics, artifacts, logs, latest, nodeActivity, epochPulse, logPulse, artifactPulse, motionAllowed, newArtifactIds])

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
    setWorking(true); setNotice('')
    try {
      const updated = await api[kind](taskId)
      setTask(updated)
      if (updated.current_run_id && updated.current_run_id !== selectedRunId) setSelectedRunId(updated.current_run_id)
      else await refresh()
    } catch (reason) { setNotice(reason instanceof Error ? reason.message : '操作失败') } finally { setWorking(false) }
  }

  async function saveLayout() {
    if (!task) return
    await api.saveLayout(task.id, { nodes: nodes.map(({ id, position }) => ({ id, position })), viewport: getViewport() })
    setNotice('画布布局已保存'); window.setTimeout(() => setNotice(''), 1800)
  }

  if (loading || !task || !viewTask) return <div className="canvas-loading"><LoaderCircle className="spin" />正在装载训练画布…</div>
  const canStart = task.status === 'draft'
  const canResume = ['stopped', 'interrupted'].includes(task.status)
  const canStop = ['queued', 'starting', 'running'].includes(task.status)
  const canRetry = ['failed', 'stopped', 'interrupted', 'completed'].includes(task.status)
  const launchModeLabel = { fresh: '首次训练', resume: '继续训练', retry: '重新训练' }[selectedRun?.launch_mode || 'fresh']
  const heartbeatLabel = !isCurrentRun ? '历史快照' : syncDelayed ? '同步延迟' : outputWaiting ? '等待新输出' : currentIsActive ? '实时同步' : '数据已同步'
  const flowLabels: Partial<Record<TrainingTask['status'], string>> = {
    queued: '排队等待算力', starting: '正在启动训练', running: '训练流程运行中', stopping: '正在停止训练',
    stopped: '训练已停止', interrupted: '训练已中断', failed: '训练失败 · 请查看日志', completed: '训练已完成',
  }
  const flowLabel = !isCurrentRun ? '历史运行快照' : syncDelayed ? '同步延迟 · 动效暂停' : outputWaiting ? '暂未收到新输出' : flowLabels[viewTask.status] || '当前训练流程'

  return <div className={`canvas-page ${motionAllowed ? '' : 'motion-paused'}`}>
    <header className="canvas-toolbar">
      <button className="icon-button" onClick={() => navigate('/')}><ArrowLeft size={19} /></button>
      <div className="canvas-title"><span className={`status-dot ${viewTask.status}`} /><div><strong>{task.name}</strong><span>{task.task_type} · {viewTask.model_path.split('/').pop()}</span></div></div>
      {runs.length > 0 && <label className="attempt-picker"><History size={15} /><select value={selectedRunId} onChange={(event) => setSelectedRunId(event.target.value)}>{runs.map((run) => <option key={run.id} value={run.id}>Attempt {String(run.attempt_number).padStart(3, '0')} · {{ fresh: '首次训练', resume: '继续训练', retry: '重新训练' }[run.launch_mode]} · {statusLabel(run.status)}</option>)}</select></label>}
      <div className="canvas-state"><Clock3 size={14} />{selectedRun ? `Attempt ${String(selectedRun.attempt_number).padStart(3, '0')} · ${launchModeLabel} · ` : ''}{statusLabel(viewTask.status)} · Epoch {viewTask.completed_epochs}/{viewTask.epochs}</div>
      <div className={`canvas-heartbeat ${currentIsActive ? 'active' : ''} ${syncDelayed || outputWaiting ? 'delayed' : ''}`} title={syncError || `接口响应：${syncAge} 秒前；本次查看中最近观察到输出变化：${outputAge} 秒前。无新输出不代表训练失败。`}><Radio size={13} /><i /><span>{heartbeatLabel}</span><strong>{syncAge}秒前</strong></div>
      <div className="canvas-actions">
        {notice && <span className="toast"><Check size={14} />{notice}</span>}
        {syncError && <span className="canvas-sync-error" role="status">{syncError}</span>}
        <button className="button ghost compact effects-toggle" aria-pressed={effectsEnabled} title={reduceMotion ? '系统已启用减少动态效果，动画保持关闭' : '仅控制画布动画，不影响训练'} onClick={() => setEffectsEnabled((enabled) => !enabled)}><Activity size={15} />动效{effectsEnabled ? '开' : '关'}</button>
        <button className="button ghost compact" onClick={() => fitView({ padding: 0.16, duration: reduceMotion ? 0 : 500 })}><Maximize size={15} />适应画布</button>
        <button className="button ghost compact" onClick={saveLayout}><Save size={15} />保存布局</button>
        {isCurrentRun && canStart && <button className="button primary compact" disabled={working} onClick={() => action('start')}><Play size={15} />启动训练</button>}
        {isCurrentRun && canResume && <button className="button primary compact" disabled={working} onClick={() => action('resume')}><RefreshCw size={15} />继续训练</button>}
        {isCurrentRun && canRetry && <button className="button ghost compact" disabled={working} onClick={() => action('retry')}><RotateCcw size={15} />重新训练</button>}
        {isCurrentRun && canStop && <button className="button danger compact" disabled={working} onClick={() => action('stop')}><Square size={14} />停止</button>}
      </div>
    </header>
    <div className={`canvas-wrap flow-${isCurrentRun ? viewTask.status : 'history'}`}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeDoubleClick={(_, node) => { if (isCurrentRun && node.id === 'config') setEditingParameters(true); if (isCurrentRun && (node.id === 'dataset' || node.id === 'model')) setEditingResource(node.id); if (node.id === 'logs') setViewingLogs(true); if (['runtime', 'status', 'metrics', 'artifacts'].includes(node.id)) setDetailKind(node.id as CanvasDetailKind) }} fitView fitViewOptions={{ padding: 0.15 }} minZoom={0.25} maxZoom={1.6}>
        <Background color="#24313d" gap={24} size={1} />
        <MiniMap pannable zoomable nodeColor={(node) => node.id === 'status' ? '#70f0bd' : '#344657'} maskColor="rgba(5,9,13,.72)" />
        <Controls showInteractive={false} />
      </ReactFlow>
      <div className="canvas-legend"><span><i className={currentIsActive && !syncDelayed && !outputWaiting ? 'green live' : 'green'} />{flowLabel}</span><span>{events.length} 条任务事件</span><span>拖拽节点 · 滚轮缩放 · 空白处平移</span></div>
    </div>
    <EditParametersModal task={task} open={editingParameters} onClose={() => setEditingParameters(false)} onSaved={(updated) => { setTask(updated); setEditingParameters(false); setNotice('训练参数已保存'); window.setTimeout(() => setNotice(''), 1800) }} />
    <EditResourceModal task={task} mode={editingResource} onClose={() => setEditingResource(null)} onSaved={(updated) => { setTask(updated); setEditingResource(null); setNotice('训练资源已更新'); window.setTimeout(() => setNotice(''), 1800) }} />
    <LogViewerModal task={viewTask} lines={logs} open={viewingLogs} onClose={() => setViewingLogs(false)} />
    <CanvasDetailModal kind={detailKind} task={viewTask} run={selectedRun} metrics={metrics} artifacts={artifacts} events={events} onClose={() => setDetailKind(null)} />
  </div>
}

export function TaskCanvas() {
  const { taskId } = useParams()
  return <ReactFlowProvider key={taskId}><TaskCanvasInner /></ReactFlowProvider>
}
