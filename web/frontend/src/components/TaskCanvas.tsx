import { useCallback, useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Background, Controls, MarkerType, MiniMap, ReactFlow, ReactFlowProvider, addEdge, useEdgesState, useNodesState, useReactFlow, type Connection, type Edge, type Node } from '@xyflow/react'
import { Activity, ArrowLeft, Box, Braces, Check, Clock3, Cpu, Database, FileArchive, FileText, Gauge, LoaderCircle, Maximize, Play, RefreshCw, Save, SlidersHorizontal, Square, Terminal, Zap } from 'lucide-react'
import { useNavigate, useParams } from 'react-router-dom'
import { api, formatBytes, metricValue, statusLabel } from '../api'
import type { Artifact, MetricPoint, TaskEvent, TrainingTask } from '../types'
import { CanvasNode, type PanelNodeData } from './CanvasNode'
import { EditParametersModal } from './EditParametersModal'
import { EditResourceModal } from './EditResourceModal'
import { LogViewerModal } from './LogViewerModal'

const nodeTypes = { panel: CanvasNode }
const positions: Record<string, { x: number; y: number }> = {
  dataset: { x: 20, y: 80 }, model: { x: 20, y: 420 }, config: { x: 400, y: 60 }, runtime: { x: 400, y: 430 },
  status: { x: 800, y: 170 }, metrics: { x: 1180, y: 20 }, logs: { x: 1180, y: 430 }, artifacts: { x: 1580, y: 180 },
}

function row(label: string, value: React.ReactNode) { return <div className="node-row"><span>{label}</span><strong>{value}</strong></div> }

function chartOption(metrics: MetricPoint[]) {
  const values = (key: string) => metrics.map((point) => typeof point.values[key] === 'number' ? point.values[key] : null)
  return {
    animation: false,
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
  const [metrics, setMetrics] = useState<MetricPoint[]>([])
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [events, setEvents] = useState<TaskEvent[]>([])
  const [logs, setLogs] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [working, setWorking] = useState(false)
  const [notice, setNotice] = useState('')
  const [editingParameters, setEditingParameters] = useState(false)
  const [editingResource, setEditingResource] = useState<'dataset' | 'model' | null>(null)
  const [viewingLogs, setViewingLogs] = useState(false)
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])
  const { fitView, getViewport } = useReactFlow()

  const refresh = useCallback(async () => {
    try {
      // Validate the task first so a stale canvas URL does not fire several
      // dependent 404 requests on every polling cycle.
      const taskData = await api.task(taskId)
      const [metricData, artifactData, eventData, logData] = await Promise.all([
        api.metrics(taskId), api.artifacts(taskId), api.events(taskId), api.logs(taskId),
      ])
      setTask(taskData); setMetrics(metricData); setArtifacts(artifactData); setEvents(eventData); setLogs(logData.lines)
    } catch (reason) {
      const status = reason instanceof Error ? (reason as Error & { status?: number }).status : undefined
      if (status === 404) {
        navigate('/', { replace: true })
        return
      }
      setNotice(reason instanceof Error ? reason.message : '加载失败')
    } finally { setLoading(false) }
  }, [navigate, taskId])

  useEffect(() => { refresh(); const timer = window.setInterval(refresh, 3000); return () => window.clearInterval(timer) }, [refresh])

  const latest = task?.latest_metrics || {}
  const nodeData = useMemo<Record<string, PanelNodeData> | null>(() => task ? {
    dataset: { title: task.data_path.split('/').pop() || '数据集', kicker: 'DATASET · 双击编辑', icon: <Database size={17} />, content: <><p className="mono path-value">{task.data_path}</p>{row('配置状态', task.config.path_issues && (task.config.path_issues as { data_missing?: boolean }).data_missing ? <span className="bad">路径失效</span> : <span className="good">可用</span>)}<div className="node-edit-hint"><SlidersHorizontal size={12} />双击更换训练数据集</div></> },
    model: { title: task.model_path.split('/').pop() || '基础模型', kicker: 'MODEL · 双击编辑', icon: <Box size={17} />, accent: '#6da8ff', content: <><p className="mono path-value">{task.model_path}</p>{row('任务类型', task.task_type)}{row('预训练', task.config.pretrained === false ? '关闭' : '启用')}<div className="node-edit-hint"><SlidersHorizontal size={12} />双击更换基础模型</div></> },
    config: { title: '超参数配置', kicker: 'CONFIGURATION · 双击编辑', icon: <Braces size={17} />, accent: '#d99cff', content: <><div className="node-grid">{row('Epochs', task.epochs)}{row('Batch', task.batch)}{row('Image', task.imgsz)}{row('Optimizer', task.optimizer)}{row('Workers', task.workers)}{row('Patience', String(task.config.patience ?? '—'))}</div><div className="node-edit-hint"><SlidersHorizontal size={12} />双击添加或修改训练参数</div></> },
    runtime: { title: `GPU ${task.device}`, kicker: 'RUNTIME', icon: <Cpu size={17} />, accent: '#ffba6b', content: <>{row('进程 PID', task.pid || '—')}{row('输出目录', task.run_dir ? task.run_dir.split('/').slice(-3).join('/') : '等待创建')}{row('训练环境', 'PyTorch · CUDA')}</> },
    status: { title: statusLabel(task.status), kicker: 'TRAINING STATUS', icon: task.status === 'running' ? <Zap size={17} /> : <Gauge size={17} />, content: <><div className="big-progress"><strong>{Math.round(task.progress)}%</strong><span>Epoch {task.completed_epochs} / {task.epochs}</span></div><div className="progress"><i style={{ width: `${task.progress}%` }} /></div><div className="node-grid metrics-mini">{row('mAP50', metricValue(task, ['metrics/mAP50(B)']))}{row('Precision', metricValue(task, ['metrics/precision(B)']))}{row('Recall', metricValue(task, ['metrics/recall(B)']))}{row('Box loss', typeof latest['train/box_loss'] === 'number' ? latest['train/box_loss'].toFixed(4) : '—')}</div>{task.error_message && <p className="node-error">{task.error_message}</p>}</> },
    metrics: { title: '验证指标趋势', kicker: 'METRICS', icon: <Activity size={17} />, accent: '#53d6e8', content: metrics.length ? <ReactECharts option={chartOption(metrics)} style={{ height: 220, width: 380 }} /> : <div className="node-empty">训练产生首个 Epoch 后显示曲线</div> },
    logs: { title: '实时训练日志', kicker: 'LIVE OUTPUT · 双击放大', icon: <Terminal size={17} />, accent: '#95a3b3', content: <><pre className="log-view">{logs.slice(-18).join('\n') || '等待训练日志…'}</pre><div className="node-edit-hint"><Maximize size={12} />双击打开大屏日志</div></> },
    artifacts: { title: '训练产物', kicker: 'ARTIFACTS', icon: <FileArchive size={17} />, accent: '#ffd56d', content: artifacts.length ? <div className="artifact-list">{artifacts.slice(0, 12).map((item) => <a key={item.id} href={`/api/artifacts/${item.id}/download`}><FileText size={14} /><span>{item.name}</span><small>{formatBytes(item.size)}</small></a>)}</div> : <div className="node-empty">暂无训练产物</div> },
  } : null, [task, metrics, artifacts, logs, latest])

  useEffect(() => {
    if (!task || !nodeData) return
    setNodes((current) => Object.entries(nodeData).map(([id, data]) => {
      const existing = current.find((node) => node.id === id)
      const saved = task.canvas_layout.nodes?.find((node) => node.id === id)
      return { id, type: 'panel', data, position: existing?.position || saved?.position || positions[id], className: `node-${id}` }
    }))
    setEdges((current) => current.length ? current : [
      ['dataset', 'config'], ['model', 'config'], ['config', 'status'], ['runtime', 'status'], ['status', 'metrics'], ['status', 'logs'], ['metrics', 'artifacts'],
    ].map(([source, target], index) => ({ id: `edge-${index}`, source, target, markerEnd: { type: MarkerType.ArrowClosed }, animated: target === 'status' && task.status === 'running' })))
  }, [task, nodeData, setNodes, setEdges])

  const onConnect = useCallback((connection: Connection) => setEdges((items) => addEdge(connection, items)), [setEdges])

  async function action(kind: 'start' | 'stop' | 'resume') {
    setWorking(true); setNotice('')
    try { await api[kind](taskId); await refresh() } catch (reason) { setNotice(reason instanceof Error ? reason.message : '操作失败') } finally { setWorking(false) }
  }

  async function saveLayout() {
    if (!task) return
    await api.saveLayout(task.id, { nodes: nodes.map(({ id, position }) => ({ id, position })), viewport: getViewport() })
    setNotice('画布布局已保存'); window.setTimeout(() => setNotice(''), 1800)
  }

  if (loading || !task) return <div className="canvas-loading"><LoaderCircle className="spin" />正在装载训练画布…</div>
  const canStart = ['draft', 'failed'].includes(task.status)
  const canResume = ['stopped', 'interrupted'].includes(task.status)
  const canStop = ['queued', 'starting', 'running'].includes(task.status)

  return <div className="canvas-page">
    <header className="canvas-toolbar">
      <button className="icon-button" onClick={() => navigate('/')}><ArrowLeft size={19} /></button>
      <div className="canvas-title"><span className={`status-dot ${task.status}`} /><div><strong>{task.name}</strong><span>{task.task_type} · {task.model_path.split('/').pop()}</span></div></div>
      <div className="canvas-state"><Clock3 size={14} />{statusLabel(task.status)} · Epoch {task.completed_epochs}/{task.epochs}</div>
      <div className="canvas-actions">
        {notice && <span className="toast"><Check size={14} />{notice}</span>}
        <button className="button ghost compact" onClick={() => fitView({ padding: 0.16, duration: 500 })}><Maximize size={15} />适应画布</button>
        <button className="button ghost compact" onClick={saveLayout}><Save size={15} />保存布局</button>
        {canStart && <button className="button primary compact" disabled={working} onClick={() => action('start')}><Play size={15} />启动训练</button>}
        {canResume && <button className="button primary compact" disabled={working} onClick={() => action('resume')}><RefreshCw size={15} />继续训练</button>}
        {canStop && <button className="button danger compact" disabled={working} onClick={() => action('stop')}><Square size={14} />停止</button>}
      </div>
    </header>
    <div className="canvas-wrap">
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onConnect={onConnect} onNodeDoubleClick={(_, node) => { if (node.id === 'config') setEditingParameters(true); if (node.id === 'dataset' || node.id === 'model') setEditingResource(node.id); if (node.id === 'logs') setViewingLogs(true) }} fitView fitViewOptions={{ padding: 0.15 }} minZoom={0.25} maxZoom={1.6}>
        <Background color="#24313d" gap={24} size={1} />
        <MiniMap pannable zoomable nodeColor={(node) => node.id === 'status' ? '#70f0bd' : '#344657'} maskColor="rgba(5,9,13,.72)" />
        <Controls showInteractive={false} />
      </ReactFlow>
      <div className="canvas-legend"><span><i className="green" />数据流</span><span>{events.length} 条任务事件</span><span>拖拽节点 · 滚轮缩放 · 空白处平移</span></div>
    </div>
    <EditParametersModal task={task} open={editingParameters} onClose={() => setEditingParameters(false)} onSaved={(updated) => { setTask(updated); setEditingParameters(false); setNotice('训练参数已保存'); window.setTimeout(() => setNotice(''), 1800) }} />
    <EditResourceModal task={task} mode={editingResource} onClose={() => setEditingResource(null)} onSaved={(updated) => { setTask(updated); setEditingResource(null); setNotice('训练资源已更新'); window.setTimeout(() => setNotice(''), 1800) }} />
    <LogViewerModal task={task} lines={logs} open={viewingLogs} onClose={() => setViewingLogs(false)} />
  </div>
}

export function TaskCanvas() { return <ReactFlowProvider><TaskCanvasInner /></ReactFlowProvider> }
