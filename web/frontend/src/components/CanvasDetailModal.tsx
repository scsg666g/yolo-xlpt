import { useEffect, useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Activity, Cpu, Download, FileArchive, FileText, Gauge, HardDrive, History, Image, Save, Terminal, Trophy, X } from 'lucide-react'
import { formatBytes, statusLabel } from '../api'
import type { Artifact, MetricPoint, TaskEvent, TrainingRun, TrainingTask } from '../types'

export type CanvasDetailKind = 'runtime' | 'status' | 'metrics' | 'artifacts' | 'weights'

type Props = {
  kind: CanvasDetailKind | null
  task: TrainingTask
  run: TrainingRun | null
  metrics: MetricPoint[]
  artifacts: Artifact[]
  events: TaskEvent[]
  onClose: () => void
}

const detailMeta = {
  runtime: { eyebrow: 'RUNTIME DETAILS', title: '运行环境详情', icon: Cpu, accent: '#ffba6b' },
  status: { eyebrow: 'TRAINING RUN', title: '训练状态详情', icon: Gauge, accent: '#70f0bd' },
  metrics: { eyebrow: 'METRICS EXPLORER', title: '完整训练指标', icon: Activity, accent: '#53d6e8' },
  artifacts: { eyebrow: 'RUN ARTIFACTS', title: '训练产物浏览', icon: FileArchive, accent: '#ffd56d' },
  weights: { eyebrow: 'MODEL CHECKPOINTS', title: '模型权重详情', icon: Save, accent: '#7da8ff' },
} satisfies Record<CanvasDetailKind, { eyebrow: string; title: string; icon: typeof Cpu; accent: string }>

const launchModeLabel = { fresh: '首次训练', resume: '继续训练', retry: '重新训练' }

function formatDate(value: string | null | undefined) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function formatDuration(seconds: number | null | undefined) {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const rest = Math.floor(seconds % 60)
  return `${hours ? `${hours}时 ` : ''}${minutes}分 ${rest}秒`
}

function runDuration(run: TrainingRun | null) {
  const startedAt = run?.process_started_at || run?.started_at
  if (!startedAt) return null
  const start = new Date(startedAt).getTime()
  const end = run.finished_at ? new Date(run.finished_at).getTime() : Date.now()
  return Number.isFinite(start) && Number.isFinite(end) ? Math.max(0, (end - start) / 1000) : null
}

function displayValue(value: unknown) {
  if (value == null || value === '') return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(5).replace(/0+$/, '').replace(/\.$/, '')
  if (typeof value === 'boolean') return value ? '是' : '否'
  return String(value)
}

function DetailField({ label, value, wide = false, mono = false }: { label: string; value: unknown; wide?: boolean; mono?: boolean }) {
  return <div className={`canvas-detail-field ${wide ? 'wide' : ''}`}><span>{label}</span><strong className={mono ? 'mono' : ''} title={displayValue(value)}>{displayValue(value)}</strong></div>
}

function metricOption(metrics: MetricPoint[], keys: Array<{ key: string; name: string; color: string }>, max?: number) {
  return {
    animationDuration: 260,
    grid: { left: 52, right: 18, top: 40, bottom: 34 },
    tooltip: { trigger: 'axis', backgroundColor: '#111a24', borderColor: '#344654', textStyle: { color: '#d9e4ed' } },
    legend: { data: keys.map((item) => item.name), top: 5, textStyle: { color: '#8293a0', fontSize: 10 } },
    xAxis: { type: 'category', name: 'Epoch', nameTextStyle: { color: '#70818e' }, data: metrics.map((point) => point.epoch), axisLabel: { color: '#71818e' }, axisLine: { lineStyle: { color: '#34434f' } } },
    yAxis: { type: 'value', min: 0, max, axisLabel: { color: '#71818e' }, splitLine: { lineStyle: { color: '#26343f' } } },
    series: keys.map((item) => ({
      name: item.name,
      type: 'line',
      showSymbol: false,
      smooth: true,
      data: metrics.map((point) => typeof point.values[item.key] === 'number' ? point.values[item.key] : null),
      lineStyle: { color: item.color, width: 2 },
      itemStyle: { color: item.color },
    })),
  }
}

function RuntimeDetails({ task, run }: Pick<Props, 'task' | 'run'>) {
  return <div className="canvas-detail-sections">
    <section className="canvas-detail-section">
      <h3>进程与设备</h3>
      <div className="canvas-detail-grid">
        <DetailField label="设备" value={`GPU ${task.device}`} />
        <DetailField label="进程 PID" value={task.pid} />
        <DetailField label="运行状态" value={statusLabel(task.status)} />
        <DetailField label="退出代码" value={task.exit_code} />
        <DetailField label="进程启动时间" value={formatDate(run?.process_started_at)} />
        <DetailField label="进程运行时长" value={formatDuration(runDuration(run))} />
      </div>
    </section>
    <section className="canvas-detail-section">
      <h3>文件与输出路径</h3>
      <div className="canvas-detail-grid">
        <DetailField label="基础模型" value={task.model_path} wide mono />
        <DetailField label="数据配置" value={task.data_path} wide mono />
        <DetailField label="输出目录" value={task.run_dir || '等待创建'} wide mono />
        <DetailField label="日志文件" value={task.log_path || '等待创建'} wide mono />
        <DetailField label="恢复检查点" value={run?.checkpoint_path || '未使用'} wide mono />
      </div>
    </section>
    <section className="canvas-detail-section">
      <h3>启动命令</h3>
      <pre className="canvas-command"><Terminal size={15} />{run?.command?.length ? run.command.join(' ') : '该运行没有保存启动命令（可能是历史目录导入任务）'}</pre>
    </section>
  </div>
}

function StatusDetails({ task, run, events }: Pick<Props, 'task' | 'run' | 'events'>) {
  const metricEntries = Object.entries(task.latest_metrics || {})
  return <div className="canvas-detail-sections">
    <div className="run-summary-strip">
      <div><span>当前状态</span><strong className={`detail-status ${task.status}`}>{statusLabel(task.status)}</strong></div>
      <div><span>运行批次</span><strong>{run ? `Attempt ${String(run.attempt_number).padStart(3, '0')}` : '未创建'}</strong></div>
      <div><span>启动方式</span><strong>{run ? launchModeLabel[run.launch_mode] : '—'}</strong></div>
      <div><span>训练进度</span><strong>{Math.round(task.progress)}%</strong></div>
    </div>
    <section className="canvas-detail-section">
      <div className="detail-progress-heading"><h3>Epoch {task.completed_epochs} / {task.epochs}</h3><span>{task.progress.toFixed(2)}%</span></div>
      <div className="detail-progress"><i style={{ width: `${task.progress}%` }} /></div>
    </section>
    {task.error_message && <div className="canvas-detail-error"><strong>本次运行错误</strong><pre>{task.error_message}</pre></div>}
    <section className="canvas-detail-section">
      <h3>运行时间</h3>
      <div className="canvas-detail-grid">
        <DetailField label="进入队列" value={formatDate(task.queued_at)} />
        <DetailField label="开始训练" value={formatDate(task.started_at)} />
        <DetailField label="训练结束" value={formatDate(task.finished_at)} />
        <DetailField label="退出代码" value={task.exit_code} />
      </div>
    </section>
    <section className="canvas-detail-section">
      <h3>最新指标</h3>
      {metricEntries.length ? <div className="metric-value-grid">{metricEntries.map(([key, value]) => <div key={key}><span title={key}>{key}</span><strong>{displayValue(value)}</strong></div>)}</div> : <div className="canvas-detail-empty">暂无指标数据</div>}
    </section>
    <section className="canvas-detail-section">
      <h3>任务事件 <span>{events.length}</span></h3>
      {events.length ? <div className="event-timeline">{events.map((event) => <article key={event.id}><i className={event.level} /><div><strong>{event.message}</strong><span>{event.event_type} · {formatDate(event.created_at)}</span></div></article>)}</div> : <div className="canvas-detail-empty">本次运行暂无事件记录</div>}
    </section>
  </div>
}

function MetricsDetails({ metrics, task }: Pick<Props, 'metrics' | 'task'>) {
  const latest = metrics.at(-1)
  const summaryKeys = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)', 'train/box_loss', 'val/box_loss']
  const accuracyOption = useMemo(() => metricOption(metrics, [
    { key: 'metrics/mAP50(B)', name: 'mAP50', color: '#70f0bd' },
    { key: 'metrics/mAP50-95(B)', name: 'mAP50-95', color: '#6da8ff' },
    { key: 'metrics/precision(B)', name: 'Precision', color: '#ffd56d' },
    { key: 'metrics/recall(B)', name: 'Recall', color: '#d99cff' },
  ], 1), [metrics])
  const lossOption = useMemo(() => metricOption(metrics, [
    { key: 'train/box_loss', name: 'Train box', color: '#70f0bd' },
    { key: 'train/cls_loss', name: 'Train cls', color: '#53d6e8' },
    { key: 'val/box_loss', name: 'Val box', color: '#ff9d76' },
    { key: 'val/cls_loss', name: 'Val cls', color: '#ffd56d' },
  ]), [metrics])
  if (!metrics.length) return <div className="canvas-detail-empty large">训练产生首个 Epoch 后，这里会显示完整指标曲线。</div>
  return <div className="canvas-detail-sections">
    <div className="metric-summary-strip">{summaryKeys.map((key) => <div key={key}><span>{key.replace('metrics/', '').replace('train/', '').replace('val/', '')}</span><strong>{displayValue(latest?.values[key] ?? task.latest_metrics[key])}</strong></div>)}</div>
    <section className="canvas-detail-chart"><h3>精度与召回趋势</h3><ReactECharts option={accuracyOption} style={{ height: 330, width: '100%' }} /></section>
    <section className="canvas-detail-chart"><h3>训练与验证损失</h3><ReactECharts option={lossOption} style={{ height: 330, width: '100%' }} /></section>
    <section className="canvas-detail-section">
      <h3>最新 Epoch 完整数据 <span>Epoch {latest?.epoch}</span></h3>
      <div className="metric-value-grid">{Object.entries(latest?.values || {}).map(([key, value]) => <div key={key}><span title={key}>{key}</span><strong>{displayValue(value)}</strong></div>)}</div>
    </section>
  </div>
}

function ArtifactsDetails({ artifacts }: Pick<Props, 'artifacts'>) {
  const files = artifacts.filter((item) => item.kind !== 'weight')
  const images = files.filter((item) => item.kind === 'image')
  if (!files.length) return <div className="canvas-detail-empty large">当前运行暂未生成训练产物。</div>
  return <div className="canvas-detail-sections">
    {images.length > 0 && <section className="canvas-detail-section"><h3>图片预览 <span>{images.length}</span></h3><div className="artifact-gallery">{images.map((item) => <a key={item.id} href={`/api/artifacts/${item.id}/download`} target="_blank" rel="noreferrer" title={`打开 ${item.name}`}><img src={`/api/artifacts/${item.id}/download`} alt={item.name} loading="lazy" /><span><Image size={13} />{item.name}</span></a>)}</div></section>}
    <section className="canvas-detail-section">
      <h3>全部文件 <span>{files.length}</span></h3>
      <div className="artifact-detail-list">{files.map((item) => <article key={item.id}><div className={`artifact-kind kind-${item.kind}`}><FileText size={16} /></div><div><strong>{item.name}</strong><span title={item.path}>{item.path}</span></div><small>{item.kind.toUpperCase()} · {formatBytes(item.size)} · {formatDate(item.modified_at)}</small><a href={`/api/artifacts/${item.id}/download`} title={`下载 ${item.name}`} aria-label={`下载 ${item.name}`}><Download size={16} /></a></article>)}</div>
    </section>
  </div>
}

function WeightsDetails({ artifacts, task, run }: Pick<Props, 'artifacts' | 'task' | 'run'>) {
  const weights = artifacts.filter((item) => item.kind === 'weight')
  const knownWeights = [
    { name: 'best.pt', label: '最佳模型', description: '训练过程中验证指标表现最好的检查点，通常用于部署、推理和导出。', icon: Trophy },
    { name: 'last.pt', label: '最新模型', description: '最后一个 Epoch 保存的检查点，保留训练状态，通常用于中断后继续训练。', icon: History },
  ]
  return <div className="canvas-detail-sections">
    <div className="weight-detail-summary"><div><span>运行批次</span><strong>{run ? `Attempt ${String(run.attempt_number).padStart(3, '0')}` : '—'}</strong></div><div><span>训练状态</span><strong>{statusLabel(task.status)}</strong></div><div><span>权重数量</span><strong>{weights.length}</strong></div><div><span>输出目录</span><strong title={task.run_dir || ''}>{task.run_dir?.split('/').slice(-3).join('/') || '等待创建'}</strong></div></div>
    <div className="weight-card-grid">{knownWeights.map(({ name, label, description, icon: Icon }) => {
      const artifact = weights.find((item) => item.name.split('/').pop() === name)
      return <article className={`weight-detail-card ${artifact ? '' : 'pending'}`} key={name}>
        <div className="weight-card-top"><div className="weight-card-icon"><Icon size={24} /></div><div><span>{label}</span><h3>{name}</h3></div><em>{artifact ? '已生成' : '等待生成'}</em></div>
        <p>{description}</p>
        <div className="weight-card-meta"><div><HardDrive size={14} /><span>文件大小</span><strong>{artifact ? formatBytes(artifact.size) : '—'}</strong></div><div><History size={14} /><span>更新时间</span><strong>{artifact ? formatDate(artifact.modified_at) : '—'}</strong></div></div>
        <div className="weight-card-path" title={artifact?.path}>{artifact?.path || `${task.run_dir || '训练输出目录'}/weights/${name}`}</div>
        {artifact ? <a className="button primary weight-download" href={`/api/artifacts/${artifact.id}/download`}><Download size={16} />下载 {name}</a> : <button className="button ghost weight-download" disabled><Download size={16} />暂不可下载</button>}
      </article>
    })}</div>
    {weights.filter((item) => !['best.pt', 'last.pt'].includes(item.name.split('/').pop() || '')).length > 0 && <section className="canvas-detail-section"><h3>其他模型权重</h3><div className="artifact-detail-list">{weights.filter((item) => !['best.pt', 'last.pt'].includes(item.name.split('/').pop() || '')).map((item) => <article key={item.id}><div className="artifact-kind kind-weight"><Save size={16} /></div><div><strong>{item.name}</strong><span title={item.path}>{item.path}</span></div><small>{formatBytes(item.size)} · {formatDate(item.modified_at)}</small><a href={`/api/artifacts/${item.id}/download`} title={`下载 ${item.name}`}><Download size={16} /></a></article>)}</div></section>}
  </div>
}

export function CanvasDetailModal({ kind, task, run, metrics, artifacts, events, onClose }: Props) {
  useEffect(() => {
    if (!kind) return
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [kind, onClose])

  if (!kind) return null
  const meta = detailMeta[kind]
  const Icon = meta.icon
  return <div className="modal-backdrop canvas-detail-backdrop" onMouseDown={onClose}>
    <div className={`modal canvas-detail-modal detail-${kind}`} role="dialog" aria-modal="true" aria-labelledby="canvas-detail-title" style={{ '--detail-accent': meta.accent } as React.CSSProperties} onMouseDown={(event) => event.stopPropagation()}>
      <header><div className="modal-heading"><div className="heading-icon canvas-detail-icon"><Icon size={19} /></div><div><span className="eyebrow">{meta.eyebrow}</span><h2 id="canvas-detail-title">{meta.title}</h2><p>{task.name}{run ? ` · Attempt ${String(run.attempt_number).padStart(3, '0')}` : ''}</p></div></div><button className="icon-button" title="关闭" aria-label="关闭详情弹窗" onClick={onClose}><X size={20} /></button></header>
      <div className="canvas-detail-body">
        {kind === 'runtime' && <RuntimeDetails task={task} run={run} />}
        {kind === 'status' && <StatusDetails task={task} run={run} events={events} />}
        {kind === 'metrics' && <MetricsDetails task={task} metrics={metrics} />}
        {kind === 'artifacts' && <ArtifactsDetails artifacts={artifacts} />}
        {kind === 'weights' && <WeightsDetails task={task} run={run} artifacts={artifacts} />}
      </div>
      <footer><span>双击画布中的其他节点可切换查看对应内容</span><button className="button ghost" onClick={onClose}>关闭</button></footer>
    </div>
  </div>
}
