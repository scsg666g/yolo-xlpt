import { useCallback, useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Activity, AlertCircle, BarChart3, Check, ChevronDown, Cpu, Gauge, Layers3, Search, Trophy, X } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api, metricValue, statusLabel } from '../api'
import type { MetricPoint, TrainingRun, TrainingTask } from '../types'

interface ComparisonRun {
  id: string
  task: TrainingTask
  run: TrainingRun
  view: TrainingTask
}

const palette = ['#159a78', '#347bd1', '#a45fc7', '#e28a32']
const metricOptions = [
  { key: 'metrics/mAP50(B)', label: 'mAP50', group: 'accuracy' },
  { key: 'metrics/mAP50-95(B)', label: 'mAP50-95', group: 'accuracy' },
  { key: 'metrics/precision(B)', label: 'Precision', group: 'accuracy' },
  { key: 'metrics/recall(B)', label: 'Recall', group: 'accuracy' },
  { key: 'train/box_loss', label: 'Train Box Loss', group: 'loss' },
  { key: 'val/box_loss', label: 'Val Box Loss', group: 'loss' },
  { key: 'train/cls_loss', label: 'Train Cls Loss', group: 'loss' },
  { key: 'val/cls_loss', label: 'Val Cls Loss', group: 'loss' },
]

const parameterRows: Array<{ label: string; value: (task: TrainingTask) => string | number }> = [
  { label: '任务类型', value: (task) => task.task_type },
  { label: '基础模型', value: (task) => task.model_path.split('/').pop() || task.model_path },
  { label: '数据集', value: (task) => task.data_path.split('/').pop() || task.data_path },
  { label: '训练轮数', value: (task) => task.epochs },
  { label: 'Batch size', value: (task) => task.batch },
  { label: '输入尺寸', value: (task) => task.imgsz },
  { label: 'GPU 设备', value: (task) => task.device },
  { label: 'Workers', value: (task) => task.workers },
  { label: '优化器', value: (task) => task.optimizer },
  { label: '初始学习率', value: (task) => String(task.config.lr0 ?? '自动') },
  { label: 'Patience', value: (task) => String(task.config.patience ?? '—') },
  { label: 'AMP', value: (task) => task.config.amp === false ? '关闭' : '开启' },
  { label: 'Mosaic', value: (task) => String(task.config.mosaic ?? '—') },
  { label: 'MixUp', value: (task) => String(task.config.mixup ?? '—') },
  { label: 'Weight decay', value: (task) => String(task.config.weight_decay ?? '—') },
  { label: 'Momentum', value: (task) => String(task.config.momentum ?? '—') },
]

function numberMetric(task: TrainingTask, keys: string[]): number | null {
  for (const key of keys) if (typeof task.latest_metrics[key] === 'number') return task.latest_metrics[key] as number
  return null
}

function elapsedLabel(seconds: number | undefined): string {
  if (!seconds) return '—'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  return `${hours}h ${minutes}m`
}

function runView(task: TrainingTask, run: TrainingRun): TrainingTask {
  const snapshot = run.parameter_snapshot || {}
  return {
    ...task,
    status: run.status,
    model_path: run.model_path,
    data_path: run.data_path,
    epochs: Number(snapshot.epochs ?? task.epochs),
    batch: Number(snapshot.batch ?? task.batch),
    imgsz: Number(snapshot.imgsz ?? task.imgsz),
    device: String(snapshot.device ?? task.device),
    workers: Number(snapshot.workers ?? task.workers),
    optimizer: String(snapshot.optimizer ?? task.optimizer),
    config: (snapshot.config as Record<string, unknown> | undefined) || {},
    run_dir: run.run_dir,
    log_path: run.log_path,
    completed_epochs: run.completed_epochs,
    progress: run.progress,
    latest_metrics: run.latest_metrics,
    pid: run.pid,
    exit_code: run.exit_code,
    error_message: run.error_message,
    queued_at: run.queued_at,
    started_at: run.started_at,
    finished_at: run.finished_at,
  }
}

export function ExperimentCompare() {
  const [experiments, setExperiments] = useState<ComparisonRun[]>([])
  const [metrics, setMetrics] = useState<Record<string, MetricPoint[]>>({})
  const [metricKey, setMetricKey] = useState('metrics/mAP50(B)')
  const [search, setSearch] = useState('')
  const [pickerOpen, setPickerOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [params, setParams] = useSearchParams()
  const [selectedIds, setSelectedIds] = useState<string[]>(() => {
    const incoming = (params.get('runs') || params.get('tasks') || '').split(',').filter(Boolean)
    let persisted: string[] = []
    try { persisted = JSON.parse(localStorage.getItem('yolo-forge-compare-runs') || localStorage.getItem('yolo-forge-compare-tasks') || '[]') } catch { persisted = [] }
    return [...new Set([...persisted, ...incoming])].slice(0, 4)
  })
  const selected = useMemo(() => selectedIds.map((id) => experiments.find((item) => item.id === id)).filter(Boolean) as ComparisonRun[], [experiments, selectedIds])

  useEffect(() => {
    localStorage.setItem('yolo-forge-compare-runs', JSON.stringify(selectedIds))
    const next = new URLSearchParams(window.location.search)
    next.delete('tasks')
    if (selectedIds.length) next.set('runs', selectedIds.join(',')); else next.delete('runs')
    if (next.toString() !== params.toString()) setParams(next, { replace: true })
  }, [selectedIds, params, setParams])

  const load = useCallback(async () => {
    try {
      const taskList = await api.tasks()
      const runLists = await Promise.all(taskList.map(async (task) => [task, await api.runs(task.id)] as const))
      const items = runLists.flatMap(([task, runs]) => runs.map((run) => ({ id: run.id, task, run, view: runView(task, run) })))
      setExperiments(items)
      const normalizedIds = selectedIds.map((id) => items.find((item) => item.id === id)?.id || items.find((item) => item.task.id === id && item.id === item.task.current_run_id)?.id).filter(Boolean) as string[]
      if (normalizedIds.join(',') !== selectedIds.join(',')) setSelectedIds(normalizedIds.slice(0, 4))
      const selectedItems = normalizedIds.map((id) => items.find((item) => item.id === id)).filter(Boolean) as ComparisonRun[]
      const points = await Promise.all(selectedItems.map(async (item) => [item.id, await api.runMetrics(item.task.id, item.run.id)] as const))
      setMetrics(Object.fromEntries(points)); setError('')
    } catch (reason) { setError(reason instanceof Error ? reason.message : '加载对比数据失败') } finally { setLoading(false) }
  }, [selectedIds])

  useEffect(() => { load(); const timer = window.setInterval(load, 5000); return () => window.clearInterval(timer) }, [load])

  function setSelection(ids: string[]) {
    setSelectedIds([...new Set(ids)].slice(0, 4))
  }

  function toggleTask(id: string) {
    if (selectedIds.includes(id)) setSelection(selectedIds.filter((value) => value !== id))
    else if (selectedIds.length < 4) setSelection([...selectedIds, id])
  }

  const bestMap = useMemo(() => selected.reduce<ComparisonRun | null>((best, item) => {
    const value = numberMetric(item.view, ['metrics/mAP50(B)']) ?? -1
    return !best || value > (numberMetric(best.view, ['metrics/mAP50(B)']) ?? -1) ? item : best
  }, null), [selected])
  const lowestLoss = useMemo(() => selected.reduce<ComparisonRun | null>((best, item) => {
    const value = numberMetric(item.view, ['val/box_loss', 'train/box_loss']) ?? Infinity
    return !best || value < (numberMetric(best.view, ['val/box_loss', 'train/box_loss']) ?? Infinity) ? item : best
  }, null), [selected])
  const fastest = useMemo(() => selected.reduce<ComparisonRun | null>((best, item) => {
    const value = metrics[item.id]?.at(-1)?.elapsed ?? Infinity
    const bestValue = best ? metrics[best.id]?.at(-1)?.elapsed ?? Infinity : Infinity
    return !best || value < bestValue ? item : best
  }, null), [selected, metrics])

  const chart = useMemo(() => ({
    animationDuration: 350,
    color: palette,
    tooltip: { trigger: 'axis', backgroundColor: '#14202a', borderColor: '#334754', textStyle: { color: '#e7eef2' } },
    legend: { data: selected.map((item) => `${item.task.name} · A${String(item.run.attempt_number).padStart(3, '0')}`), top: 5, textStyle: { color: '#6c8290', fontSize: 10 } },
    grid: { left: 55, right: 22, top: 48, bottom: 40 },
    xAxis: { type: 'value', name: 'Epoch', nameTextStyle: { color: '#728692' }, axisLabel: { color: '#718692' }, axisLine: { lineStyle: { color: '#bccdd5' } }, splitLine: { show: false } },
    yAxis: { type: 'value', axisLabel: { color: '#718692' }, axisLine: { show: false }, splitLine: { lineStyle: { color: '#dbe6eb', type: 'dashed' } } },
    series: selected.map((item, index) => ({
      name: `${item.task.name} · A${String(item.run.attempt_number).padStart(3, '0')}`, type: 'line', smooth: true, showSymbol: false, symbolSize: 5,
      lineStyle: { width: 2.4, color: palette[index] }, itemStyle: { color: palette[index] },
      areaStyle: { opacity: 0.035, color: palette[index] },
      data: (metrics[item.id] || []).map((point) => [point.epoch, typeof point.values[metricKey] === 'number' ? point.values[metricKey] : null]),
    })),
  }), [selected, metrics, metricKey])

  const filteredExperiments = experiments.filter((item) => item.task.name.toLowerCase().includes(search.toLowerCase()) || item.view.model_path.toLowerCase().includes(search.toLowerCase()) || `attempt ${item.run.attempt_number}`.includes(search.toLowerCase()))

  return <div className="page compare-page">
    <header className="page-header compare-header"><div><span className="eyebrow">EXPERIMENT ANALYSIS</span><h1>实验对比</h1><p>并排比较训练配置、收敛趋势和最终效果。</p></div><button className="button primary" onClick={() => setPickerOpen((value) => !value)}><Layers3 size={16} />选择实验 <span className="selection-count">{selected.length}/4</span><ChevronDown size={15} /></button></header>
    {pickerOpen && <div className="compare-picker">
      <header><div><strong>选择参与对比的训练运行</strong><span>已选择 {selectedIds.length} 条，最多同时选择 4 条；同一任务可选择多个 Attempt</span></div><div className="picker-header-actions"><button className="picker-done" onClick={() => setPickerOpen(false)}>完成</button><button className="icon-button" onClick={() => setPickerOpen(false)}><X size={18} /></button></div></header>
      <div className="compare-picker-search"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索任务、模型或 Attempt" /></div>
      <div className="compare-picker-list">{filteredExperiments.map((item) => { const checked = selectedIds.includes(item.id); return <button key={item.id} onClick={() => toggleTask(item.id)} disabled={!checked && selectedIds.length >= 4}><i className={checked ? 'checked' : ''}>{checked && <Check size={12} />}</i><span><strong>{item.task.name} · Attempt {String(item.run.attempt_number).padStart(3, '0')}</strong><small>{item.view.model_path.split('/').pop()} · Epoch {item.view.completed_epochs}/{item.view.epochs}</small></span><em className={`status ${item.run.status}`}>{statusLabel(item.run.status)}</em></button> })}</div>
    </div>}
    {error && <div className="alert error">{error}</div>}
    {loading ? <div className="compare-empty">正在载入实验数据…</div> : !selected.length ? <div className="compare-empty"><Activity size={34} /><strong>选择要比较的训练任务</strong><span>可同时选择 2～4 条任务，查看参数差异和指标曲线。</span><button className="button primary" onClick={() => setPickerOpen(true)}>选择训练任务</button></div> : <>
      {selected.length < 2 && <div className="compare-notice"><AlertCircle size={17} /><div><strong>再选择一条任务即可开始对比</strong><span>当前仍会展示这条任务的配置和训练指标。</span></div><button className="button ghost" onClick={() => setPickerOpen(true)}>继续选择</button></div>}
      <section className="selected-experiments">{selected.map((item, index) => <article key={item.id} style={{ '--experiment-color': palette[index] } as React.CSSProperties}><button onClick={() => setSelection(selectedIds.filter((id) => id !== item.id))}><X size={14} /></button><span>ATTEMPT {String(item.run.attempt_number).padStart(3, '0')} · {item.run.launch_mode.toUpperCase()}</span><h3>{item.task.name}</h3><p>{item.view.model_path.split('/').pop()} · {item.view.task_type}</p><div><i style={{ width: `${item.view.progress}%` }} /></div><footer><strong>{metricValue(item.view, ['metrics/mAP50(B)'])}</strong><small>mAP50</small><strong>{Math.round(item.view.progress)}%</strong><small>进度</small></footer></article>)}</section>
      <section className="compare-summary">
        <article><div className="summary-icon trophy"><Trophy size={18} /></div><span>最高 mAP50</span><strong>{bestMap ? metricValue(bestMap.view, ['metrics/mAP50(B)']) : '—'}</strong><small>{bestMap ? `${bestMap.task.name} · A${String(bestMap.run.attempt_number).padStart(3, '0')}` : '等待指标'}</small></article>
        <article><div className="summary-icon loss"><Gauge size={18} /></div><span>最低 Box Loss</span><strong>{lowestLoss ? metricValue(lowestLoss.view, ['val/box_loss', 'train/box_loss']) : '—'}</strong><small>{lowestLoss ? `${lowestLoss.task.name} · A${String(lowestLoss.run.attempt_number).padStart(3, '0')}` : '等待指标'}</small></article>
        <article><div className="summary-icon speed"><Activity size={18} /></div><span>最短训练耗时</span><strong>{fastest ? elapsedLabel(metrics[fastest.id]?.at(-1)?.elapsed) : '—'}</strong><small>{fastest ? `${fastest.task.name} · A${String(fastest.run.attempt_number).padStart(3, '0')}` : '等待指标'}</small></article>
        <article><div className="summary-icon gpu"><Cpu size={18} /></div><span>参与运行</span><strong>{selected.length}</strong><small>{new Set(selected.map((item) => item.view.device)).size} 种设备配置</small></article>
      </section>
      <section className="compare-panel chart-panel"><header><div><BarChart3 size={17} /><div><h2>训练指标趋势</h2><span>按 Epoch 对齐叠加曲线</span></div></div><select value={metricKey} onChange={(event) => setMetricKey(event.target.value)}>{metricOptions.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}</select></header><ReactECharts option={chart} style={{ height: 390 }} /></section>
      <section className="compare-panel parameter-panel"><header><div><Layers3 size={17} /><div><h2>训练参数差异</h2><span>来自每次运行的固化参数快照</span></div></div></header><div className="comparison-table"><table><thead><tr><th>参数</th>{selected.map((item, index) => <th key={item.id}><i style={{ background: palette[index] }} />{item.task.name} · A{String(item.run.attempt_number).padStart(3, '0')}</th>)}</tr></thead><tbody>{parameterRows.map((parameter) => { const values = selected.map((item) => parameter.value(item.view)); const different = new Set(values.map(String)).size > 1; return <tr className={different ? 'different' : ''} key={parameter.label}><td>{parameter.label}{different && <em>不同</em>}</td>{values.map((value, index) => <td key={`${parameter.label}-${selected[index].id}`}>{value}</td>)}</tr> })}</tbody></table></div></section>
    </>}
  </div>
}
