import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, AlertCircle, ArrowDown, ArrowUp, BarChart3, Check, ChevronDown, Layers3, Minus, Search, Trophy, X } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api, metricValue, statusLabel } from '../api'
import type { MetricPoint, TrainingRun, TrainingTask } from '../types'
import { LineChartView } from './LineChart'

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

interface SummaryMetric {
  key: string
  label: string
  direction: 'higher' | 'lower' | 'neutral'
  unit: 'points' | 'value'
}

const summaryMetricRows: SummaryMetric[] = [
  { key: 'metrics/mAP50(B)', label: 'mAP50', direction: 'higher', unit: 'points' },
  { key: 'metrics/mAP50-95(B)', label: 'mAP50-95', direction: 'higher', unit: 'points' },
  { key: 'metrics/precision(B)', label: 'Precision', direction: 'higher', unit: 'points' },
  { key: 'metrics/recall(B)', label: 'Recall', direction: 'higher', unit: 'points' },
  { key: 'val/box_loss', label: '验证 Box Loss', direction: 'lower', unit: 'value' },
  { key: 'train/box_loss', label: '训练 Box Loss', direction: 'neutral', unit: 'value' },
]

function displayedMetric(values: MetricPoint['values'] | undefined, key: string): number | null {
  const value = values?.[key]
  // Compare at the displayed precision so equal-looking values also tie.
  return typeof value === 'number' && Number.isFinite(value) ? Number(value.toFixed(4)) : null
}

function bestEvaluationPoint(points: MetricPoint[] | undefined): MetricPoint | undefined {
  let best: MetricPoint | undefined
  let bestValue = -Infinity
  for (const point of points || []) {
    const value = point.values['metrics/mAP50-95(B)']
    if (typeof value === 'number' && Number.isFinite(value) && value > bestValue) { best = point; bestValue = value }
  }
  return best
}

function MetricComparisonCell({ value, baselineValue, baseline, best, tied, row, comparisonEnabled }: {
  value: number | null
  baselineValue: number | null
  baseline: boolean
  best: boolean
  tied: boolean
  row: SummaryMetric
  comparisonEnabled: boolean
}) {
  const delta = value !== null && baselineValue !== null ? Number((value - baselineValue).toFixed(4)) : null
  const ranked = comparisonEnabled && row.direction !== 'neutral'
  const improvement = delta === null || delta === 0 || !ranked ? 'neutral' : (row.direction === 'higher' ? delta > 0 : delta < 0) ? 'improved' : 'regressed'
  const Arrow = delta === null || delta === 0 ? Minus : delta > 0 ? ArrowUp : ArrowDown
  return <td className={best ? 'metric-best-cell' : ''}>
    <div className="metric-value-line"><strong>{value === null ? '暂无数据' : value.toFixed(4)}</strong>{best && <span className="metric-best-badge"><Trophy size={11} />{tied ? '并列最优' : '本项最优'}</span>}</div>
    {comparisonEnabled && !baseline && value !== null && <small className={`metric-delta ${improvement}`}>
      {delta === null ? '基准暂无数据' : <><Arrow size={12} />{delta === 0 ? '持平' : row.unit === 'points' ? `${Math.abs(delta * 100).toFixed(2)} 个百分点` : Math.abs(delta).toFixed(4)}{delta !== 0 && ranked && <span>{improvement === 'improved' ? '改善' : '下降'}</span>}</>}
    </small>}
  </td>
}

function bestMapLabel(points: MetricPoint[] | undefined): string {
  let best: { value: number; epoch: number } | undefined
  for (const point of points || []) {
    const value = point.values['metrics/mAP50(B)']
    if (typeof value === 'number' && Number.isFinite(value) && (!best || value > best.value)) best = { value, epoch: point.epoch }
  }
  return best ? `${best.value.toFixed(4)} · 第 ${best.epoch} 轮` : '暂无数据'
}

function elapsedLabel(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return '暂无数据'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  if (hours) return `${hours} 小时 ${minutes} 分钟`
  return minutes ? `${minutes} 分钟 ${Math.floor(seconds % 60)} 秒` : `${Math.floor(seconds)} 秒`
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
  const [metricErrors, setMetricErrors] = useState<Record<string, string>>({})
  const requestVersion = useRef(0)
  const activeRequest = useRef<AbortController | null>(null)
  const [metricKey, setMetricKey] = useState('metrics/mAP50(B)')
  const [baselineId, setBaselineId] = useState('')
  const [evaluationMode, setEvaluationMode] = useState<'latest' | 'best'>('latest')
  const [search, setSearch] = useState('')
  const [pickerOpen, setPickerOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [params, setParams] = useSearchParams()
  const [selectedIds, setSelectedIds] = useState<string[]>(() => {
    const incoming = (params.get('runs') || params.get('tasks') || '').split(',').filter(Boolean)
    if (params.has('runs') || params.has('tasks')) return [...new Set(incoming)].slice(0, 4)
    try {
      const persisted: unknown = JSON.parse(localStorage.getItem('yolo-forge-compare-runs') || localStorage.getItem('yolo-forge-compare-tasks') || '[]')
      return Array.isArray(persisted) ? [...new Set(persisted.filter((id): id is string => typeof id === 'string' && !!id))].slice(0, 4) : []
    } catch { return [] }
  })
  const selected = useMemo(() => selectedIds.map((id) => experiments.find((item) => item.id === id)).filter(Boolean) as ComparisonRun[], [experiments, selectedIds])

  useEffect(() => {
    localStorage.setItem('yolo-forge-compare-runs', JSON.stringify(selectedIds))
    const next = new URLSearchParams(window.location.search)
    next.delete('tasks')
    if (selectedIds.length) next.set('runs', selectedIds.join(',')); else next.delete('runs')
    if (next.toString() !== params.toString()) setParams(next, { replace: true })
  }, [selectedIds, params, setParams])

  const invalidateActiveRequest = useCallback(() => {
    requestVersion.current++
    activeRequest.current?.abort()
    activeRequest.current = null
  }, [])

  const load = useCallback(async () => {
    const version = ++requestVersion.current
    activeRequest.current?.abort()
    const controller = new AbortController()
    activeRequest.current = controller
    try {
      const taskList = await api.tasks(controller.signal)
      const runLists = await Promise.all(taskList.map(async (task) => [task, await api.runs(task.id, controller.signal)] as const))
      const items = runLists.flatMap(([task, runs]) => {
        const run = runs.find((item) => item.id === task.current_run_id) || runs[0]
        return run ? [{ id: run.id, task, run, view: runView(task, run) }] : []
      })
      if (version !== requestVersion.current) return
      setExperiments(items)
      const normalizedIds = selectedIds.map((id) => items.find((item) => item.id === id)?.id || items.find((item) => item.task.id === id && item.id === item.task.current_run_id)?.id).filter(Boolean) as string[]
      if (normalizedIds.join(',') !== selectedIds.join(',')) setSelectedIds(normalizedIds.slice(0, 4))
      const selectedItems = normalizedIds.map((id) => items.find((item) => item.id === id)).filter(Boolean) as ComparisonRun[]
      const results = await Promise.allSettled(selectedItems.map((item) => api.runMetrics(item.task.id, item.run.id, controller.signal)))
      if (version !== requestVersion.current) return
      const points: Record<string, MetricPoint[]> = {}
      const failures: Record<string, string> = {}
      results.forEach((result, index) => {
        const id = selectedItems[index].id
        if (result.status === 'fulfilled') points[id] = result.value
        else failures[id] = result.reason instanceof Error ? result.reason.message : '指标加载失败'
      })
      setMetrics(points); setMetricErrors(failures); setError('')
    } catch (reason) {
      if (!controller.signal.aborted && version === requestVersion.current) setError(reason instanceof Error ? reason.message : '加载对比数据失败')
    } finally {
      if (activeRequest.current === controller) activeRequest.current = null
      if (!controller.signal.aborted && version === requestVersion.current) setLoading(false)
    }
  }, [selectedIds])

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    async function poll() {
      await load()
      if (!cancelled) timer = window.setTimeout(poll, 5000)
    }

    void poll()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
      invalidateActiveRequest()
    }
  }, [invalidateActiveRequest, load])

  function setSelection(ids: string[]) {
    invalidateActiveRequest()
    const next = [...new Set(ids)].slice(0, 4)
    setSelectedIds(next)
    setBaselineId((current) => next.includes(current) ? current : next[0] || '')
  }

  function toggleTask(id: string) {
    if (selectedIds.includes(id)) setSelection(selectedIds.filter((value) => value !== id))
    else if (selectedIds.length < 4) setSelection([...selectedIds, id])
  }

  const differentDatasets = new Set(selected.map((item) => item.view.data_path)).size > 1
  const differentTaskTypes = new Set(selected.map((item) => item.view.task_type)).size > 1
  const comparisonEnabled = selected.length > 1 && !differentDatasets && !differentTaskTypes && selected.every((item) => !!item.view.data_path)
  const lossComparable = new Set(selected.map((item) => item.view.model_path)).size === 1 && new Set(selected.map((item) => item.view.imgsz)).size === 1
  const baselineRun = selected.find((item) => item.id === baselineId) || selected[0]
  const evaluationPoints = useMemo(() => Object.fromEntries(selected.map((item) => [item.id, bestEvaluationPoint(metrics[item.id])])), [selected, metrics])
  const evaluationValues = (item: ComparisonRun) => evaluationMode === 'latest' ? item.run.latest_metrics : evaluationPoints[item.id]?.values

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
  const experimentCount = Math.max(selected.length, 1)
  const experimentLayout = {
    '--experiment-count': experimentCount,
    '--experiment-column-width': `calc(${100 / experimentCount}% - ${170 / experimentCount}px)`,
    '--parameter-table-min-width': `${170 + experimentCount * 180}px`,
  } as React.CSSProperties

  return <div className="page compare-page" style={experimentLayout}>
    <header className="page-header compare-header"><div><span className="eyebrow">EXPERIMENT ANALYSIS</span><h1>实验对比</h1><p>并排比较不同任务当前运行的训练配置、收敛趋势和指标。</p></div><button className="button primary" onClick={() => setPickerOpen((value) => !value)}><Layers3 size={16} />选择任务 <span className="selection-count">{selected.length}/4</span><ChevronDown size={15} /></button></header>
    {pickerOpen && <div className="compare-picker">
      <header><div><strong>选择参与对比的训练任务</strong><span>已选择 {selectedIds.length} 个，最多同时选择 4 个；每个任务使用当前运行</span></div><div className="picker-header-actions"><button className="picker-done" onClick={() => setPickerOpen(false)}>完成</button><button className="icon-button" onClick={() => setPickerOpen(false)}><X size={18} /></button></div></header>
      <div className="compare-picker-search"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索任务或模型" /></div>
      <div className="compare-picker-list">{filteredExperiments.map((item) => { const checked = selectedIds.includes(item.id); return <button key={item.id} onClick={() => toggleTask(item.id)} disabled={!checked && selectedIds.length >= 4}><i className={checked ? 'checked' : ''}>{checked && <Check size={12} />}</i><span><strong>{item.task.name} · Attempt {String(item.run.attempt_number).padStart(3, '0')}</strong><small>{item.view.model_path.split('/').pop()} · Epoch {item.view.completed_epochs}/{item.view.epochs}</small></span><em className={`status ${item.run.status}`}>{statusLabel(item.run.status)}</em></button> })}</div>
    </div>}
    {error && <div className="alert error">{error}</div>}
    {loading ? <div className="compare-empty">正在载入实验数据…</div> : !selected.length ? <div className="compare-empty"><Activity size={34} /><strong>选择要比较的训练任务</strong><span>可同时选择 2～4 个任务，查看参数差异和指标曲线。</span><button className="button primary" onClick={() => setPickerOpen(true)}>选择训练任务</button></div> : <>
      {selected.length < 2 && <div className="compare-notice"><AlertCircle size={17} /><div><strong>至少需要两个不同任务才能对比</strong><span>当前只有一个任务，暂时无法生成实验对比。</span></div><button className="button ghost" onClick={() => setPickerOpen(true)}>继续选择</button></div>}
      <section className="selected-experiments">{selected.map((item, index) => <article key={item.id} style={{ '--experiment-color': palette[index] } as React.CSSProperties}><button onClick={() => setSelection(selectedIds.filter((id) => id !== item.id))}><X size={14} /></button><span>ATTEMPT {String(item.run.attempt_number).padStart(3, '0')} · {item.run.launch_mode.toUpperCase()}</span><h3>{item.task.name}</h3><p>{item.view.model_path.split('/').pop()} · {item.view.task_type}</p><div><i style={{ width: `${item.view.progress}%` }} /></div><footer><strong>{metricValue(item.view, ['metrics/mAP50(B)'])}</strong><small>mAP50</small><strong>{Math.round(item.view.progress)}%</strong><small>进度</small></footer></article>)}</section>
      {selected.length >= 2 && <>
      <section className="compare-panel metric-summary-panel">
        <header><div><Activity size={17} /><div><h2>实验指标对比</h2><span>每列对应一个任务的当前运行，颜色与下方曲线一致</span></div></div><div className="metric-summary-controls"><label>指标口径<select aria-label="指标口径" value={evaluationMode} onChange={(event) => setEvaluationMode(event.target.value as 'latest' | 'best')}><option value="latest">最新指标</option><option value="best">最佳轮次指标</option></select></label><span className="metric-run-count">{selected.length} 个任务</span></div></header>
        <div className="metric-summary-notes">
          <p>{evaluationMode === 'latest' ? '最新指标截至各实验已记录的训练轮次。' : '最佳轮次按各实验历史最高 mAP50-95 选取，以下六项指标均来自该轮；缺少 mAP50-95 时显示暂无数据。'} mAP50-95 为主要效果参考；训练 Loss 和累计耗时不参与效果排名。</p>
          {comparisonEnabled && <><p>差值相对于基准实验；同数据集配置仍需确认验证集版本、类别与评估条件一致。颜色表示数值差异，小幅变化不代表显著改善。</p><div className="metric-comparison-legend"><span className="improved"><ArrowUp size={12} />改善</span><span className="regressed"><ArrowDown size={12} />下降</span><span><Minus size={12} />持平 / 仅展示差值</span><span><Trophy size={12} />本行有效数据最优</span></div></>}
          {differentDatasets && <p className="dataset-comparison-note"><AlertCircle size={14} />数据集配置路径不同，仅展示指标，不进行优劣高亮或基准差值比较。</p>}
          {differentTaskTypes && <p className="dataset-comparison-note"><AlertCircle size={14} />任务类型不同，仅展示指标，不进行优劣排名。</p>}
          {comparisonEnabled && !lossComparable && <p>模型配置或输入尺寸不同，验证 Loss 仅展示数值差异，不标记最优。</p>}
        </div>
        <div className="comparison-table metric-comparison-table"><table style={{ minWidth: 170 + selected.length * 220 }}>
          <thead><tr><th scope="col">指标</th>{selected.map((item, index) => <th scope="col" key={item.id} style={{ '--experiment-color': palette[index] } as React.CSSProperties}><div className="metric-experiment-heading"><i style={{ background: palette[index] }} /><span>{item.task.name}<small>Attempt {String(item.run.attempt_number).padStart(3, '0')}</small></span></div><button className={`metric-baseline-button ${item.id === baselineRun?.id ? 'active' : ''}`} aria-pressed={item.id === baselineRun?.id} aria-label={`将 ${item.task.name} Attempt ${String(item.run.attempt_number).padStart(3, '0')} 设为基准`} onClick={() => setBaselineId(item.id)}>{item.id === baselineRun?.id ? <><Check size={12} />基准实验</> : '设为基准'}</button></th>)}</tr></thead>
          <tbody>
            <tr><th scope="row">训练状态</th>{selected.map((item) => <td key={item.id}><span className={`status ${item.run.status}`}>{statusLabel(item.run.status)}</span>{metricErrors[item.id] && <small className="metric-load-error" title={metricErrors[item.id]}>曲线数据加载失败</small>}</td>)}</tr>
            <tr><th scope="row">已记录 / 计划轮数</th>{selected.map((item) => <td key={item.id}>{item.view.completed_epochs} / {item.view.epochs}</td>)}</tr>
            {evaluationMode === 'best' && <tr><th scope="row">指标对应轮次</th>{selected.map((item) => {
              const point = evaluationPoints[item.id]
              return <td key={item.id}>{metricErrors[item.id] ? '加载失败' : metrics[item.id] === undefined ? '加载中…' : point ? `第 ${point.epoch} 轮` : '暂无数据'}</td>
            })}</tr>}
            {summaryMetricRows.map((row) => {
              const values = selected.map((item) => displayedMetric(evaluationValues(item), row.key))
              const valid = values.filter((value): value is number => value !== null)
              const direction = row.direction === 'lower' && !lossComparable ? 'neutral' : row.direction
              const best = comparisonEnabled && direction !== 'neutral' && valid.length > 1 ? (direction === 'higher' ? Math.max(...valid) : Math.min(...valid)) : null
              const tied = best !== null && valid.filter((value) => value === best).length > 1
              const baselineValue = baselineRun ? displayedMetric(evaluationValues(baselineRun), row.key) : null
              const label = `${evaluationMode === 'latest' ? '最新' : '最佳轮次'}${row.unit === 'points' ? ' ' : ''}${row.label}`
              return <tr key={row.key}><th scope="row"><span>{label}</span><small className="metric-direction">{row.key === 'metrics/mAP50-95(B)' ? '主指标 · ' : ''}{direction === 'higher' ? '↑ 越高越好' : direction === 'lower' ? '↓ 越低越好 · 辅助参考' : '仅展示差值'}</small></th>{selected.map((item, index) => <MetricComparisonCell key={item.id} value={values[index]} baselineValue={baselineValue} baseline={item.id === baselineRun?.id} best={best !== null && values[index] === best} tied={tied} row={{ ...row, direction }} comparisonEnabled={comparisonEnabled} />)}</tr>
            })}
            <tr><th scope="row">历史最佳 mAP50</th>{selected.map((item) => <td key={item.id}>{metricErrors[item.id] ? '加载失败' : metrics[item.id] === undefined ? '加载中…' : bestMapLabel(metrics[item.id])}</td>)}</tr>
            <tr><th scope="row">累计训练耗时</th>{selected.map((item) => <td key={item.id}>{metricErrors[item.id] ? '加载失败' : metrics[item.id] === undefined ? '加载中…' : elapsedLabel(metrics[item.id]?.at(-1)?.elapsed)}</td>)}</tr>
            <tr className="metric-dataset-row"><th scope="row">数据集配置</th>{selected.map((item) => <td key={item.id} title={item.view.data_path}>{item.view.data_path}</td>)}</tr>
          </tbody>
        </table></div>
      </section>
      <section className="compare-panel chart-panel"><header><div><BarChart3 size={17} /><div><h2>训练指标趋势</h2><span>按 Epoch 对齐叠加曲线</span></div></div><select value={metricKey} onChange={(event) => setMetricKey(event.target.value)}>{metricOptions.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}</select></header><LineChartView option={chart} style={{ height: 390 }} /></section>
      <section className="compare-panel parameter-panel"><header><div><Layers3 size={17} /><div><h2>训练参数差异</h2><span>来自每次运行的固化参数快照</span></div></div></header><div className="comparison-table"><table><thead><tr><th>参数</th>{selected.map((item, index) => <th key={item.id}><i style={{ background: palette[index] }} />{item.task.name} · A{String(item.run.attempt_number).padStart(3, '0')}</th>)}</tr></thead><tbody>{parameterRows.map((parameter) => { const values = selected.map((item) => parameter.value(item.view)); const different = new Set(values.map(String)).size > 1; return <tr className={different ? 'different' : ''} key={parameter.label}><td>{parameter.label}{different && <em>不同</em>}</td>{values.map((value, index) => <td key={`${parameter.label}-${selected[index].id}`}>{value}</td>)}</tr> })}</tbody></table></div></section>
      </>}
    </>}
  </div>
}
