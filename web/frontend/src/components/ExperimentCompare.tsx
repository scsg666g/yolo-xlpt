import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Activity, AlertCircle, ArrowDown, ArrowUp, BarChart3, Check, ChevronDown, Folder, Info, Layers3, LoaderCircle, Minus, Search, Trash2, Trophy, X } from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { api, metricValue, statusLabel } from '../api'
import { readStoredCompareRuns, useCompareLimit, writeStoredCompareRuns } from '../preferences'
import type { MetricPoint, TaskGroup, TrainingRun, TrainingTask } from '../types'
import { CompareRunDetailModal, type CompareRunDetails } from './CompareRunDetailModal'
import { LineChartView } from './LineChart'
import { AttemptDeletionHistory } from './AttemptDeletionHistory'

interface ComparisonRun {
  id: string
  task: TrainingTask
  run: TrainingRun
  view: TrainingTask
}

const palette = ['#159a78', '#347bd1', '#a45fc7', '#e28a32', '#d95c72', '#178fa5', '#b59a28', '#7b75d6']
const collapsedGroupsKey = 'yolo-task-groups-collapsed-v1'
const ungroupedKey = '__ungrouped__'
const activeRunStatuses = new Set<TrainingRun['status']>(['queued', 'starting', 'running', 'stopping'])

function initialCollapsedGroups(): Set<string> {
  try {
    const value = JSON.parse(window.localStorage.getItem(collapsedGroupsKey) || '[]')
    return new Set(Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [])
  } catch { return new Set() }
}

function deletionStatusLabel(status: TrainingRun['deletion_status']): string {
  if (status === 'failed') return '清理失败'
  if (status === 'succeeded') return '已清理'
  if (status === 'pending' || status === 'running') return '清理中'
  return ''
}

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

function MetricComparisonCell({ value, baselineValue, baseline, baselineComplete, complete, best, tied, row, comparisonEnabled }: {
  value: number | null
  baselineValue: number | null
  baseline: boolean
  baselineComplete: boolean
  complete: boolean
  best: boolean
  tied: boolean
  row: SummaryMetric
  comparisonEnabled: boolean
}) {
  const delta = value !== null && baselineValue !== null ? Number((value - baselineValue).toFixed(4)) : null
  const ranked = comparisonEnabled && complete && baselineComplete && row.direction !== 'neutral'
  const improvement = delta === null || delta === 0 || !ranked ? 'neutral' : (row.direction === 'higher' ? delta > 0 : delta < 0) ? 'improved' : 'regressed'
  const Arrow = delta === null || delta === 0 ? Minus : delta > 0 ? ArrowUp : ArrowDown
  return <td className={best ? 'metric-best-cell' : ''}>
    <div className="metric-value-line"><strong>{value === null ? '暂无数据' : value.toFixed(4)}</strong>{best && <span className="metric-best-badge"><Trophy size={11} />{tied ? '并列最优' : '已完成中最优'}</span>}</div>
    {comparisonEnabled && !baseline && value !== null && <small className={`metric-delta ${improvement}`}>
      {delta === null ? '基准暂无数据' : <><Arrow size={12} />{delta === 0 ? '持平' : row.unit === 'points' ? `${Math.abs(delta * 100).toFixed(2)} 个百分点` : Math.abs(delta).toFixed(4)}{!complete ? <span>阶段值</span> : !baselineComplete ? <span>基准未完成</span> : delta !== 0 && ranked && <span>{improvement === 'improved' ? '改善' : '下降'}</span>}</>}
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
  const compareLimit = useCompareLimit()
  const [experiments, setExperiments] = useState<ComparisonRun[]>([])
  const [groups, setGroups] = useState<TaskGroup[]>([])
  const [metrics, setMetrics] = useState<Record<string, MetricPoint[]>>({})
  const [metricErrors, setMetricErrors] = useState<Record<string, string>>({})
  const [includeIncomplete, setIncludeIncomplete] = useState(false)
  const [detailRunId, setDetailRunId] = useState<string | null>(null)
  const [runDetails, setRunDetails] = useState<Record<string, CompareRunDetails>>({})
  const detailRequest = useRef<AbortController | null>(null)
  const requestVersion = useRef(0)
  const activeRequest = useRef<AbortController | null>(null)
  const catalog = useRef<ComparisonRun[]>([])
  const catalogLoadedAt = useRef(0)
  const metricCache = useRef(new Map<string, { revision: string; points: MetricPoint[] }>())
  const [metricKey, setMetricKey] = useState('metrics/mAP50(B)')
  const [baselineId, setBaselineId] = useState('')
  const [evaluationMode, setEvaluationMode] = useState<'latest' | 'best'>('latest')
  const [search, setSearch] = useState('')
  const [pickerOpen, setPickerOpen] = useState(false)
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(initialCollapsedGroups)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<ComparisonRun | null>(null)
  const [deleteError, setDeleteError] = useState('')
  const [deleteNotice, setDeleteNotice] = useState('')
  const [deletingRunId, setDeletingRunId] = useState<string | null>(null)
  const [params, setParams] = useSearchParams()
  const [selectedIds, setSelectedIds] = useState<string[]>(() => {
    const incoming = (params.get('runs') || params.get('tasks') || '').split(',').filter(Boolean)
    if (params.has('runs') || params.has('tasks')) return [...new Set(incoming)].slice(0, compareLimit)
    return readStoredCompareRuns().slice(0, compareLimit)
  })
  const lastSelectionUrl = useRef(params.toString())
  const selected = useMemo(() => selectedIds.map((id) => experiments.find((item) => item.id === id)).filter(Boolean) as ComparisonRun[], [experiments, selectedIds])

  useEffect(() => {
    if (lastSelectionUrl.current !== params.toString()) {
      lastSelectionUrl.current = params.toString()
      requestVersion.current++
      activeRequest.current?.abort()
      const incoming = (params.get('runs') || params.get('tasks') || '').split(',').filter(Boolean)
      const next = [...new Set(incoming)].slice(0, compareLimit)
      setSelectedIds(next)
      setBaselineId((current) => next.includes(current) ? current : next[0] || '')
      return
    }
    writeStoredCompareRuns(selectedIds)
    const next = new URLSearchParams(window.location.search)
    next.delete('tasks')
    if (selectedIds.length) next.set('runs', selectedIds.join(',')); else next.delete('runs')
    if (next.toString() !== params.toString()) {
      lastSelectionUrl.current = next.toString()
      setParams(next, { replace: true })
    }
  }, [compareLimit, selectedIds, params, setParams])

  const invalidateActiveRequest = useCallback(() => {
    requestVersion.current++
    activeRequest.current?.abort()
    activeRequest.current = null
  }, [])

  useEffect(() => {
    if (selectedIds.length <= compareLimit) return
    invalidateActiveRequest()
    const next = selectedIds.slice(0, compareLimit)
    setSelectedIds(next)
    setBaselineId((current) => next.includes(current) ? current : next[0] || '')
  }, [compareLimit, invalidateActiveRequest, selectedIds])

  const load = useCallback(async () => {
    const version = ++requestVersion.current
    activeRequest.current?.abort()
    const controller = new AbortController()
    activeRequest.current = controller
    try {
      const refreshCatalog = Date.now() - catalogLoadedAt.current >= 60_000
      let items = catalog.current
      let taskGroups: TaskGroup[] | undefined
      if (refreshCatalog) {
        const [firstPage, groups] = await Promise.all([api.experiments(0, controller.signal), api.taskGroups(controller.signal)])
        const rows = [...firstPage.items]
        for (let offset = firstPage.limit; offset < firstPage.total; offset += firstPage.limit) {
          rows.push(...(await api.experiments(offset, controller.signal)).items)
        }
        items = [...new Map(rows.map(({ task, run }) => [run.id, { id: run.id, task, run, view: runView(task, run) }])).values()]
        taskGroups = groups
      } else {
        const refreshIds = [...new Set([...selectedIds, ...items.filter((item) =>
          activeRunStatuses.has(item.run.status) || ['pending', 'running'].includes(item.run.deletion_status || '')
        ).map((item) => item.id)])]
        const refreshed = new Map<string, ComparisonRun>()
        for (let offset = 0; offset < refreshIds.length; offset += 200) {
          const page = await api.experiments(0, controller.signal, refreshIds.slice(offset, offset + 200))
          page.items.forEach(({ task, run }) => refreshed.set(run.id, { id: run.id, task, run, view: runView(task, run) }))
        }
        const requested = new Set(refreshIds)
        const known = new Set(items.map((item) => item.id))
        items = items.flatMap((item) => refreshed.has(item.id) ? [refreshed.get(item.id)!] : requested.has(item.id) ? [] : [item])
        items.push(...[...refreshed.values()].filter((item) => !known.has(item.id)))
      }
      if (version !== requestVersion.current) return
      catalog.current = items
      if (refreshCatalog) catalogLoadedAt.current = Date.now()
      setExperiments(items)
      if (taskGroups) setGroups(taskGroups)
      const normalizedIds = selectedIds.map((id) => {
        const exactRun = items.find((item) => item.id === id)
        if (exactRun) return exactRun.run.deletion_status ? undefined : exactRun.id
        const taskRuns = items.filter((item) => item.task.id === id && !item.run.deletion_status)
        return taskRuns.find((item) => item.id === item.task.current_run_id)?.id || taskRuns[0]?.id
      }).filter(Boolean) as string[]
      const limitedIds = normalizedIds.slice(0, compareLimit)
      if (limitedIds.join(',') !== selectedIds.join(',')) setSelectedIds(limitedIds)
      const deletingIds = new Set(items.filter((item) => item.run.deletion_status).map((item) => item.id))
      if (deletingIds.size) {
        setRunDetails((current) => Object.fromEntries(Object.entries(current).filter(([id]) => !deletingIds.has(id))))
        setMetrics((current) => Object.fromEntries(Object.entries(current).filter(([id]) => !deletingIds.has(id))))
        setMetricErrors((current) => Object.fromEntries(Object.entries(current).filter(([id]) => !deletingIds.has(id))))
        if (detailRunId && deletingIds.has(detailRunId)) {
          detailRequest.current?.abort()
          detailRequest.current = null
          setDetailRunId(null)
        }
      }
      const selectedItems = limitedIds.map((id) => items.find((item) => item.id === id)).filter(Boolean) as ComparisonRun[]
      const results = await Promise.allSettled(selectedItems.map(async (item) => {
        const revision = `${item.run.updated_at}:${item.run.completed_epochs}:${item.run.status}`
        const cached = metricCache.current.get(item.id)
        if (cached?.revision === revision) return cached.points
        const points = await api.runMetrics(item.task.id, item.run.id, controller.signal)
        if (version === requestVersion.current) metricCache.current.set(item.id, { revision, points })
        return points
      }))
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
  }, [compareLimit, detailRunId, selectedIds])

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    async function poll() {
      if (document.hidden) return
      await load()
      const active = catalog.current.some((item) => activeRunStatuses.has(item.run.status)
        || ['pending', 'running'].includes(item.run.deletion_status || ''))
      if (!cancelled && !document.hidden) timer = window.setTimeout(poll, active ? 5000 : 30_000)
    }
    function visibilityChanged() {
      window.clearTimeout(timer)
      if (document.hidden) invalidateActiveRequest()
      else void poll()
    }
    document.addEventListener('visibilitychange', visibilityChanged)
    void poll()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', visibilityChanged)
      invalidateActiveRequest()
    }
  }, [invalidateActiveRequest, load])

  useEffect(() => () => detailRequest.current?.abort(), [])

  useEffect(() => {
    try { window.localStorage.setItem(collapsedGroupsKey, JSON.stringify([...collapsedGroups])) } catch { /* Storage may be unavailable. */ }
  }, [collapsedGroups])

  function setSelection(ids: string[]) {
    invalidateActiveRequest()
    const next = [...new Set(ids)].slice(0, compareLimit)
    setSelectedIds(next)
    setBaselineId((current) => next.includes(current) ? current : next[0] || '')
  }

  function removeRunFromComparison(id: string) {
    invalidateActiveRequest()
    setSelectedIds((current) => current.filter((value) => value !== id))
    setBaselineId((current) => current === id ? '' : current)
    setMetrics((current) => {
      const next = { ...current }
      delete next[id]
      return next
    })
    setMetricErrors((current) => {
      const next = { ...current }
      delete next[id]
      return next
    })
    setRunDetails((current) => {
      const next = { ...current }
      delete next[id]
      return next
    })
  }

  function openDeleteRun(item: ComparisonRun) {
    if (deletingRunId) return
    if (activeRunStatuses.has(item.run.status)) {
      setError('训练进行中不能删除 Attempt，请先停止训练并等待状态稳定。')
      return
    }
    if (item.run.deletion_status === 'pending' || item.run.deletion_status === 'running') return
    setDeleteError('')
    setDeleteNotice('')
    setDeleteTarget(item)
  }

  async function confirmDeleteRun() {
    const target = deleteTarget
    if (!target || deletingRunId) return
    if (activeRunStatuses.has(target.run.status)) {
      setDeleteError('训练进行中不能删除 Attempt，请先停止训练并等待状态稳定。')
      return
    }
    setDeletingRunId(target.id)
    setDeleteError('')
    try {
      const result = await api.deleteRun(target.task.id, target.run.id)
      setExperiments((current) => current.map((item) => item.id === target.id
        ? { ...item, run: { ...item.run, deletion_status: result.status, deletion_error: null } }
        : item))
      removeRunFromComparison(target.id)
      if (detailRunId === target.id) closeRunDetails()
      setDeleteTarget(null)
      setDeleteNotice(result.message || 'Attempt 清理已提交，正在等待后台完成。')
      catalogLoadedAt.current = 0
      void load()
    } catch (reason) {
      setDeleteError(reason instanceof Error ? reason.message : '删除 Attempt 失败，请稍后重试')
    } finally {
      setDeletingRunId(null)
    }
  }

  function toggleTask(id: string) {
    const item = experiments.find((candidate) => candidate.id === id)
    if (item?.run.deletion_status) return
    if (selectedIds.includes(id)) setSelection(selectedIds.filter((value) => value !== id))
    else if (selectedIds.length < compareLimit) setSelection([...selectedIds, id])
  }

  function toggleGroup(key: string) {
    setCollapsedGroups((current) => {
      const next = new Set(current)
      if (next.has(key)) next.delete(key); else next.add(key)
      return next
    })
  }

  async function loadRunDetails(item: ComparisonRun, refresh = false) {
    detailRequest.current?.abort()
    detailRequest.current = null
    if (!refresh && runDetails[item.id]?.loaded) return
    const controller = new AbortController()
    detailRequest.current = controller
    setRunDetails((current) => ({ ...current, [item.id]: { ...current[item.id], loading: true, loaded: current[item.id]?.loaded || false, errors: [] } }))
    const results = await Promise.allSettled([
      api.runMetrics(item.task.id, item.run.id, controller.signal),
      api.runArtifacts(item.task.id, item.run.id, controller.signal),
      api.runEffectiveBestWeight(item.task.id, item.run.id, controller.signal),
      api.runEvents(item.task.id, item.run.id, controller.signal),
      api.runLogs(item.task.id, item.run.id, controller.signal),
    ])
    if (controller.signal.aborted) return
    const [metricsResult, artifactsResult, weightResult, eventsResult, logsResult] = results
    const errors = results.flatMap((result) => result.status === 'rejected'
      ? [result.reason instanceof Error ? result.reason.message : '部分运行详情加载失败'] : [])
    setRunDetails((current) => ({
      ...current,
      [item.id]: {
        loading: false,
        loaded: true,
        errors,
        ...(metricsResult.status === 'fulfilled' ? { metrics: metricsResult.value } : {}),
        ...(artifactsResult.status === 'fulfilled' ? { artifacts: artifactsResult.value } : {}),
        ...(weightResult.status === 'fulfilled' ? { bestWeight: weightResult.value } : {}),
        ...(eventsResult.status === 'fulfilled' ? { events: eventsResult.value } : {}),
        ...(logsResult.status === 'fulfilled' ? { logs: logsResult.value } : {}),
      },
    }))
    if (detailRequest.current === controller) detailRequest.current = null
  }

  function openRunDetails(item: ComparisonRun) {
    setDetailRunId(item.id)
    void loadRunDetails(item)
  }

  function closeRunDetails() {
    detailRequest.current?.abort()
    detailRequest.current = null
    setDetailRunId(null)
  }

  const differentDatasets = new Set(selected.map((item) => item.view.data_path)).size > 1
  const differentTaskTypes = new Set(selected.map((item) => item.view.task_type)).size > 1
  const comparisonEnabled = selected.length > 1 && !differentDatasets && !differentTaskTypes && selected.every((item) => !!item.view.data_path)
  const lossComparable = new Set(selected.map((item) => item.view.model_path)).size === 1 && new Set(selected.map((item) => item.view.imgsz)).size === 1
  const baselineRun = selected.find((item) => item.id === baselineId) || selected[0]
  const evaluationPoints = useMemo(() => Object.fromEntries(selected.map((item) => [item.id, bestEvaluationPoint(metrics[item.id])])), [selected, metrics])
  const latestRunValues = (item: ComparisonRun) => {
    const latestPoint = metrics[item.id]?.at(-1)
    if (latestPoint) return latestPoint.values
    return item.run.launch_mode === 'resume' ? {} : item.run.latest_metrics
  }
  const evaluationValues = (item: ComparisonRun) => {
    if (evaluationMode === 'best') return evaluationPoints[item.id]?.values
    return latestRunValues(item)
  }

  const chart = useMemo(() => ({
    animationDuration: 350,
    color: palette,
    tooltip: { trigger: 'axis', backgroundColor: '#14202a', borderColor: '#334754', textStyle: { color: '#e7eef2' } },
    legend: { type: selected.length > 4 ? 'scroll' : 'plain', data: selected.map((item) => `${item.task.name} · A${String(item.run.attempt_number).padStart(3, '0')}`), top: 5, left: 44, right: 44, textStyle: { color: '#6c8290', fontSize: 10 } },
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

  const filteredExperiments = experiments.filter((item) => {
    const keyword = search.trim().toLowerCase()
    const groupName = groups.find((group) => group.id === item.task.group_id)?.name.toLowerCase() || ''
    return item.task.name.toLowerCase().includes(keyword)
      || item.view.model_path.toLowerCase().includes(keyword)
      || groupName.includes(keyword)
      || `attempt ${item.run.attempt_number}`.includes(keyword)
      || `a${String(item.run.attempt_number).padStart(3, '0')}`.includes(keyword)
  })
  const incompleteTerminalStatuses = new Set(['failed', 'stopped', 'interrupted'])
  const hiddenIncompleteCount = filteredExperiments.filter((item) => incompleteTerminalStatuses.has(item.run.status) && !selectedIds.includes(item.id)).length
  const visibleExperiments = filteredExperiments.filter((item) => !incompleteTerminalStatuses.has(item.run.status)
    || includeIncomplete || selectedIds.includes(item.id) || Boolean(item.run.deletion_status))
  const groupedExperiments = useMemo(() => {
    const knownGroupIds = new Set(groups.map((group) => group.id))
    const views = groups.map((group) => ({
      key: group.id,
      name: group.name,
      items: visibleExperiments.filter((item) => item.task.group_id === group.id),
    })).filter((group) => group.items.length)
    const ungroupedItems = visibleExperiments.filter((item) => !item.task.group_id || !knownGroupIds.has(item.task.group_id))
    if (ungroupedItems.length) views.push({ key: ungroupedKey, name: '未分组', items: ungroupedItems })
    return views
  }, [groups, visibleExperiments])
  const hasIncompleteSelected = selected.some((item) => item.run.status !== 'completed')
  const detailItem = detailRunId ? experiments.find((item) => item.id === detailRunId) : undefined
  const experimentCount = Math.max(selected.length, 1)
  const experimentLayout = {
    '--experiment-count': experimentCount,
    '--experiment-column-width': `max(180px, calc(${100 / experimentCount}% - ${170 / experimentCount}px))`,
    '--parameter-table-min-width': `${170 + experimentCount * 200}px`,
  } as React.CSSProperties

  return <div className="page compare-page" style={experimentLayout}>
    <div className="compare-cleanup-actions"><AttemptDeletionHistory /></div>
    <header className="page-header compare-header"><div><span className="eyebrow">EXPERIMENT ANALYSIS</span><h1>实验对比</h1><p>并排比较任意历史 Attempt 的训练配置、收敛趋势和指标。</p></div><button className="button primary" onClick={() => setPickerOpen((value) => !value)}><Layers3 size={16} />选择运行 <span className="selection-count">{selected.length}/{compareLimit}</span><ChevronDown size={15} /></button></header>
    {pickerOpen && <div className="compare-picker">
      <header><div><strong>选择参与对比的训练运行</strong><span>已选择 {selectedIds.length} 个，最多同时选择 {compareLimit} 个；可选择每个任务的任意历史 Attempt</span></div><div className="picker-header-actions"><button className="picker-done" onClick={() => setPickerOpen(false)}>完成</button><button className="icon-button" onClick={() => setPickerOpen(false)}><X size={18} /></button></div></header>
      <div className="compare-picker-search"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索任务、模型或 Attempt" /></div>
      <label className="compare-incomplete-filter"><input type="checkbox" checked={includeIncomplete} onChange={(event) => setIncludeIncomplete(event.target.checked)} /><span>显示失败、已停止和已中断运行</span><small>当前筛选另有 {hiddenIncompleteCount} 个未选</small></label>
      <div className="compare-picker-list">{groupedExperiments.length ? groupedExperiments.map((group) => {
        const collapsed = collapsedGroups.has(group.key)
        return <section className={`compare-picker-group${collapsed ? ' is-collapsed' : ''}`} key={group.key}>
          <header>
            <button type="button" className="compare-picker-group-toggle" aria-expanded={!collapsed} aria-controls={`compare-picker-group-${group.key}`} onClick={() => toggleGroup(group.key)}>
              <ChevronDown size={15} /><Folder size={15} /><strong>{group.name}</strong><small>{group.items.length} 个运行</small>
            </button>
          </header>
          {!collapsed && <div id={`compare-picker-group-${group.key}`} className="compare-picker-group-items">{group.items.map((item) => {
            const checked = selectedIds.includes(item.id)
            const deletionStatus = item.run.deletion_status
            const isDeleting = deletionStatus === 'pending' || deletionStatus === 'running'
            const active = activeRunStatuses.has(item.run.status)
            const deleteDisabled = active || isDeleting || deletingRunId === item.id
            const deleteTitle = active ? '训练进行中，停止后才能删除 Attempt' : isDeleting ? 'Attempt 文件正在清理' : deletionStatus === 'failed' ? '清理失败，点击重试' : '删除该 Attempt 的训练文件'
            return <div className="compare-run-option" key={item.id}>
              <div className="compare-run-option-row">
                <button className="compare-run-select" onClick={() => toggleTask(item.id)} disabled={Boolean(deletionStatus) || (!checked && selectedIds.length >= compareLimit)} title={deletionStatus ? `${deletionStatusLabel(deletionStatus)}，不可加入对比` : !checked && selectedIds.length >= compareLimit ? `最多同时选择 ${compareLimit} 个运行` : undefined}>
                  <i className={checked ? 'checked' : ''}>{checked && <Check size={12} />}</i>
                  <span><strong>{item.task.name} · Attempt {String(item.run.attempt_number).padStart(3, '0')}</strong><small>{item.view.model_path.split('/').pop()} · Epoch {item.view.completed_epochs}/{item.view.epochs}</small>{deletionStatus && <small className={`compare-run-deletion-state ${deletionStatus === 'failed' ? 'failed' : ''}`} role={deletionStatus === 'failed' ? 'alert' : 'status'}>{deletionStatus === 'failed' ? (item.run.deletion_error || '清理失败，可重试') : `${deletionStatusLabel(deletionStatus)}，已从对比中移除`}</small>}</span>
                  <em className={`status ${deletionStatus ? `deletion-${deletionStatus}` : item.run.status}`}>{deletionStatus ? deletionStatusLabel(deletionStatus) : statusLabel(item.run.status)}</em>
                </button>
                <button type="button" className="compare-run-delete" disabled={deleteDisabled} aria-label={`${deleteTitle}：${item.task.name} Attempt ${item.run.attempt_number}`} title={deleteTitle} onClick={() => openDeleteRun(item)}>{deletingRunId === item.id ? <LoaderCircle size={14} className="spin" /> : <Trash2 size={14} />}</button>
                <button className="compare-run-details-toggle" aria-haspopup="dialog" aria-expanded={detailRunId === item.id} aria-label={`查看 ${item.task.name} Attempt ${item.run.attempt_number} 详情`} onClick={() => openRunDetails(item)} title="打开运行详情"><Info size={14} /><span>{item.run.status === 'interrupted' ? '中断详情' : item.run.status === 'failed' ? '失败详情' : item.run.status === 'stopped' ? '停止详情' : '查看详情'}</span></button>
              </div>
            </div>
          })}</div>}
        </section>
      }) : <p className="compare-picker-no-results">没有匹配的训练运行。</p>}</div>
    </div>}
    {detailItem && <CompareRunDetailModal task={detailItem.task} run={detailItem.run} view={detailItem.view} details={runDetails[detailItem.id]} onClose={closeRunDetails} onRefresh={() => void loadRunDetails(detailItem, true)} />}
    {deleteTarget && <div className="modal-backdrop compare-run-delete-backdrop" onMouseDown={() => !deletingRunId && setDeleteTarget(null)}>
      <div className="modal compare-run-delete-modal" role="dialog" aria-modal="true" aria-labelledby="compare-run-delete-title" onMouseDown={(event) => event.stopPropagation()}>
        <header><div className="compare-run-delete-heading"><Trash2 size={19} /><div><span className="eyebrow">DELETE TRAINING ATTEMPT</span><h2 id="compare-run-delete-title">删除训练 Attempt</h2><p>{deleteTarget.task.name} · Attempt {String(deleteTarget.run.attempt_number).padStart(3, '0')}</p></div></div><button type="button" className="icon-button" title="关闭" aria-label="关闭删除 Attempt 弹窗" disabled={Boolean(deletingRunId)} onClick={() => setDeleteTarget(null)}><X size={18} /></button></header>
        <div className="compare-run-delete-body"><p>此操作会清理该 Attempt 的训练目录、独立日志、权重、指标图表和 Trace 文件。</p><ul><li>保留任务配置、其他 Attempt、基础模型和数据集。</li><li>独立保存的模型检测历史不会删除。</li><li>{deleteTarget.task.execution_node_id ? '远程任务会等待 Agent 确认清理；失败时保留记录并可重试。' : '本地任务会在后台清理；失败时保留记录并可重试。'}</li></ul>{deleteError && <div className="alert error" role="alert">{deleteError}</div>}</div>
        <footer><span>删除后该 Attempt 将从实验对比中移除。</span><button type="button" className="button ghost" disabled={Boolean(deletingRunId)} onClick={() => setDeleteTarget(null)}>取消</button><button type="button" className="button danger" disabled={Boolean(deletingRunId)} onClick={() => void confirmDeleteRun()}>{deletingRunId ? <><LoaderCircle size={14} className="spin" />提交中…</> : <><Trash2 size={14} />{deleteTarget.run.deletion_status === 'failed' ? '重试清理' : '确认删除 Attempt'}</>}</button></footer>
      </div>
    </div>}
    {deleteNotice && <div className="compare-notice compare-delete-success" role="status"><Check size={17} /><div><strong>Attempt 清理已提交</strong><span>{deleteNotice}</span></div></div>}
    {error && <div className="alert error">{error}</div>}
    {loading ? <div className="compare-empty">正在载入实验数据…</div> : !selected.length ? <div className="compare-empty"><Activity size={34} /><strong>选择要比较的训练运行</strong><span>{compareLimit === 2 ? '请选择 2 个运行' : `可同时选择 2～${compareLimit} 个运行`}，查看参数差异和指标曲线。</span><button className="button primary" onClick={() => setPickerOpen(true)}>选择训练运行</button></div> : <>
      {selected.length < 2 && <div className="compare-notice"><AlertCircle size={17} /><div><strong>至少需要两个训练运行才能对比</strong><span>当前只有一个运行，暂时无法生成实验对比。</span></div><button className="button ghost" onClick={() => setPickerOpen(true)}>继续选择</button></div>}
      <section className={`selected-experiments${selected.length > 4 ? ' many' : ''}`}>{selected.map((item, index) => <article key={item.id} style={{ '--experiment-color': palette[index] } as React.CSSProperties}><button onClick={() => setSelection(selectedIds.filter((id) => id !== item.id))}><X size={14} /></button><span>ATTEMPT {String(item.run.attempt_number).padStart(3, '0')} · {item.run.launch_mode.toUpperCase()}</span><h3>{item.task.name}</h3><p>{item.view.model_path.split('/').pop()} · {item.view.task_type}</p><div className="selected-run-status"><span className={`status ${item.run.status}`}>{statusLabel(item.run.status)}</span>{item.run.status !== 'completed' && <small>阶段性数据</small>}</div><div className="selected-run-progress"><i style={{ width: `${item.view.progress}%` }} /></div><footer><strong>{metricValue({ ...item.view, latest_metrics: latestRunValues(item) }, ['metrics/mAP50(B)'])}</strong><small>mAP50</small><strong>{Math.round(item.view.progress)}%</strong><small>进度</small></footer></article>)}</section>
      {selected.length >= 2 && <>
      <section className="compare-panel metric-summary-panel">
        <header><div><Activity size={17} /><div><h2>实验指标对比</h2><span>每列对应一个已选 Attempt，颜色与下方曲线一致</span></div></div><div className="metric-summary-controls"><label>指标口径<select aria-label="指标口径" value={evaluationMode} onChange={(event) => setEvaluationMode(event.target.value as 'latest' | 'best')}><option value="latest">最新指标</option><option value="best">最佳轮次指标</option></select></label><span className="metric-run-count">{selected.length} 个运行</span></div></header>
        <div className="metric-summary-notes">
          <p>{evaluationMode === 'latest' ? '最新指标截至各实验已记录的训练轮次。' : '最佳轮次按各实验历史最高 mAP50-95 选取，以下六项指标均来自该轮；缺少 mAP50-95 时显示暂无数据。'} mAP50-95 为主要效果参考；训练 Loss 和累计耗时不参与效果排名。</p>
          {hasIncompleteSelected && <p className="dataset-comparison-note"><AlertCircle size={14} />排队、训练中、停止中、失败、停止和中断的运行只展示已记录阶段指标，不参与“最优”排名；差值标为阶段参考。续训 Attempt 尚无本次指标时不会沿用上次 Attempt 的指标。</p>}
          {comparisonEnabled && <><p>差值相对于基准实验；同数据集配置仍需确认验证集版本、类别与评估条件一致。颜色表示数值差异，小幅变化不代表显著改善。</p><div className="metric-comparison-legend"><span className="improved"><ArrowUp size={12} />改善</span><span className="regressed"><ArrowDown size={12} />下降</span><span><Minus size={12} />持平 / 仅展示差值</span><span><Trophy size={12} />{hasIncompleteSelected ? '已完成运行中最优' : '本行有效数据最优'}</span></div></>}
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
              const completedValues = selected.map((item, index) => item.run.status === 'completed' ? values[index] : null)
              const valid = completedValues.filter((value): value is number => value !== null)
              const direction = row.direction === 'lower' && !lossComparable ? 'neutral' : row.direction
              const best = comparisonEnabled && direction !== 'neutral' && valid.length > 1 ? (direction === 'higher' ? Math.max(...valid) : Math.min(...valid)) : null
              const tied = best !== null && valid.filter((value) => value === best).length > 1
              const baselineValue = baselineRun ? displayedMetric(evaluationValues(baselineRun), row.key) : null
              const label = `${evaluationMode === 'latest' ? '最新' : '最佳轮次'}${row.unit === 'points' ? ' ' : ''}${row.label}`
              return <tr key={row.key}><th scope="row"><span>{label}</span><small className="metric-direction">{row.key === 'metrics/mAP50-95(B)' ? '主指标 · ' : ''}{direction === 'higher' ? '↑ 越高越好' : direction === 'lower' ? '↓ 越低越好 · 辅助参考' : '仅展示差值'}{hasIncompleteSelected ? ' · 最优仅比较已完成运行' : ''}</small></th>{selected.map((item, index) => <MetricComparisonCell key={item.id} value={values[index]} baselineValue={baselineValue} baseline={item.id === baselineRun?.id} baselineComplete={baselineRun?.run.status === 'completed'} complete={item.run.status === 'completed'} best={best !== null && item.run.status === 'completed' && values[index] === best} tied={tied} row={{ ...row, direction }} comparisonEnabled={comparisonEnabled} />)}</tr>
            })}
            <tr><th scope="row">历史最佳 mAP50</th>{selected.map((item) => <td key={item.id}>{metricErrors[item.id] ? '加载失败' : metrics[item.id] === undefined ? '加载中…' : bestMapLabel(metrics[item.id])}</td>)}</tr>
            <tr><th scope="row">累计训练耗时</th>{selected.map((item) => <td key={item.id}>{metricErrors[item.id] ? '加载失败' : metrics[item.id] === undefined ? '加载中…' : elapsedLabel(metrics[item.id]?.at(-1)?.elapsed)}</td>)}</tr>
            <tr className="metric-dataset-row"><th scope="row">数据集配置</th>{selected.map((item) => <td key={item.id} title={item.view.data_path}>{item.view.data_path}</td>)}</tr>
          </tbody>
        </table></div>
      </section>
      <section className="compare-panel chart-panel"><header><div><BarChart3 size={17} /><div><h2>训练指标趋势</h2><span>按 Epoch 对齐叠加曲线</span></div></div><select value={metricKey} onChange={(event) => setMetricKey(event.target.value)}>{metricOptions.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}</select></header><LineChartView option={chart} style={{ height: 390 }} /></section>
      <section className="compare-panel parameter-panel"><header><div><Layers3 size={17} /><div><h2>训练参数差异</h2><span>来自每次运行的固化参数快照</span></div></div></header><div className="comparison-table"><table style={{ minWidth: 170 + selected.length * 200 }}><thead><tr><th>参数</th>{selected.map((item, index) => <th key={item.id}><i style={{ background: palette[index] }} />{item.task.name} · A{String(item.run.attempt_number).padStart(3, '0')}</th>)}</tr></thead><tbody>{parameterRows.map((parameter) => { const values = selected.map((item) => parameter.value(item.view)); const different = new Set(values.map(String)).size > 1; return <tr className={different ? 'different' : ''} key={parameter.label}><td>{parameter.label}{different && <em>不同</em>}</td>{values.map((value, index) => <td key={`${parameter.label}-${selected[index].id}`}>{value}</td>)}</tr> })}</tbody></table></div></section>
      </>}
    </>}
  </div>
}
