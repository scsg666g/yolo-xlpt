import { metricProfile, primaryMetric } from './metricProfiles'
import type { TrainingTask } from './types'

export function taskMetric(task: TrainingTask, key: string): number | null {
  const raw = task.latest_metrics?.[key]
  if (typeof raw !== 'number' && typeof raw !== 'string') return null
  if (typeof raw === 'string' && !raw.trim()) return null
  const value = Number(raw)
  return Number.isFinite(value) && value >= 0 && value <= 1 ? value : null
}

/** Rank each task's latest epoch by mAP50, then read mAP50-95 from that same task. */
export function groupMetricSummary(tasks: TrainingTask[]) {
  const candidates = tasks.flatMap((task) => {
    const profile = metricProfile(task.task_type)
    const metric = profile?.scoreSeries[0]
    const value = metric ? taskMetric(task, metric.key) : null
    return metric && value !== null ? [{ task, key: metric.key, value }] : []
  })
  const highest = candidates.reduce<(typeof candidates)[number] | null>(
    (current, candidate) => !current || candidate.value > current.value ? candidate : current,
    null,
  )
  // Equal dataset paths on different nodes do not establish comparable sources.
  const sources = new Set(candidates.map(({ task, key }) => JSON.stringify([
    task.execution_node_id || 'local', task.data_path, task.task_type, key,
  ])))
  const mixedMetrics = new Set(candidates.map(({ task, key }) => `${task.task_type}:${key}`)).size > 1
  const companion = highest && highest.task.task_type !== 'classify'
    ? primaryMetric(highest.task.task_type) : undefined
  return {
    highest: mixedMetrics ? null : highest,
    label: mixedMetrics ? '多种任务指标' : metricProfile(highest?.task.task_type)?.scoreSeries[0]?.name || '主指标',
    companion: mixedMetrics || !highest || !companion ? null : {
      label: companion.name, value: taskMetric(highest.task, companion.key),
    },
    mixedMetrics,
    count: candidates.length,
    mixedSources: sources.size > 1 || candidates.some(({ task }) => !task.data_path?.trim()),
  }
}
