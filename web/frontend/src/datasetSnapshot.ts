export interface DatasetTrainingSnapshot {
  version: 1
  node_id: string
  task_type: string
  source: 'managed_catalog' | 'runtime_config'
  captured_at: string | null
  verification: 'catalog_snapshot' | 'config_only' | 'path_only' | 'unavailable'
  configuration_scope: 'launch_configuration' | 'resume_request' | null
  config_sha256: string | null
  fingerprint: string | null
  checked_at: string | null
  classes: string[]
  stats: Record<string, number>
}

/** Old attempts intentionally stay unknown; never fill them from today's catalogue. */
export function datasetTrainingSnapshot(parameters?: Record<string, unknown>): DatasetTrainingSnapshot | null {
  const value = parameters?.dataset_snapshot
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const item = value as Record<string, unknown>
  if (item.version !== 1 || !['managed_catalog', 'runtime_config'].includes(String(item.source))
    || !['catalog_snapshot', 'config_only', 'path_only', 'unavailable'].includes(String(item.verification))) return null
  const text = (key: string) => typeof item[key] === 'string' ? item[key] as string : null
  const digest = (key: string) => { const value = text(key); return value && /^[0-9a-f]{64}$/i.test(value) ? value : null }
  const rawStats = item.stats && typeof item.stats === 'object' && !Array.isArray(item.stats) ? item.stats : {}
  const stats = Object.fromEntries(Object.entries(rawStats).filter((entry): entry is [string, number] => typeof entry[1] === 'number' && Number.isFinite(entry[1]) && entry[1] >= 0))
  return {
    version: 1, node_id: text('node_id') || '未记录', task_type: text('task_type') || '未记录',
    source: item.source as DatasetTrainingSnapshot['source'], captured_at: text('captured_at'),
    verification: item.verification as DatasetTrainingSnapshot['verification'],
    configuration_scope: item.configuration_scope === 'launch_configuration' || item.configuration_scope === 'resume_request'
      ? item.configuration_scope : null,
    config_sha256: digest('config_sha256'), fingerprint: digest('fingerprint'), checked_at: text('checked_at'),
    classes: Array.isArray(item.classes) ? item.classes.filter((value): value is string => typeof value === 'string') : [], stats,
  }
}
