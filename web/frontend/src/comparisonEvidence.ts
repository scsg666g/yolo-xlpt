import { datasetTrainingSnapshot } from './datasetSnapshot'

interface ComparisonEvidenceRun {
  taskType: string
  nodeId: string | null
  dataPath: string
  parameters: Record<string, unknown>
}

/** Only historical launch evidence is considered. Today's catalogue is never consulted. */
export function comparisonEvidence(runs: readonly ComparisonEvidenceRun[]) {
  if (runs.length < 2) return { comparable: false, reason: '选择至少两个运行后核验可比性。' }
  if (new Set(runs.map((run) => run.taskType)).size > 1) return { comparable: false, reason: '任务类型不同，仅展示指标，不进行优劣排名。' }
  const snapshots = runs.map((run) => datasetTrainingSnapshot(run.parameters))
  if (snapshots.some((snapshot, index) => !snapshot || snapshot.configuration_scope !== 'launch_configuration'
    || snapshot.verification !== 'catalog_snapshot' || !snapshot.fingerprint || !snapshot.config_sha256
    || snapshot.node_id !== (runs[index].nodeId || 'local') || snapshot.task_type !== runs[index].taskType
    || !snapshot.checked_at || !snapshot.classes.length || !runs[index].dataPath.trim())) {
    return { comparable: false, reason: '可比性未核验：缺少完整的历史数据集校验证据，或本次为续训请求。仅展示数值与曲线，不标记改善、最优或基准差值。' }
  }
  const identities = runs.map((run, index) => {
    const snapshot = snapshots[index]!
    return JSON.stringify([run.taskType, snapshot.node_id, run.dataPath, snapshot.config_sha256,
      snapshot.fingerprint, snapshot.classes, Object.entries(snapshot.stats).sort(([left], [right]) => left.localeCompare(right))])
  })
  if (new Set(identities).size > 1) return { comparable: false, reason: '服务器、数据集配置或历史校验证据不同，仅展示指标，不进行优劣排名。' }
  return { comparable: true, reason: '历史配置与校验证据匹配；数据文件仍可能在校验后改变，请确认验证集版本、类别及评估条件一致。' }
}
