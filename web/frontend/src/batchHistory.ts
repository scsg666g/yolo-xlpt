import type { TrainingRun, TrainingTask } from './types'

export function validBatchHistory(history: number[] | null | undefined): number[] {
  return Array.isArray(history)
    ? history.filter((value) => Number.isSafeInteger(value) && value > 0)
    : []
}

export function batchHistoryForRun(task: TrainingTask, run: TrainingRun | null): number[] {
  const taskHistory = validBatchHistory(task.batch_history)
  if (!run) return taskHistory
  const runHistory = validBatchHistory(run.batch_history)
  const requestedBatch = Number(run.parameter_snapshot?.batch)
  // Task and Run requests can race during live polling. Only the active Run
  // may use the newer task value, and its starting Batch must still match.
  // Historical Runs must stay isolated from any later training attempt.
  if (run.id === task.current_run_id && taskHistory.length > runHistory.length
    && (!Number.isSafeInteger(requestedBatch) || requestedBatch <= 0 || taskHistory[0] === requestedBatch)) return taskHistory
  return runHistory
}

export function batchPath(configuredBatch: number, history: number[] | null | undefined) {
  const values = validBatchHistory(history)
  const adjusted = values.length > 1
  const path = values.join(' → ')
  const submittedBatch = values[0] ?? configuredBatch
  return {
    adjusted,
    submittedBatch,
    cardLabel: adjusted ? `Batch ${path}` : `设置 Batch ${submittedBatch}`,
    detailLabel: adjusted ? path : '未记录自动调整',
    title: adjusted
      ? `本次 Attempt 设置 Batch ${values[0]}，自动调整路径：${path}；最新尝试值 ${values.at(-1)}`
      : `设置 Batch ${submittedBatch}；未记录本次 Attempt 的自动调整`,
  }
}
