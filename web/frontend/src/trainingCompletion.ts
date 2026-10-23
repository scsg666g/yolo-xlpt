import { statusLabel } from './api'
import type { TrainingCompletionInfo, TrainingTask } from './types'

type TrainingStatus = { status: string; completion_info?: TrainingCompletionInfo | null }

function positiveNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

export function trainingStatusLabel(training: TrainingStatus): string {
  return training.status === 'completed' && training.completion_info?.reason === 'early_stopping'
    ? '已完成 · 提前停止'
    : statusLabel(training.status)
}

export function earlyStoppingExplanation(training: TrainingStatus): string | null {
  if (training.status !== 'completed' || training.completion_info?.reason !== 'early_stopping') return null
  const { best_epoch: bestEpoch, patience } = training.completion_info
  const details = [
    positiveNumber(patience) ? `连续 ${patience} 轮未改善，提前停止` : '达到早停条件，训练已正常结束',
    positiveNumber(bestEpoch) ? `最佳结果在第 ${bestEpoch} 轮` : null,
  ]
  return details.filter(Boolean).join('；')
}

export function trainingEpochTarget(training: Pick<TrainingTask, 'epochs' | 'completion_info'>): number {
  const expectedEpochs = training.completion_info?.expected_epochs
  return positiveNumber(expectedEpochs) ? expectedEpochs : training.epochs
}

export function trainingEpochProgress(training: Pick<TrainingTask, 'epochs' | 'completed_epochs' | 'completion_info'>): number {
  return Math.min(100, Math.max(0, training.completed_epochs / Math.max(1, trainingEpochTarget(training)) * 100))
}
