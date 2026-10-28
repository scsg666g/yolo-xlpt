import type { MetricPoint } from './types'

export type MetricSeriesDefinition = { key: string; name: string; color: string }
export type MetricProfile = {
  scoreTitle: string
  scoreSeries: MetricSeriesDefinition[]
  lossSeries: MetricSeriesDefinition[]
  summaryKeys: string[]
}

const scoreColors = ['#70f0bd', '#6da8ff', '#ffd56d', '#d99cff']
const lossColors = ['#70f0bd', '#53d6e8', '#ff9d76', '#ffd56d']
export const metricProfiles: Record<string, MetricProfile> = {
  detect: {
    scoreTitle: '精度与召回趋势',
    scoreSeries: [
      { key: 'metrics/mAP50(B)', name: 'mAP50', color: scoreColors[0] },
      { key: 'metrics/mAP50-95(B)', name: 'mAP50-95', color: scoreColors[1] },
      { key: 'metrics/precision(B)', name: 'Precision', color: scoreColors[2] },
      { key: 'metrics/recall(B)', name: 'Recall', color: scoreColors[3] },
    ],
    lossSeries: [
      { key: 'train/box_loss', name: 'Train box', color: lossColors[0] },
      { key: 'train/cls_loss', name: 'Train cls', color: lossColors[1] },
      { key: 'val/box_loss', name: 'Val box', color: lossColors[2] },
      { key: 'val/cls_loss', name: 'Val cls', color: lossColors[3] },
    ],
    summaryKeys: ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)', 'train/box_loss', 'val/box_loss'],
  },
  segment: {
    scoreTitle: '分割精度与召回趋势',
    scoreSeries: [
      { key: 'metrics/mAP50(M)', name: 'Mask mAP50', color: scoreColors[0] },
      { key: 'metrics/mAP50-95(M)', name: 'Mask mAP50-95', color: scoreColors[1] },
      { key: 'metrics/precision(M)', name: 'Mask precision', color: scoreColors[2] },
      { key: 'metrics/recall(M)', name: 'Mask recall', color: scoreColors[3] },
    ],
    lossSeries: [
      { key: 'train/seg_loss', name: 'Train seg', color: lossColors[0] },
      { key: 'train/box_loss', name: 'Train box', color: lossColors[1] },
      { key: 'val/seg_loss', name: 'Val seg', color: lossColors[2] },
      { key: 'val/box_loss', name: 'Val box', color: lossColors[3] },
    ],
    summaryKeys: ['metrics/mAP50(M)', 'metrics/mAP50-95(M)', 'metrics/precision(M)', 'metrics/recall(M)', 'train/seg_loss', 'val/seg_loss'],
  },
  classify: {
    scoreTitle: '分类准确率趋势',
    scoreSeries: [
      { key: 'metrics/accuracy_top1', name: 'Top-1 accuracy', color: scoreColors[0] },
      { key: 'metrics/accuracy_top5', name: 'Top-5 accuracy', color: scoreColors[1] },
    ],
    lossSeries: [
      { key: 'train/loss', name: 'Train loss', color: lossColors[0] },
      { key: 'val/loss', name: 'Val loss', color: lossColors[2] },
    ],
    summaryKeys: ['metrics/accuracy_top1', 'metrics/accuracy_top5', 'train/loss', 'val/loss'],
  },
  pose: {
    scoreTitle: '姿态精度与召回趋势',
    scoreSeries: [
      { key: 'metrics/mAP50(P)', name: 'Pose mAP50', color: scoreColors[0] },
      { key: 'metrics/mAP50-95(P)', name: 'Pose mAP50-95', color: scoreColors[1] },
      { key: 'metrics/precision(P)', name: 'Pose precision', color: scoreColors[2] },
      { key: 'metrics/recall(P)', name: 'Pose recall', color: scoreColors[3] },
    ],
    lossSeries: [
      { key: 'train/pose_loss', name: 'Train pose', color: lossColors[0] },
      { key: 'train/kobj_loss', name: 'Train kobj', color: lossColors[1] },
      { key: 'val/pose_loss', name: 'Val pose', color: lossColors[2] },
      { key: 'val/kobj_loss', name: 'Val kobj', color: lossColors[3] },
    ],
    summaryKeys: ['metrics/mAP50(P)', 'metrics/mAP50-95(P)', 'metrics/precision(P)', 'metrics/recall(P)', 'train/pose_loss', 'val/pose_loss'],
  },
  obb: {
    scoreTitle: '旋转框精度与召回趋势',
    scoreSeries: [
      { key: 'metrics/mAP50(B)', name: 'OBB mAP50', color: scoreColors[0] },
      { key: 'metrics/mAP50-95(B)', name: 'OBB mAP50-95', color: scoreColors[1] },
      { key: 'metrics/precision(B)', name: 'OBB precision', color: scoreColors[2] },
      { key: 'metrics/recall(B)', name: 'OBB recall', color: scoreColors[3] },
    ],
    lossSeries: [
      { key: 'train/box_loss', name: 'Train box', color: lossColors[0] },
      { key: 'train/angle_loss', name: 'Train angle', color: lossColors[1] },
      { key: 'val/box_loss', name: 'Val box', color: lossColors[2] },
      { key: 'val/angle_loss', name: 'Val angle', color: lossColors[3] },
    ],
    summaryKeys: ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)', 'train/angle_loss', 'val/angle_loss'],
  },
}


/** Unknown future task types keep raw metrics in detail views. */
export function metricProfile(taskType: string | null | undefined): MetricProfile | undefined {
  return taskType && Object.hasOwn(metricProfiles, taskType) ? metricProfiles[taskType] : undefined
}

export function primaryMetric(taskType: string | null | undefined): MetricSeriesDefinition | undefined {
  const profile = metricProfile(taskType)
  return taskType === 'classify' ? profile?.scoreSeries[0] : profile?.scoreSeries[1]
}

export function bestMetricPoint(points: readonly MetricPoint[] | undefined, taskType: string | null | undefined): MetricPoint | undefined {
  const key = primaryMetric(taskType)?.key
  if (!key) return undefined
  let best: MetricPoint | undefined
  for (const point of points || []) {
    const value = point.values[key]
    if (typeof value === 'number' && Number.isFinite(value)
      && (!best || value > Number(best.values[key]))) best = point
  }
  return best
}
