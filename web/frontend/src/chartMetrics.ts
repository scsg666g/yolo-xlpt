import type { MetricPoint } from './types'

/** Return only finite numeric metric values; nulls are gaps, not data points. */
export function finiteMetricValue(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function metricPointCount(metrics: MetricPoint[], key: string): number {
  return metrics.reduce((count, point) => count + (finiteMetricValue(point.values[key]) ? 1 : 0), 0)
}

/** Keep a single validation result visible when there is no line segment yet. */
export function metricSeriesDisplay(values: readonly unknown[]) {
  const valid = values.map(finiteMetricValue)
  const count = valid.reduce<number>((total, value) => total + (value ? 1 : 0), 0)
  const hasIsolatedPoint = valid.some((value, index) => value && !valid[index - 1] && !valid[index + 1])
  return {
    showSymbol: count > 0 && (count <= 2 || hasIsolatedPoint),
    showAllSymbol: hasIsolatedPoint,
    symbol: 'circle',
    symbolSize: count === 1 ? 8 : 5,
    connectNulls: false,
  }
}

export function hasMetricValue(metrics: MetricPoint[], keys: string[]): boolean {
  return keys.some((key) => metricPointCount(metrics, key) > 0)
}

export function hasAnyMetricValue(metrics: MetricPoint[]): boolean {
  return metrics.some((point) => Object.values(point.values).some(finiteMetricValue))
}
