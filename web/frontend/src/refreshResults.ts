export function settledValue<T>(result: PromiseSettledResult<T>, current: T): T {
  return result.status === 'fulfilled' ? result.value : current
}

export function settledRunValue<T>(result: PromiseSettledResult<T>, current: T, sameRun: boolean, empty: T): T {
  return settledValue(result, sameRun ? current : empty)
}

export function refreshFailures(entries: Array<[string, PromiseSettledResult<unknown>]>): string[] {
  return entries.flatMap(([label, result]) => {
    if (result.status !== 'rejected') return []
    const reason = result.reason
    const message = reason instanceof Error ? reason.message
      : typeof reason === 'object' && reason !== null && 'message' in reason && typeof reason.message === 'string'
        ? reason.message : '加载失败'
    return [`${label}：${message}`]
  })
}
