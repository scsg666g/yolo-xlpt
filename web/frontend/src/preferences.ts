import { useSyncExternalStore } from 'react'

export const COMPARE_LIMIT_MIN = 2
export const COMPARE_LIMIT_MAX = 8
export const COMPARE_LIMIT_DEFAULT = 4
export const COMPARE_LIMIT_STORAGE_KEY = 'yolo-forge-compare-limit'
export const COMPARE_RUNS_STORAGE_KEY = 'yolo-forge-compare-runs'

const COMPARE_LIMIT_EVENT = 'yolo-forge-compare-limit-change'

export function normalizeCompareLimit(value: unknown): number {
  const parsed = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(parsed)) return COMPARE_LIMIT_DEFAULT
  return Math.min(COMPARE_LIMIT_MAX, Math.max(COMPARE_LIMIT_MIN, Math.round(parsed)))
}

export function readCompareLimit(): number {
  try {
    const stored = localStorage.getItem(COMPARE_LIMIT_STORAGE_KEY)
    return stored === null ? COMPARE_LIMIT_DEFAULT : normalizeCompareLimit(stored)
  } catch {
    return COMPARE_LIMIT_DEFAULT
  }
}

export function readStoredCompareRuns(): string[] {
  try {
    const stored: unknown = JSON.parse(
      localStorage.getItem(COMPARE_RUNS_STORAGE_KEY)
        || localStorage.getItem('yolo-forge-compare-tasks')
        || '[]',
    )
    return Array.isArray(stored)
      ? [...new Set(stored.filter((id): id is string => typeof id === 'string' && !!id))]
      : []
  } catch {
    return []
  }
}

export function writeCompareLimit(value: number): number {
  const limit = normalizeCompareLimit(value)
  localStorage.setItem(COMPARE_LIMIT_STORAGE_KEY, String(limit))
  window.dispatchEvent(new Event(COMPARE_LIMIT_EVENT))
  return limit
}

export function writeStoredCompareRuns(ids: string[]): void {
  localStorage.setItem(COMPARE_RUNS_STORAGE_KEY, JSON.stringify([...new Set(ids)]))
}

function subscribeCompareLimit(listener: () => void) {
  const onStorage = (event: StorageEvent) => {
    if (event.key === COMPARE_LIMIT_STORAGE_KEY) listener()
  }
  window.addEventListener(COMPARE_LIMIT_EVENT, listener)
  window.addEventListener('storage', onStorage)
  return () => {
    window.removeEventListener(COMPARE_LIMIT_EVENT, listener)
    window.removeEventListener('storage', onStorage)
  }
}

export function useCompareLimit(): number {
  return useSyncExternalStore(subscribeCompareLimit, readCompareLimit, () => COMPARE_LIMIT_DEFAULT)
}
