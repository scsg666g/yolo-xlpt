import { useSyncExternalStore } from 'react'

export const COMPARE_LIMIT_MIN = 2
export const COMPARE_LIMIT_MAX = 8
export const COMPARE_LIMIT_DEFAULT = 4
export const COMPARE_LIMIT_STORAGE_KEY = 'yolo-forge-compare-limit'
export const COMPARE_RUNS_STORAGE_KEY = 'yolo-forge-compare-runs'
export const TASK_CENTER_PREFERENCES_STORAGE_KEY = 'yolo-forge-task-center-v1'

export type TaskCenterPreferences = {
  layoutMode: 'free' | 'grid'
  cardDensity: 'comfortable' | 'compact'
  defaultZoom: number
  showMiniMap: boolean
  showConnections: boolean
}

export type TaskCenterPreferencesSnapshot = TaskCenterPreferences & { storageAvailable: boolean }

export const TASK_CENTER_PREFERENCES_DEFAULTS: Readonly<TaskCenterPreferences> = Object.freeze({
  layoutMode: 'free',
  cardDensity: 'comfortable',
  defaultZoom: 0.9,
  showMiniMap: true,
  showConnections: true,
})

const COMPARE_LIMIT_EVENT = 'yolo-forge-compare-limit-change'
const LEGACY_TASK_CENTER_STORAGE_KEYS = [
  'yolo-forge-task-overview-preferences',
  'yolo-forge-task-overview-settings',
] as const
const taskCenterListeners = new Set<() => void>()
const taskCenterServerSnapshot: TaskCenterPreferencesSnapshot = Object.freeze({
  ...TASK_CENTER_PREFERENCES_DEFAULTS,
  storageAvailable: false,
})
let taskCenterSnapshot = taskCenterServerSnapshot
let taskCenterInitialized = false

function normalizeTaskCenterZoom(value: unknown, fallback: number): number {
  const parsed = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.round(Math.min(1.2, Math.max(0.6, parsed)) * 100) / 100
}

export function normalizeTaskCenterPreferences(
  value: unknown,
  fallback: TaskCenterPreferences = TASK_CENTER_PREFERENCES_DEFAULTS,
): TaskCenterPreferences {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return { ...fallback }
  const candidate = value as Record<string, unknown>
  const legacyLayout = candidate.layout
  const legacyDensity = candidate.density
  return {
    layoutMode: candidate.layoutMode === 'free' || candidate.layoutMode === 'grid'
      ? candidate.layoutMode
      : legacyLayout === 'grid' || legacyLayout === 'auto' ? 'grid'
        : legacyLayout === 'free' || legacyLayout === 'manual' ? 'free' : fallback.layoutMode,
    cardDensity: candidate.cardDensity === 'comfortable' || candidate.cardDensity === 'compact'
      ? candidate.cardDensity
      : legacyDensity === 'compact' ? 'compact'
        : legacyDensity === 'comfortable' || legacyDensity === 'standard' ? 'comfortable' : fallback.cardDensity,
    defaultZoom: normalizeTaskCenterZoom(candidate.defaultZoom ?? candidate.zoom, fallback.defaultZoom),
    showMiniMap: typeof candidate.showMiniMap === 'boolean' ? candidate.showMiniMap
      : typeof candidate.minimap === 'boolean' ? candidate.minimap : fallback.showMiniMap,
    showConnections: typeof candidate.showConnections === 'boolean' ? candidate.showConnections
      : typeof candidate.showEdges === 'boolean' ? candidate.showEdges : fallback.showConnections,
  }
}

function taskCenterPreferencesEqual(first: TaskCenterPreferencesSnapshot, second: TaskCenterPreferencesSnapshot): boolean {
  return first.layoutMode === second.layoutMode
    && first.cardDensity === second.cardDensity
    && first.defaultZoom === second.defaultZoom
    && first.showMiniMap === second.showMiniMap
    && first.showConnections === second.showConnections
    && first.storageAvailable === second.storageAvailable
}

function readTaskCenterStoredPreferences(): TaskCenterPreferencesSnapshot {
  if (typeof window === 'undefined') return taskCenterServerSnapshot
  try {
    const keys = [TASK_CENTER_PREFERENCES_STORAGE_KEY, ...LEGACY_TASK_CENTER_STORAGE_KEYS]
    for (const key of keys) {
      const stored = window.localStorage.getItem(key)
      if (stored === null) continue
      try {
        const parsed: unknown = JSON.parse(stored)
        return Object.freeze({ ...normalizeTaskCenterPreferences(parsed), storageAvailable: true })
      } catch {
        // A malformed current value should not hide a valid legacy preference.
      }
    }
    return Object.freeze({ ...TASK_CENTER_PREFERENCES_DEFAULTS, storageAvailable: true })
  } catch {
    return taskCenterServerSnapshot
  }
}

function publishTaskCenterPreferences(next: TaskCenterPreferencesSnapshot): void {
  if (taskCenterPreferencesEqual(taskCenterSnapshot, next)) return
  taskCenterSnapshot = Object.freeze(next)
  taskCenterListeners.forEach((listener) => listener())
}

export function initializeTaskCenterPreferences(): void {
  if (taskCenterInitialized || typeof window === 'undefined') return
  taskCenterInitialized = true
  publishTaskCenterPreferences(readTaskCenterStoredPreferences())
  window.addEventListener('storage', (event) => {
    if (event.key !== null
      && event.key !== TASK_CENTER_PREFERENCES_STORAGE_KEY
      && !LEGACY_TASK_CENTER_STORAGE_KEYS.includes(event.key as typeof LEGACY_TASK_CENTER_STORAGE_KEYS[number])) return
    try {
      if (event.storageArea && event.storageArea !== window.localStorage) return
    } catch { return }
    publishTaskCenterPreferences(readTaskCenterStoredPreferences())
  })
}

function subscribeTaskCenterPreferences(listener: () => void): () => void {
  taskCenterListeners.add(listener)
  return () => { taskCenterListeners.delete(listener) }
}

export function useTaskCenterPreferences(): TaskCenterPreferencesSnapshot {
  initializeTaskCenterPreferences()
  return useSyncExternalStore(
    subscribeTaskCenterPreferences,
    () => taskCenterSnapshot,
    () => taskCenterServerSnapshot,
  )
}

export function updateTaskCenterPreferences(patch: Partial<TaskCenterPreferences>): void {
  initializeTaskCenterPreferences()
  const preferences = normalizeTaskCenterPreferences(patch, normalizeTaskCenterPreferences(taskCenterSnapshot))
  let storageAvailable = false
  try {
    window.localStorage.setItem(TASK_CENTER_PREFERENCES_STORAGE_KEY, JSON.stringify(preferences))
    storageAvailable = true
  } catch {
    // Keep the preference active in memory when browser storage is unavailable.
  }
  publishTaskCenterPreferences({ ...preferences, storageAvailable })
}

export function resetTaskCenterPreferences(): void {
  updateTaskCenterPreferences(TASK_CENTER_PREFERENCES_DEFAULTS)
}

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
