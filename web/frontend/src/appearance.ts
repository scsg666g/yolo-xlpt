import { useSyncExternalStore } from 'react'

export type ThemeMode = 'light' | 'dark' | 'system'

export type AppearancePreferences = {
  mode: ThemeMode
  accent: 'default' | 'blue' | 'violet' | 'amber' | 'custom'
  customColor: string
  fontSize: 'standard' | 'large'
  density: 'default' | 'compact' | 'comfortable'
}

export const APPEARANCE_DEFAULTS: AppearancePreferences = Object.freeze({
  mode: 'light',
  accent: 'default',
  customColor: '#159a78',
  fontSize: 'standard',
  density: 'default',
})

export const ACCENT_PRESETS: ReadonlyArray<{
  id: Exclude<AppearancePreferences['accent'], 'custom'>
  label: string
  color: string
}> = [
  { id: 'default', label: '默认绿', color: '#159a78' },
  { id: 'blue', label: '科技蓝', color: '#347bd1' },
  { id: 'violet', label: '紫罗兰', color: '#7c3aed' },
  { id: 'amber', label: '琥珀橙', color: '#d97706' },
]

const STORAGE_KEY = 'yolo-forge-appearance-v1'
const LEGACY_THEME_KEY = 'yolo-forge-theme'
type ResolvedTheme = 'light' | 'dark'
type AppearanceSnapshot = AppearancePreferences & {
  resolvedTheme: ResolvedTheme
  storageAvailable: boolean
}

const serverSnapshot: AppearanceSnapshot = Object.freeze({
  ...APPEARANCE_DEFAULTS,
  resolvedTheme: 'light',
  storageAvailable: false,
})
let snapshot = serverSnapshot
let initialized = false
let systemTheme: MediaQueryList | undefined
const listeners = new Set<() => void>()

function normalizeColor(value: unknown, fallback: string): string {
  if (typeof value !== 'string') return fallback
  const color = value.trim().toLowerCase()
  if (/^#[0-9a-f]{6}$/.test(color)) return color
  if (/^#[0-9a-f]{3}$/.test(color)) {
    return `#${[...color.slice(1)].map((digit) => digit.repeat(2)).join('')}`
  }
  return fallback
}

function normalizePreferences(value: unknown, fallback = APPEARANCE_DEFAULTS): AppearancePreferences {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return { ...fallback }
  const candidate = value as Record<string, unknown>
  return {
    mode: candidate.mode === 'light' || candidate.mode === 'dark' || candidate.mode === 'system'
      ? candidate.mode : fallback.mode,
    accent: candidate.accent === 'default' || candidate.accent === 'blue'
      || candidate.accent === 'violet' || candidate.accent === 'amber' || candidate.accent === 'custom'
      ? candidate.accent : fallback.accent,
    customColor: normalizeColor(candidate.customColor, fallback.customColor),
    fontSize: candidate.fontSize === 'standard' || candidate.fontSize === 'large'
      ? candidate.fontSize : fallback.fontSize,
    density: candidate.density === 'default' || candidate.density === 'compact' || candidate.density === 'comfortable'
      ? candidate.density : fallback.density,
  }
}

function readStoredPreferences(): { preferences: AppearancePreferences; storageAvailable: boolean } {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    if (stored !== null) {
      try {
        const parsed: unknown = JSON.parse(stored)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          return { preferences: normalizePreferences(parsed), storageAvailable: true }
        }
      } catch { /* Invalid saved data falls back to the previous theme preference. */ }
    }
    const legacyTheme = window.localStorage.getItem(LEGACY_THEME_KEY)
    return {
      preferences: { ...APPEARANCE_DEFAULTS, mode: legacyTheme === 'dark' ? 'dark' : 'light' },
      storageAvailable: true,
    }
  } catch {
    return { preferences: { ...APPEARANCE_DEFAULTS }, storageAvailable: false }
  }
}

function resolveTheme(mode: ThemeMode): ResolvedTheme {
  return mode === 'system' ? (systemTheme?.matches ? 'dark' : 'light') : mode
}

type RGB = [number, number, number]

function rgb(color: string): RGB {
  return [1, 3, 5].map((offset) => parseInt(color.slice(offset, offset + 2), 16)) as RGB
}

function luminance(color: RGB): number {
  const channels = color.map((channel) => {
    const value = channel / 255
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
  })
  return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
}

function contrast(first: RGB, second: RGB): number {
  const a = luminance(first)
  const b = luminance(second)
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)
}

function readableAccent(color: string, theme: ResolvedTheme): string {
  const original = rgb(color)
  const backgrounds = (theme === 'light' ? ['#ffffff'] : ['#18242f', '#0e151d']).map(rgb)
  const isReadable = (foreground: RGB) => backgrounds.every((background) => contrast(foreground, background) >= 4.5)
  if (isReadable(original)) return color
  // Mixing only toward black or white keeps the chosen hue while adjusting readability.
  const endpoint = theme === 'light' ? 0 : 255
  let low = 0
  let high = 1
  let result: RGB = [endpoint, endpoint, endpoint]
  for (let step = 0; step < 16; step += 1) {
    const amount = (low + high) / 2
    const mixed = original.map((channel) => Math.round(channel + (endpoint - channel) * amount)) as RGB
    if (isReadable(mixed)) {
      high = amount
      result = mixed
    } else {
      low = amount
    }
  }
  return `#${result.map((channel) => channel.toString(16).padStart(2, '0')).join('')}`
}

function applyAppearance(value: AppearanceSnapshot): void {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  root.dataset.theme = value.resolvedTheme
  root.dataset.accent = value.accent
  root.dataset.fontSize = value.fontSize
  root.dataset.density = value.density
  root.style.colorScheme = value.resolvedTheme

  const color = value.accent === 'default'
    ? (value.resolvedTheme === 'light' ? '#159a78' : '#76e9ba')
    : value.accent === 'custom'
      ? value.customColor
      : ACCENT_PRESETS.find((preset) => preset.id === value.accent)!.color
  const channels = rgb(color)
  const foreground = value.accent === 'default'
    ? (value.resolvedTheme === 'light' ? '#ffffff' : '#0b1b16')
    : (contrast(channels, [255, 255, 255]) >= contrast(channels, [0, 0, 0]) ? '#ffffff' : '#000000')
  root.style.setProperty('--accent', color)
  root.style.setProperty('--accent-contrast', foreground)
  root.style.setProperty('--accent-text', readableAccent(color, value.resolvedTheme))
  root.style.setProperty('--accent-soft', `rgba(${channels.join(', ')}, 0.12)`)
  root.style.setProperty('--accent-border', `rgba(${channels.join(', ')}, 0.35)`)
}

function publish(preferences: AppearancePreferences, storageAvailable: boolean): void {
  const next: AppearanceSnapshot = {
    ...preferences,
    resolvedTheme: resolveTheme(preferences.mode),
    storageAvailable,
  }
  applyAppearance(next)
  if ((Object.keys(next) as Array<keyof AppearanceSnapshot>).every((key) => next[key] === snapshot[key])) return
  snapshot = Object.freeze(next)
  listeners.forEach((listener) => listener())
}

export function initializeAppearance(): void {
  if (initialized || typeof window === 'undefined') return
  initialized = true
  if (typeof window.matchMedia === 'function') {
    systemTheme = window.matchMedia('(prefers-color-scheme: dark)')
    const onSystemThemeChange = () => {
      if (snapshot.mode === 'system') publish(normalizePreferences(snapshot), snapshot.storageAvailable)
    }
    if (typeof systemTheme.addEventListener === 'function') systemTheme.addEventListener('change', onSystemThemeChange)
    else systemTheme.addListener(onSystemThemeChange)
  }
  const stored = readStoredPreferences()
  publish(stored.preferences, stored.storageAvailable)
  window.addEventListener('storage', (event) => {
    if (event.key !== STORAGE_KEY && event.key !== LEGACY_THEME_KEY && event.key !== null) return
    // Ignore unrelated sessionStorage events, which can use the same key in another frame.
    try {
      if (event.storageArea && event.storageArea !== window.localStorage) return
    } catch { return }
    const updated = readStoredPreferences()
    publish(updated.preferences, updated.storageAvailable)
  })
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => { listeners.delete(listener) }
}

export function useAppearance(): AppearanceSnapshot {
  initializeAppearance()
  return useSyncExternalStore(subscribe, () => snapshot, () => serverSnapshot)
}

export function updateAppearance(patch: Partial<AppearancePreferences>): void {
  initializeAppearance()
  const preferences = normalizePreferences(patch, normalizePreferences(snapshot))
  let storageAvailable = false
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences))
    window.localStorage.setItem(LEGACY_THEME_KEY, resolveTheme(preferences.mode))
    storageAvailable = true
  } catch { /* Preferences remain active in memory when browser storage is unavailable. */ }
  publish(preferences, storageAvailable)
}

export function resetAppearance(): void {
  updateAppearance(APPEARANCE_DEFAULTS)
}
