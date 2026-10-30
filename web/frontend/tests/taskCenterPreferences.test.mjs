import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const storageKey = 'yolo-forge-task-center-v1'
const source = readFileSync(new URL('../src/preferences.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText

function browser({ saved = {}, blocked = false, readOnly = false } = {}) {
  const values = new Map(Object.entries(saved))
  const listeners = new Map()
  const storage = {
    getItem(key) { return values.get(key) ?? null },
    setItem(key, value) {
      if (readOnly) throw new Error('Storage quota exceeded')
      values.set(key, value)
    },
  }
  const window = {
    get localStorage() {
      if (blocked) throw new Error('Storage access denied')
      return storage
    },
    addEventListener(type, listener) {
      if (!listeners.has(type)) listeners.set(type, new Set())
      listeners.get(type).add(listener)
    },
    dispatchEvent() {},
  }
  const module = { exports: {} }
  let subscribe
  vm.runInNewContext(compiled, {
    exports: module.exports,
    Event,
    localStorage: storage,
    require(name) {
      assert.equal(name, 'react')
      return { useSyncExternalStore(nextSubscribe, getSnapshot) {
        subscribe = nextSubscribe
        return getSnapshot()
      } }
    },
    window,
  })
  const api = module.exports
  return {
    api,
    values,
    snapshot() { return api.useTaskCenterPreferences() },
    listen(listener) { api.useTaskCenterPreferences(); return subscribe(listener) },
    storageEvent(key, newValue, storageArea = storage) {
      if (key === null) values.clear()
      else if (newValue === null) values.delete(key)
      else values.set(key, newValue)
      listeners.get('storage')?.forEach((listener) => listener({ key, newValue, storageArea }))
    },
  }
}

test('task center defaults preserve readable free-layout behavior', () => {
  const page = browser()
  assert.deepEqual({ ...page.snapshot() }, {
    layoutMode: 'free',
    cardDensity: 'comfortable',
    defaultZoom: 0.9,
    showMiniMap: true,
    showConnections: true,
    storageAvailable: true,
  })
})

test('old and malformed preference values are normalized field by field', () => {
  const normalized = browser({ saved: {
    [storageKey]: JSON.stringify({
      layout: 'auto', density: 'standard', zoom: 9, minimap: false, showEdges: false,
    }),
  } }).snapshot()
  assert.equal(normalized.layoutMode, 'grid')
  assert.equal(normalized.cardDensity, 'comfortable')
  assert.equal(normalized.defaultZoom, 1.2)
  assert.equal(normalized.showMiniMap, false)
  assert.equal(normalized.showConnections, false)

  const malformed = browser({ saved: {
    [storageKey]: JSON.stringify({ layoutMode: 'broken', cardDensity: 'tiny', defaultZoom: 'nope' }),
  } }).snapshot()
  assert.equal(malformed.layoutMode, 'free')
  assert.equal(malformed.cardDensity, 'comfortable')
  assert.equal(malformed.defaultZoom, 0.9)
})

test('legacy storage migrates in memory and new writes use the versioned key', () => {
  const page = browser({ saved: {
    'yolo-forge-task-overview-settings': JSON.stringify({ layout: 'manual', density: 'compact', zoom: 0.72 }),
  } })
  assert.equal(page.snapshot().layoutMode, 'free')
  assert.equal(page.snapshot().cardDensity, 'compact')
  assert.equal(page.snapshot().defaultZoom, 0.72)
  page.api.updateTaskCenterPreferences({ showMiniMap: false })
  assert.equal(JSON.parse(page.values.get(storageKey)).showMiniMap, false)
  assert.equal(JSON.parse(page.values.get(storageKey)).cardDensity, 'compact')
})

test('updates clamp zoom, preserve unrelated fields and reset all preferences', () => {
  const page = browser()
  page.api.updateTaskCenterPreferences({ layoutMode: 'grid', cardDensity: 'compact', defaultZoom: 0.1, showConnections: false })
  assert.equal(page.snapshot().layoutMode, 'grid')
  assert.equal(page.snapshot().cardDensity, 'compact')
  assert.equal(page.snapshot().defaultZoom, 0.6)
  assert.equal(page.snapshot().showMiniMap, true)
  assert.equal(page.snapshot().showConnections, false)
  page.api.resetTaskCenterPreferences()
  assert.deepEqual(JSON.parse(page.values.get(storageKey)), {
    layoutMode: 'free', cardDensity: 'comfortable', defaultZoom: 0.9, showMiniMap: true, showConnections: true,
  })
})

test('blocked storage keeps live in-memory settings and cross-tab storage changes synchronize', () => {
  const blocked = browser({ blocked: true })
  blocked.api.updateTaskCenterPreferences({ cardDensity: 'compact' })
  assert.equal(blocked.snapshot().cardDensity, 'compact')
  assert.equal(blocked.snapshot().storageAvailable, false)

  const readOnly = browser({ readOnly: true })
  readOnly.api.updateTaskCenterPreferences({ layoutMode: 'grid' })
  assert.equal(readOnly.snapshot().layoutMode, 'grid')
  assert.equal(readOnly.snapshot().storageAvailable, false)

  const page = browser()
  let updates = 0
  const unsubscribe = page.listen(() => { updates += 1 })
  page.storageEvent(storageKey, JSON.stringify({ cardDensity: 'compact', defaultZoom: 1.05 }))
  assert.equal(page.snapshot().cardDensity, 'compact')
  assert.equal(page.snapshot().defaultZoom, 1.05)
  assert.equal(updates, 1)
  page.storageEvent('unrelated', 'ignored')
  assert.equal(updates, 1)
  page.storageEvent(null, null)
  assert.equal(page.snapshot().cardDensity, 'comfortable')
  assert.equal(updates, 2)
  unsubscribe()
})
