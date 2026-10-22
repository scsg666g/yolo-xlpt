import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const storageKey = 'yolo-forge-appearance-v1'
const legacyKey = 'yolo-forge-theme'
const source = readFileSync(new URL('../src/appearance.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText

// Run the real store in isolated browser contexts without requiring a DOM test dependency.
function browser({ saved = {}, dark = false, blocked = false, readOnly = false } = {}) {
  const values = new Map(Object.entries(saved))
  const properties = new Map()
  const events = new Map()
  const mediaEvents = new Set()
  const media = {
    matches: dark,
    addEventListener(_type, listener) { mediaEvents.add(listener) },
  }
  const storage = {
    getItem(key) { return values.get(key) ?? null },
    setItem(key, value) {
      if (readOnly) throw new Error('Storage quota exceeded')
      values.set(key, value)
    },
  }
  const document = { documentElement: { dataset: {}, style: { setProperty(key, value) { properties.set(key, value) } } } }
  const window = {
    get localStorage() {
      if (blocked) throw new Error('Storage access denied')
      return storage
    },
    matchMedia() { return media },
    addEventListener(type, listener) {
      if (!events.has(type)) events.set(type, new Set())
      events.get(type).add(listener)
    },
  }
  const module = { exports: {} }
  let subscribe
  vm.runInNewContext(compiled, {
    exports: module.exports,
    require(name) {
      assert.equal(name, 'react')
      return { useSyncExternalStore(nextSubscribe, getSnapshot) {
        subscribe = nextSubscribe
        return getSnapshot()
      } }
    },
    window,
    document,
  })
  const api = module.exports
  api.initializeAppearance()
  return {
    api,
    values,
    properties,
    dataset: document.documentElement.dataset,
    snapshot() { return api.useAppearance() },
    listen(listener) { api.useAppearance(); return subscribe(listener) },
    systemDark(matches) {
      media.matches = matches
      mediaEvents.forEach((listener) => listener({ matches }))
    },
    storageEvent(key, newValue, storageArea = storage) {
      if (key === null) values.clear()
      else if (newValue === null) values.delete(key)
      else values.set(key, newValue)
      events.get('storage')?.forEach((listener) => listener({ key, newValue, storageArea }))
    },
  }
}

function contrast(first, second) {
  function luminance(hex) {
    const channels = [1, 3, 5].map((offset) => {
      const value = parseInt(hex.slice(offset, offset + 2), 16) / 255
      return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
    })
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
  }
  const a = luminance(first)
  const b = luminance(second)
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05)
}

test('fresh browsers retain the current light default even on a dark operating system', () => {
  const page = browser({ dark: true })
  assert.equal(page.snapshot().mode, 'light')
  assert.equal(page.dataset.theme, 'light')
  assert.equal(page.dataset.accent, 'default')
  assert.equal(page.dataset.fontSize, 'standard')
  assert.equal(page.dataset.density, 'default')
  assert.equal(page.properties.get('--accent'), '#159a78')
  assert.equal(page.properties.get('--accent-contrast'), '#ffffff')
  assert.equal(page.properties.has('--green'), false)
})

test('existing dark preference survives migration and malformed new JSON', () => {
  for (const raw of [undefined, '{broken', 'null', '[]']) {
    const page = browser({ saved: { [legacyKey]: 'dark', ...(raw ? { [storageKey]: raw } : {}) } })
    assert.equal(page.snapshot().mode, 'dark')
    assert.equal(page.dataset.theme, 'dark')
    assert.equal(page.properties.get('--accent'), '#76e9ba')
    assert.equal(page.snapshot().storageAvailable, true)
  }
})

test('malformed fields are normalized independently and cannot inject CSS', () => {
  const page = browser({ saved: { [storageKey]: JSON.stringify({
    mode: 'invalid', accent: 'custom', customColor: 'red; background: url(bad)', fontSize: 'huge', density: 'compact',
  }) } })
  assert.equal(page.snapshot().mode, 'light')
  assert.equal(page.snapshot().accent, 'custom')
  assert.equal(page.snapshot().customColor, '#159a78')
  assert.equal(page.snapshot().fontSize, 'standard')
  assert.equal(page.snapshot().density, 'compact')
  page.api.updateAppearance({ customColor: ' #AbC ' })
  assert.equal(page.properties.get('--accent'), '#aabbcc')
  page.api.updateAppearance({ customColor: 'invalid', mode: 'invalid' })
  assert.equal(page.properties.get('--accent'), '#aabbcc')
  assert.equal(page.snapshot().mode, 'light')
})

test('preferences persist across browser loads and reset restores the full default', () => {
  const page = browser()
  page.api.updateAppearance({ mode: 'dark', accent: 'custom', customColor: '#abcdef', fontSize: 'large', density: 'comfortable' })
  const reloaded = browser({ saved: Object.fromEntries(page.values) })
  assert.equal(reloaded.snapshot().mode, 'dark')
  assert.equal(reloaded.snapshot().accent, 'custom')
  assert.equal(reloaded.snapshot().customColor, '#abcdef')
  assert.equal(reloaded.dataset.fontSize, 'large')
  assert.equal(reloaded.dataset.density, 'comfortable')
  reloaded.api.resetAppearance()
  const defaults = reloaded.api.APPEARANCE_DEFAULTS
  assert.equal(reloaded.values.get(storageKey), JSON.stringify(defaults))
  assert.equal(reloaded.values.get(legacyKey), 'light')
  for (const [key, value] of Object.entries(defaults)) assert.equal(reloaded.snapshot()[key], value)
})

test('blocked reads or quota errors still allow changing and resetting appearance in memory', () => {
  for (const options of [{ blocked: true }, { readOnly: true }]) {
    const page = browser(options)
    page.api.updateAppearance({ accent: 'blue', mode: 'dark' })
    assert.equal(page.snapshot().storageAvailable, false)
    assert.equal(page.dataset.theme, 'dark')
    assert.equal(page.dataset.accent, 'blue')
    page.api.resetAppearance()
    assert.equal(page.dataset.theme, 'light')
    assert.equal(page.dataset.accent, 'default')
  }
})

test('system mode follows live OS changes while explicit modes remain unchanged', () => {
  const page = browser({ dark: true })
  page.api.updateAppearance({ mode: 'system' })
  assert.equal(page.snapshot().resolvedTheme, 'dark')
  page.systemDark(false)
  assert.equal(page.snapshot().resolvedTheme, 'light')
  assert.equal(JSON.parse(page.values.get(storageKey)).mode, 'system')
  page.api.updateAppearance({ mode: 'dark' })
  const before = page.snapshot()
  page.systemDark(true)
  page.systemDark(false)
  assert.equal(page.snapshot(), before)
})

test('storage changes synchronize tabs, ignore unrelated storage and handle clear', () => {
  const page = browser()
  page.storageEvent(storageKey, JSON.stringify({ mode: 'system', accent: 'violet', fontSize: 'large' }))
  assert.equal(page.dataset.accent, 'violet')
  assert.equal(page.dataset.fontSize, 'large')
  page.storageEvent(legacyKey, 'dark')
  assert.equal(page.snapshot().mode, 'system')
  const before = page.snapshot()
  page.storageEvent('unrelated-key', 'ignored')
  assert.equal(page.snapshot(), before)
  page.storageEvent(storageKey, JSON.stringify({ accent: 'amber' }), {})
  assert.equal(page.snapshot(), before)
  page.storageEvent(null, null)
  assert.equal(page.dataset.accent, 'default')
  assert.equal(page.snapshot().mode, 'light')
})

test('external store snapshots are stable and listeners only receive effective changes', () => {
  const page = browser()
  const before = page.snapshot()
  let updates = 0
  const unsubscribe = page.listen(() => { updates += 1 })
  page.api.initializeAppearance()
  page.api.updateAppearance({ mode: 'light' })
  assert.equal(page.snapshot(), before)
  assert.equal(updates, 0)
  page.api.updateAppearance({ accent: 'blue' })
  assert.equal(updates, 1)
  assert.notEqual(page.snapshot(), before)
  unsubscribe()
  page.api.updateAppearance({ accent: 'violet' })
  assert.equal(updates, 1)
})

test('preset and extreme custom colors remain readable in both themes without changing semantic colors', () => {
  const page = browser()
  const colors = ['#ffffff', '#000000', '#ffff00', '#0000ff', '#ff0000', '#00ff00', '#777777', '#abcdef', '#18242f', ...page.api.ACCENT_PRESETS.map((preset) => preset.color)]
  for (const mode of ['light', 'dark']) {
    for (const customColor of colors) {
      page.api.updateAppearance({ mode, accent: 'custom', customColor })
      assert.equal(page.properties.get('--accent'), customColor)
      const text = page.properties.get('--accent-text')
      assert.ok(contrast(text, mode === 'light' ? '#ffffff' : '#18242f') >= 4.5, `${mode} text ${text} for ${customColor}`)
      if (mode === 'dark') assert.ok(contrast(text, '#0e151d') >= 4.5, `dark base text ${text} for ${customColor}`)
      assert.ok(contrast(customColor, page.properties.get('--accent-contrast')) >= 4.5, `${mode} button ${customColor}`)
      assert.equal(page.properties.has('--green'), false)
    }
  }
})
