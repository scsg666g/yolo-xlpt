import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

function compiled(relative) {
  return ts.transpileModule(readFileSync(new URL(relative, import.meta.url), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText
}

const dataset = {
  id: 'dataset-a', name: '训练数据', node_id: 'local', status: 'ready', ownership: 'managed',
  created_at: null, checked_at: null, directory: '/dataset', config_path: '/dataset/data.yaml',
  stats: {}, classes: [], warnings: [], references: [], total_bytes: 0, can_delete: true,
}

// Render the actual component and run its effects with a small hook host. Network
// responses and browser visibility are controlled; no live service is contacted.
function page(api) {
  const slots = []; const effects = []; const timers = new Map(); const listeners = new Map()
  let index = 0; let timerId = 0; let tree
  const same = (a, b) => a && b && a.length === b.length && a.every((value, i) => Object.is(value, b[i]))
  const react = {
    useState(initial) {
      const i = index++
      if (!slots[i]) slots[i] = { value: typeof initial === 'function' ? initial() : initial }
      return [slots[i].value, (value) => { slots[i].value = typeof value === 'function' ? value(slots[i].value) : value }]
    },
    useRef(value) { const i = index++; return slots[i] ||= { current: value } },
    useMemo(work, deps) { const i = index++; if (!slots[i] || !same(slots[i].deps, deps)) slots[i] = { value: work(), deps }; return slots[i].value },
    useCallback(work, deps) { return react.useMemo(() => work, deps) },
    useEffect(work, deps) {
      const i = index++
      if (!slots[i] || !same(slots[i].deps, deps)) effects.push(() => { slots[i]?.cleanup?.(); slots[i] = { deps, cleanup: work() } })
    },
  }
  const document = { hidden: false, addEventListener: (name, work) => listeners.set(`document:${name}`, work), removeEventListener: name => listeners.delete(`document:${name}`) }
  const window = { addEventListener: (name, work) => listeners.set(`window:${name}`, work), removeEventListener: name => listeners.delete(`window:${name}`) }
  const module = { exports: {} }
  vm.runInNewContext(compiled('../src/components/Datasets.tsx'), {
    exports: module.exports, AbortController, Map, Set, Date, document, window,
    crypto: { randomUUID: () => `request-${++timerId}` },
    setTimeout: work => { const id = ++timerId; timers.set(id, work); return id }, clearTimeout: id => timers.delete(id),
    require(name) {
      if (name === 'react') return react
      if (name === 'react/jsx-runtime') return { jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) }
      if (name === '../api') return { api, formatBytes: String, statusLabel: String }
      return new Proxy({}, { get: (_target, key) => String(key) })
    },
  })
  function render() { index = 0; tree = module.exports.Datasets(); for (const effect of effects.splice(0)) effect(); return tree }
  function find(predicate) {
    function visit(node) {
      if (!node || typeof node !== 'object') return null
      if (Array.isArray(node)) { for (const child of node) { const found = visit(child); if (found) return found }; return null }
      return predicate(node) ? node : visit(node.props?.children)
    }
    return visit(tree)
  }
  const settle = async () => { for (let i = 0; i < 15; i++) await Promise.resolve(); render() }
  render()
  return {
    find, render, settle,
    async tick() { const work = [...timers.values()]; timers.clear(); for (const next of work) next(); await settle() },
    async focus() { listeners.get('window:focus')?.(); await settle() },
    async visible(hidden) { document.hidden = hidden; listeners.get('document:visibilitychange')?.(); await settle() },
    dispose() { for (const item of slots) item?.cleanup?.() },
  }
}

function baseApi(overrides = {}) {
  return { datasetEntries: async () => [dataset], computeNodes: async () => [],
    datasetOperationSummary: async () => ({ revision: 'a', active: 0, failed: 0, nodes: [] }),
    datasetOperation: async () => { const error = new Error('missing'); error.status = 404; throw error }, ...overrides }
}

test('a failed initial catalogue read recovers through polling without manual refresh', async () => {
  let calls = 0
  const view = page(baseApi({ datasetEntries: async () => { if (++calls === 1) throw new Error('offline'); return [dataset] } }))
  await view.settle()
  if (!view.find(item => item.props?.className === 'dataset-card')) await view.tick()
  assert.ok(view.find(item => item.props?.className === 'dataset-card'))
  assert.ok(calls >= 2)
  view.dispose()
})

test('a changed revision remains pending after a failed list request and retries next poll', async () => {
  let revision = 'a'; let fail = false; let calls = 0
  const view = page(baseApi({
    datasetEntries: async () => { calls++; if (fail) { fail = false; throw new Error('temporary failure') }; return [dataset] },
    datasetOperationSummary: async () => ({ revision, active: 0, failed: 0, nodes: [] }),
  }))
  await view.settle(); const initialCalls = calls
  revision = 'b'; fail = true
  await view.tick(); assert.equal(calls, initialCalls + 1)
  await view.tick(); assert.equal(calls, initialCalls + 2)
  await view.tick(); assert.equal(calls, initialCalls + 2)
  view.dispose()
})

test('returning to a tab reloads references even when the operation revision is unchanged', async () => {
  let calls = 0
  const view = page(baseApi({ datasetEntries: async () => { calls++; return [dataset] } }))
  await view.settle(); const initialCalls = calls
  await view.focus(); assert.equal(calls, initialCalls + 1)
  await view.visible(true); await view.tick(); assert.equal(calls, initialCalls + 1)
  await view.visible(false); assert.equal(calls, initialCalls + 2)
  view.dispose()
})

test('an uncertain reload submission retries with the same request identity', async () => {
  const requests = []
  const view = page(baseApi({ refreshDatasetEntry: async (_id, requestId) => {
    requests.push(requestId)
    if (requests.length === 1) throw new Error('connection lost after submission')
    return { status: 'completed' }
  } }))
  await view.settle()
  const reload = () => view.find(item => item.props?.['aria-label'] === '重新加载数据集：训练数据')
  reload().props.onClick(); await view.settle()
  assert.equal(reload().props.disabled, false)
  assert.ok(view.find(item => item.props?.role === 'alert'), 'a background list reload must preserve the submission error')
  reload().props.onClick(); await view.settle()
  assert.equal(requests.length, 2); assert.equal(requests[0], requests[1])
  view.dispose()
})

test('model tests distinguish a weight source from central execution with old API fallbacks', () => {
  const module = { exports: {} }
  vm.runInNewContext(compiled('../src/components/ModelTestShared.tsx'), {
    exports: module.exports, require: () => new Proxy({}, { get: (_target, key) => String(key) }),
  })
  const { modelTestExecutionLabel, modelTestWaitingMessage } = module.exports
  assert.equal(modelTestExecutionLabel({ model: { device: 'cpu', imgsz: 320 } }), '中心服务器 · CPU · 320 px')
  assert.equal(modelTestExecutionLabel({ model: { device: 'cuda:0', imgsz: 640 }, execution: { device: 'cpu' } }), '中心服务器 · CPU · 640 px')
  assert.match(modelTestWaitingMessage({ status: 'queued', queue_ahead: 2 }), /前方 2 个/)
  assert.match(modelTestWaitingMessage({ status: 'queued' }), /中心服务器的测试工作线程/)
  assert.equal(modelTestWaitingMessage({ status: 'completed' }), null)
})

test('dataset provenance never substitutes current information into old attempts', () => {
  const module = { exports: {} }
  vm.runInNewContext(compiled('../src/datasetSnapshot.ts'), { exports: module.exports })
  const snapshot = module.exports.datasetTrainingSnapshot
  assert.equal(snapshot(), null); assert.equal(snapshot({}), null)
  assert.equal(snapshot({ dataset_snapshot: { version: 2 } }), null)
  const value = snapshot({ dataset_snapshot: { version: 1, source: 'managed_catalog', verification: 'catalog_snapshot',
    node_id: 'node-b', fingerprint: 'a'.repeat(64), config_sha256: 'invalid', stats: { train_images: 3, val_images: -1, classes: '4' }, classes: ['人员', 123] } })
  assert.equal(value.node_id, 'node-b'); assert.equal(value.fingerprint, 'a'.repeat(64))
  assert.equal(value.config_sha256, null); assert.equal(value.captured_at, null)
  assert.deepEqual(Object.keys(value.stats), ['train_images']); assert.equal(value.classes.length, 1)
})
