import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

function runtimeView(api) {
  const slots = []; const effects = []; const timers = new Map()
  let index = 0; let timerId = 0; let tree
  const react = {
    useState(initial) {
      const i = index++
      slots[i] ||= { value: initial }
      return [slots[i].value, value => { slots[i].value = value }]
    },
    useRef(value) { const i = index++; return slots[i] ||= { current: value } },
    useEffect(work) { const i = index++; if (!slots[i]) { slots[i] = {}; effects.push(() => { slots[i].cleanup = work() }) } },
  }
  const document = { hidden: false }
  const window = { setTimeout(work) { const id = ++timerId; timers.set(id, work); return id }, clearTimeout(id) { timers.delete(id) } }
  const module = { exports: {} }
  const source = readFileSync(new URL('../src/components/ModelTestRuntimeStatus.tsx', import.meta.url), 'utf8')
  const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
  vm.runInNewContext(compiled, { exports: module.exports, window, document, AbortController, Error,
    require(name) {
      if (name === 'react') return react
      if (name === 'react/jsx-runtime') return { jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) }
      if (name === '../api') return { api }
      return new Proxy({}, { get: (_, key) => String(key) })
    },
  })
  function render() { index = 0; tree = module.exports.ModelTestRuntimeStatus(); for (const effect of effects.splice(0)) effect() }
  function find(node, type) {
    if (!node || typeof node !== 'object') return null
    if (Array.isArray(node)) return node.map(item => find(item, type)).find(Boolean)
    return node.type === type ? node : find(node.props?.children, type)
  }
  function text(node) {
    if (node == null || typeof node === 'boolean') return ''
    if (typeof node !== 'object') return String(node)
    if (Array.isArray(node)) return node.map(text).join('')
    return text(node.props?.children)
  }
  async function settle() { for (let i = 0; i < 12; i++) await Promise.resolve(); render() }
  render()
  return {
    settle, text: () => text(tree), button: () => find(tree, 'button'),
    async tick() { const work = [...timers.values()]; timers.clear(); work.forEach(next => next()); await settle() },
    hide(value) { document.hidden = value },
    dispose() { slots.forEach(slot => slot?.cleanup?.()) },
  }
}

const ready = { state: 'ready', model_name: 'best.pt', active_test_id: null, idle_seconds: 300,
  expires_in_seconds: 299, waiting_tests: 0, can_release: true, release_reason: 'not_loaded', device: 'CPU', location: 'center' }
const released = { ...ready, state: 'released', model_name: null, expires_in_seconds: null, can_release: false, release_reason: 'manual' }

test('runtime shows resident model and manual release without changing histories', async () => {
  let count = 0
  const view = runtimeView({ modelTestRuntime: async () => ready, releaseModelTestRuntime: async () => { count++; return released } })
  await view.settle()
  assert.match(view.text(), /模型已驻留，可复用/)
  assert.match(view.text(), /空闲 300 秒后释放/)
  assert.equal(view.button().props.disabled, false)
  await view.button().props.onClick()
  await view.settle()
  assert.equal(count, 1)
  assert.match(view.text(), /模型已手动释放/)
  assert.equal(view.button().props.disabled, true)
  view.dispose()
})

test('loading, inference and queued work disable manual release', async () => {
  for (const runtime of [
    { ...ready, state: 'loading', active_test_id: 'batch', can_release: false },
    { ...ready, state: 'running', active_test_id: 'batch', can_release: false },
    { ...ready, waiting_tests: 1, can_release: false },
  ]) {
    const view = runtimeView({ modelTestRuntime: async () => runtime })
    await view.settle()
    assert.equal(view.button().props.disabled, true)
    view.dispose()
  }
})

test('poll errors mark stale data and recovery updates idle release state', async () => {
  let fail = false
  const view = runtimeView({ modelTestRuntime: async () => { if (fail) throw new Error('offline'); return { ...released, release_reason: 'idle' } } })
  await view.settle()
  assert.match(view.text(), /空闲模型已自动释放/)
  fail = true
  await view.tick()
  assert.match(view.text(), /模型驻留状态暂不可用/)
  assert.equal(view.button().props.disabled, true)
  fail = false
  await view.tick()
  assert.match(view.text(), /空闲模型已自动释放/)
  view.dispose()
})

test('late pre-release status cannot restore a released model; hidden pages skip polling', async () => {
  let calls = 0; let resolve
  const view = runtimeView({
    modelTestRuntime: async () => { calls++; return calls === 2 ? new Promise(done => { resolve = done }) : ready },
    releaseModelTestRuntime: async () => released,
  })
  await view.settle()
  await view.tick()
  view.button().props.onClick()
  await view.settle()
  resolve(ready)
  await view.settle()
  assert.match(view.text(), /模型已手动释放/)
  view.hide(true)
  await view.tick()
  assert.equal(calls, 2)
  view.dispose()
})

test('closing the view aborts only status requests', async () => {
  let signal
  const view = runtimeView({ modelTestRuntime: async value => { signal = value; return ready } })
  await view.settle()
  assert.equal(signal.aborted, false)
  view.dispose()
  assert.equal(signal.aborted, true)
})
