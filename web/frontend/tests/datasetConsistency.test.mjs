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
function page(api, { component = 'Datasets', props = {} } = {}) {
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
  const document = { hidden: false, body: { style: { overflow: '' } }, activeElement: null, addEventListener: (name, work) => listeners.set(`document:${name}`, work), removeEventListener: name => listeners.delete(`document:${name}`) }
  const window = { addEventListener: (name, work) => listeners.set(`window:${name}`, work), removeEventListener: name => listeners.delete(`window:${name}`) }
  const module = { exports: {} }
  vm.runInNewContext(compiled(`../src/components/${component}.tsx`), {
    exports: module.exports, AbortController, Error, Map, Set, Date, URLSearchParams, document, window,
    crypto: { randomUUID: () => `request-${++timerId}` },
    setTimeout: work => { const id = ++timerId; timers.set(id, work); return id }, clearTimeout: id => timers.delete(id),
    require(name) {
      if (name === 'react') return react
      if (name === 'react/jsx-runtime') return { jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) }
      if (name === '../api') return { api, formatBytes: String, statusLabel: String }
      if (name === '../useModalFocus') return { useModalFocus: () => react.useRef(null) }
      if (name === '../datasetOperationState') return { datasetOperationWait: () => null }
      return new Proxy({}, { get: (_target, key) => String(key) })
    },
  })
  function render() { index = 0; tree = module.exports[component](props); for (const effect of effects.splice(0)) effect(); return tree }
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

function visibleText(node) {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node !== 'object') return String(node)
  if (Array.isArray(node)) return node.map(visibleText).join('')
  return visibleText(node.props?.children)
}

test('referenced external cards open registration removal and in-page adoption during training', async () => {
  const training = { ...dataset, ownership: 'external', can_delete: false, can_unregister: true,
    references: [{ id: 'task', name: '训练中', status: 'running' }] }
  const view = page(baseApi({ datasetEntries: async () => [training] }))
  await view.settle()
  const remove = view.find(item => item.props?.['aria-label'] === '删除登记：训练数据')
  assert.equal(remove.props.disabled, false)
  assert.equal(view.find(item => item.props?.['aria-label'] === '永久删除文件：训练数据'), null)
  remove.props.onClick(); await view.settle()
  let modal = view.find(item => item.type === 'DatasetRegistrationModal')
  assert.equal(modal.props.mode, 'unregister')
  modal.props.onClose(); await view.settle()
  view.find(item => item.type === 'button' && visibleText(item) === '接管文件').props.onClick()
  await view.settle()
  modal = view.find(item => item.type === 'DatasetRegistrationModal')
  assert.equal(modal.props.mode, 'adopt')
  view.dispose()
})

test('registration removal submits once, preserves errors in the modal and never calls file deletion', async () => {
  let calls = 0; let done = 0; let finish
  const view = page({ unregisterDataset: async id => {
    assert.equal(id, dataset.id); calls++
    if (calls === 1) throw new Error('后台操作尚未结束')
    await new Promise(resolve => { finish = resolve })
  }, deleteDatasetEntry: () => { throw new Error('must not delete files') } }, {
    component: 'DatasetRegistrationModal', props: { dataset, mode: 'unregister', onClose() {}, onDone() { done++ } },
  })
  await view.settle()
  assert.match(visibleText(view.find(item => item.props?.role === 'alertdialog')), /保留原始文件/)
  const form = () => view.find(item => item.type === 'form')
  form().props.onSubmit({ preventDefault() {} }); await view.settle()
  assert.match(visibleText(view.find(item => item.props?.role === 'alert')), /后台操作尚未结束/)
  assert.equal(done, 0)
  form().props.onSubmit({ preventDefault() {} })
  form().props.onSubmit({ preventDefault() {} })
  await view.settle(); assert.equal(calls, 2)
  finish(); await view.settle(); assert.equal(done, 1)
  view.dispose()
})

test('adoption modal requires the exact path and sends permission changes without file deletion', async () => {
  const calls = []; let done = 0
  const view = page({ adoptDataset: async (...args) => { calls.push(args) } }, {
    component: 'DatasetRegistrationModal', props: { dataset, mode: 'adopt', onClose() {}, onDone() { done++ } },
  })
  await view.settle()
  const submit = () => view.find(item => item.props?.type === 'submit')
  const form = () => view.find(item => item.type === 'form')
  assert.equal(submit().props.disabled, true)
  form().props.onSubmit({ preventDefault() {} }); await view.settle(); assert.equal(calls.length, 0)
  view.find(item => item.type === 'input').props.onChange({ target: { value: dataset.directory } })
  await view.settle(); assert.equal(submit().props.disabled, false)
  form().props.onSubmit({ preventDefault() {} }); await view.settle()
  assert.deepEqual(calls, [[dataset.id, dataset.directory]]); assert.equal(done, 1)
  view.dispose()
})

test('a managed dataset exposes the deletion blocker and links to its exact operation', async () => {
  const reason = '数据集正在重新加载，请先取消该操作'
  const blocked = { ...dataset, can_delete: false, delete_block_reason: reason, can_unregister: false, unregister_block_reason: reason, blocking_operation_id: 'refresh-a', blocking_operation_kind: 'refresh' }
  const view = page(baseApi({ datasetEntries: async () => [blocked], datasetOperation: async () => ({ id: 'refresh-a', status: 'running', kind: 'refresh' }) }))
  await view.settle()
  const warning = view.find(item => item.props?.className === 'dataset-delete-block')
  assert.ok(warning)
  assert.match(visibleText(warning), /登记操作暂不可用/)
  assert.ok(visibleText(warning).includes(reason), 'the reason must be visible without hovering a disabled button')
  assert.equal(view.find(item => item.props?.className === 'button danger compact').props.disabled, true)
  view.find(item => item.type === 'button' && visibleText(item) === '查看并处理相关操作').props.onClick()
  await view.settle()
  const drawer = view.find(item => item.type === 'DatasetOperationDrawer')
  assert.equal(drawer.props.initialTarget.operationId, 'refresh-a')
  assert.equal(drawer.props.initialTarget.nodeId, 'local')
  view.dispose()
})

test('deletion confirmation identifies the server and complete directory for matching dataset names', async () => {
  const remote = { ...dataset, name: '打电话', node_id: 'node-remote', node_name: '远程服务器', directory: '/training/datasets/calls' }
  const view = page(baseApi({ datasetEntries: async () => [remote], datasetDeletePreview: async () => remote }))
  await view.settle()
  view.find(item => item.props?.['aria-label'] === '永久删除文件：打电话').props.onClick({ currentTarget: { isConnected: true, focus() {} } })
  await view.settle()
  const dialog = view.find(item => item.props?.role === 'alertdialog')
  assert.ok(visibleText(dialog).includes('远程服务器'))
  assert.ok(visibleText(dialog).includes('node-remote'))
  assert.ok(visibleText(dialog).includes('/training/datasets/calls'))
  view.dispose()
})

test('deletion acceptance stays distinct from completion and polls wait, verification, cleanup, failure and recovery', async () => {
  let status = 'pending'; let stage = 'delete_verify'; let wait = true; let unavailable = false
  const operation = () => ({ id: 'delete-a', kind: 'delete', phase: 'remove', status, node_id: 'local',
    error: status === 'failed' ? '文件指纹变化' : null, waiting_path_lock: wait,
    progress: { stage, phase: 'remove', processed: 1, total: 2, unit: 'files' } })
  const current = () => status === 'completed' ? [] : [{ ...dataset, can_delete: false, blocking_operation_id: 'delete-a', blocking_operation_kind: 'delete' }]
  let submitted = false
  const view = page(baseApi({
    datasetEntries: async () => submitted ? current() : [dataset],
    datasetDeletePreview: async () => dataset,
    deleteDatasetEntry: async () => { submitted = true; return operation() },
    datasetOperation: async () => { if (unavailable) throw new Error('offline'); return operation() },
    datasetOperationSummary: async () => ({ revision: status, active: submitted ? 1 : 0, failed: status === 'failed' ? 1 : 0, nodes: [] }),
  }))
  await view.settle()
  view.find(item => item.props?.['aria-label'] === '永久删除文件：训练数据').props.onClick({ currentTarget: { isConnected: true, focus() {} } })
  await view.settle()
  view.find(item => item.type === 'button' && visibleText(item) === '确认永久删除').props.onClick()
  await view.settle(); await view.settle()
  const receipt = () => view.find(item => item.props?.className === 'dataset-deletion-receipt')
  const progress = () => view.find(item => item.type === 'DatasetOperationProgress' && item.props?.operation.id === 'delete-a')
  assert.ok(receipt())
  assert.ok(!visibleText(receipt()).includes('删除完成'))
  assert.equal(progress().props.operation.status, 'pending')
  assert.equal(progress().props.operation.waiting_path_lock, true)
  status = 'running'; wait = false
  await view.tick()
  assert.equal(progress().props.operation.progress.stage, 'delete_verify')
  stage = 'remove'; await view.tick()
  assert.equal(progress().props.operation.progress.stage, 'remove')
  unavailable = true; await view.tick()
  assert.match(visibleText(receipt()), /状态暂未更新/)
  assert.equal(progress().props.operation.status, 'running')
  unavailable = false; status = 'failed'; await view.tick()
  assert.match(visibleText(receipt()), /删除失败：文件指纹变化/)
  assert.ok(!visibleText(receipt()).includes('删除完成'))
  status = 'completed'; await view.tick()
  assert.match(visibleText(receipt()), /后台已确认文件清理和数据集登记删除完成/)
  assert.equal(view.find(item => item.props?.className === 'dataset-card'), null)
  view.dispose()
})

test('the related operation opens by identity even when older than the first history page', async () => {
  const requested = []; const controls = []; let changes = 0
  const job = { id: 'old-refresh', name: '打电话', status: 'failed', kind: 'refresh', node_id: 'remote-node', phase: 'export', result: {}, error: '校验失败', can_cancel: true }
  const view = page(baseApi({
    datasetOperationPage: async params => { requested.push(params); return { items: [], total: 0, page: 1, page_size: 20, snapshot: '2026-10-30T00:00:00Z' } },
    datasetOperation: async id => { assert.equal(id, job.id); return job },
    controlDatasetOperation: async (id, action) => { controls.push([id, action]); return { ...job, status: 'cancelled' } },
  }), { component: 'DatasetOperationDrawer', props: {
    initialTarget: { operationId: job.id, name: job.name, nodeId: job.node_id }, servers: [], summary: null, onClose() {}, onChanged() { changes++ },
  } })
  await view.settle()
  assert.equal(requested[0].get('node_id'), 'remote-node')
  assert.equal(requested[0].get('q'), '打电话')
  const toggle = view.find(item => item.props?.className === 'dataset-history-row-toggle')
  assert.equal(toggle.props['aria-expanded'], true)
  assert.ok(visibleText(toggle).includes('打电话'))
  view.find(item => item.type === 'button' && visibleText(item) === '取消操作').props.onClick()
  await view.settle()
  assert.deepEqual(controls, [['old-refresh', 'cancel']])
  assert.equal(changes, 1)
  view.dispose()
})

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
    exports: module.exports, require: (name) => {
      if (name === '../metricProfiles') {
        const profiles = { exports: {} }
        vm.runInNewContext(compiled('../src/metricProfiles.ts'), { exports: profiles.exports })
        return profiles.exports
      }
      return new Proxy({}, { get: (_target, key) => String(key) })
    },
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
