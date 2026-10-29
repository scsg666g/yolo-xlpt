import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

// Run actual component effects with deterministic network and timers. JSX stays
// inspectable and no browser, service, user data or generated disk files are used.
function historyView(component, api, query = 'runs=run-a') {
  const slots = []; const effects = []; const timers = new Map()
  let index = 0; let timerId = 0; let tree
  const same = (left, right) => left && right && left.length === right.length && left.every((value, i) => Object.is(value, right[i]))
  const react = {
    useState(initial) {
      const i = index++
      if (!slots[i]) slots[i] = { value: typeof initial === 'function' ? initial() : initial }
      return [slots[i].value, value => { slots[i].value = typeof value === 'function' ? value(slots[i].value) : value }]
    },
    useRef(value) { const i = index++; return slots[i] ||= { current: value } },
    useMemo(work, deps) { const i = index++; if (!slots[i] || !same(slots[i].deps, deps)) slots[i] = { value: work(), deps }; return slots[i].value },
    useCallback(work, deps) { return react.useMemo(() => work, deps) },
    useEffect(work, deps) {
      const i = index++
      if (!slots[i] || !same(slots[i].deps, deps)) effects.push({ index: i, deps, work })
    },
  }
  const window = {
    localStorage: { getItem: () => null, setItem: () => {} },
    setTimeout(work) { const id = ++timerId; timers.set(id, work); return id }, clearTimeout: id => timers.delete(id),
    addEventListener() {}, removeEventListener() {}, location: { search: `?${query}` },
  }
  const document = { activeElement: null, hidden: false, addEventListener() {}, removeEventListener() {} }
  const searchParams = new URLSearchParams(query)
  const cache = new Map()
  const jsx = { jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) }
  const mocked = new Proxy({}, { get: (_target, key) => String(key) })
  function load(name) {
    if (name === 'react') return react
    if (name === 'react/jsx-runtime') return jsx
    if (name === '../api') return { api, formatBytes: String, statusLabel: String, metricValue: () => '0.5000' }
    if (name === 'react-router-dom') return { useSearchParams: () => [searchParams, () => {}] }
    if (name === '../preferences') return { useCompareLimit: () => 4, readStoredCompareRuns: () => [], writeStoredCompareRuns() {} }
    let relative
    if (name === './ModelTestShared') relative = 'components/ModelTestShared.tsx'
    else if (['../modelTestHistory', '../metricProfiles', '../datasetSnapshot', '../comparisonEvidence', '../chartMetrics'].includes(name)) relative = `${name.slice(3)}.ts`
    else if (name === './datasetSnapshot') relative = 'datasetSnapshot.ts'
    else if (name === './types') return {}
    else if (name === component) relative = `components/${component}.tsx`
    else return mocked
    if (cache.has(relative)) return cache.get(relative)
    const module = { exports: {} }; cache.set(relative, module.exports)
    const source = readFileSync(new URL(`../src/${relative}`, import.meta.url), 'utf8')
    const compiled = ts.transpileModule(source, { compilerOptions: {
      target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX,
    } }).outputText
    vm.runInNewContext(compiled, { exports: module.exports, require: load, window, document, AbortController, URL, URLSearchParams })
    return module.exports
  }
  const Component = load(component)[component]
  function render() {
    index = 0
    tree = Component({ taskId: 'task-a', taskName: '测试任务', onClose() {} })
    const pending = effects.splice(0)
    for (const effect of pending) slots[effect.index]?.cleanup?.()
    for (const effect of pending) slots[effect.index] = { deps: effect.deps, cleanup: effect.work() }
  }
  const findAll = predicate => {
    const found = []
    function visit(node) {
      if (!node || typeof node !== 'object') return
      if (Array.isArray(node)) { node.forEach(visit); return }
      if (predicate(node)) found.push(node)
      visit(node.props?.children)
    }
    visit(tree); return found
  }
  const settle = async () => { for (let pass = 0; pass < 5; pass++) { for (let i = 0; i < 50; i++) await Promise.resolve(); render() } }
  render()
  return {
    findAll, settle,
    async click(node) { assert.ok(node, 'control exists'); node.props.onClick(); render(); await settle() },
    async change(node, value) { assert.ok(node, 'control exists'); node.props.onChange({ target: { value } }); render(); await settle() },
    async tick() { const work = [...timers.values()]; timers.clear(); work.forEach(next => next()); await settle() },
    dispose() { for (const item of slots) item?.cleanup?.() },
  }
}
function entry(index, status = 'completed') {
  return { id: `test-${index}`, status, task_id: 'task-a', run_id: 'run-a', confidence: 0.25, iou: 0.7,
    model_source_type: 'training_artifact', task_type: 'detect', error: null, files: [],
    model: { weight_name: 'best.pt', device: 'cpu', imgsz: 320, attempt_number: 1 },
  }
}
function fixture() {
  const entries = Array.from({ length: 100 }, (_, i) => entry(i, i === 0 || i === 80 ? 'queued' : 'completed'))
  return {
    entries,
    api: {
      tasks: async () => [], taskGroups: async () => [], computeNodes: async () => [],
      modelAssets: async () => ({ items: [] }),
      globalModelTests: async (_signal, cursor) => ({ items: entries.slice(cursor ? 50 : 0, cursor ? 100 : 50).map(item => ({ ...item })), next_cursor: cursor ? null : 'tail' }),
      modelTests: async (_task, signal, cursor) => ({ items: entries.slice(cursor ? 50 : 0, cursor ? 100 : 50).map(item => ({ ...item })), next_cursor: cursor ? null : 'tail' }),
      modelTest: async id => ({ ...entries.find(item => item.id === id) }),
      taskModelTest: async (_task, id) => ({ ...entries.find(item => item.id === id) }),
    },
  }
}
const rows = view => view.findAll(node => node.type === 'button' && node.props.className?.startsWith('model-test-history-item'))
const workspace = view => view.findAll(node => node.type?.name === 'TestWorkspace')[0]

for (const component of ['ModelTests', 'ModelTestModal']) {
  test(`${component}: quiet polling keeps 100 loaded records and the selected older active detail advances`, async () => {
    const { entries, api } = fixture()
    const view = historyView(component, api)
    try {
      await view.settle()
      if (component === 'ModelTests') await view.click(view.findAll(node => node.props.role === 'tab')[1])
      assert.equal(rows(view).length, 50)
      await view.click(view.findAll(node => node.type === 'button' && node.props.className?.includes('model-test-history-more'))[0])
      assert.equal(rows(view).length, 100)
      await view.click(rows(view)[80])
      assert.equal(workspace(view).props.test.id, 'test-80')
      assert.equal(workspace(view).props.test.status, 'queued')
      entries[80].status = 'running'
      await view.tick()
      assert.equal(rows(view).length, 100)
      assert.equal(workspace(view).props.test.id, 'test-80')
      assert.equal(workspace(view).props.test.status, 'running')
      entries[80].status = 'completed'; entries[80].finished_at = '2026-10-28T12:00:00Z'
      await view.tick()
      assert.equal(workspace(view).props.test.status, 'completed')
      assert.equal(rows(view).length, 100)
    } finally { view.dispose() }
  })
}


test('global status filter removes records that advance past queued, across loaded cursor pages', async () => {
  const { entries, api } = fixture()
  entries.forEach(item => { item.status = 'queued' })
  api.globalModelTests = async (_signal, cursor, _limit, _source, status) => {
    const matches = entries.filter(item => !status || item.status === status)
    return { items: matches.slice(cursor ? 50 : 0, cursor ? 100 : 50).map(item => ({ ...item })), next_cursor: cursor ? null : 'tail' }
  }
  const view = historyView('ModelTests', api)
  try {
    await view.settle()
    await view.click(view.findAll(node => node.props.role === 'tab')[1])
    const filters = view.findAll(node => node.type === 'select')
    await view.change(filters[1], 'queued')
    await view.click(view.findAll(node => node.type === 'button' && node.props.className?.includes('model-test-history-more'))[0])
    assert.equal(rows(view).length, 100)
    await view.click(rows(view)[51])
    entries[0].status = 'running'
    entries[51].status = 'running'
    await view.tick()
    assert.equal(rows(view).length, 98)
    assert.notEqual(workspace(view).props.test.id, 'test-51')
    assert.equal(workspace(view).props.test.status, 'queued')
  } finally { view.dispose() }
})


test('active comparison details refresh logs without an epoch change, and keep classification metrics', async () => {
  const task = { id: 'task-a', name: '分类模型', task_type: 'classify', model_path: 'models/classifier.pt', data_path: 'datasets/classes',
    epochs: 10, batch: 2, imgsz: 32, device: 'cpu', workers: 0, config: {}, current_run_id: 'run-a', group_id: null }
  const run = { id: 'run-a', task_id: task.id, status: 'running', attempt_number: 1, launch_mode: 'fresh',
    model_path: task.model_path, data_path: task.data_path, parameter_snapshot: {}, updated_at: 'initial',
    completed_epochs: 1, latest_metrics: { 'metrics/accuracy_top1': 0.5 }, progress: 10, execution_node_id: null }
  let log = 'first line'; let logRequests = 0
  const view = historyView('ExperimentCompare', {
    taskGroups: async () => [],
    experiments: async () => ({ items: [{ task: { ...task }, run: { ...run } }, { task: { ...task }, run: { ...run, id: 'run-b', attempt_number: 2 } }], total: 2, limit: 200 }),
    runMetrics: async () => [{ epoch: 1, values: run.latest_metrics }], runArtifacts: async () => [],
    runEffectiveBestWeight: async () => ({}), runEvents: async () => [],
    runLogs: async () => { logRequests++; return { lines: [log], path: null } },
  }, 'runs=run-a,run-b')
  try {
    await view.settle()
    const labels = view.findAll(node => node.type === 'option').map(node => node.props.children)
    assert.ok(labels.includes('Top-1 accuracy'), JSON.stringify({ labels, errors: view.findAll(node => node.props.className === 'alert error').map(node => node.props.children) }))
    assert.equal(labels.includes('mAP50'), false)
    await view.click(view.findAll(node => node.type === 'button' && node.props.className === 'button primary')[0])
    await view.click(view.findAll(node => node.type === 'button' && node.props.className === 'compare-run-details-toggle')[0])
    const detail = () => view.findAll(node => node.type === 'CompareRunDetailModal')[0]
    assert.equal(detail().props.details.logs.lines[0], 'first line')
    assert.equal(logRequests, 1)
    log = 'second line'
    await view.tick()
    assert.equal(detail().props.details.logs.lines[0], 'second line')
    run.status = 'completed'; run.updated_at = 'final'; log = 'final line'
    await view.tick()
    assert.equal(detail().props.run.status, 'completed')
    assert.equal(detail().props.details.logs.lines[0], 'final line')
  } finally { view.dispose() }
})

test('comparison refreshes same-Epoch metric corrections and opening picker fetches new attempts', async () => {
  const task = { id: 'task-a', name: '测试任务', task_type: 'detect', model_path: 'models/base.pt', data_path: 'datasets/data.yaml',
    epochs: 10, batch: 2, imgsz: 32, device: 'cpu', workers: 0, config: {}, current_run_id: 'run-a', group_id: null }
  const run = { id: 'run-a', task_id: task.id, status: 'completed', attempt_number: 1, launch_mode: 'fresh',
    model_path: task.model_path, data_path: task.data_path, parameter_snapshot: {}, updated_at: 'unchanged',
    completed_epochs: 1, latest_metrics: { 'metrics/mAP50(B)': 0.4 }, progress: 100, execution_node_id: null }
  let score = 0.4; let includeNew = false; let metricRequests = 0; let catalogRequests = 0
  const view = historyView('ExperimentCompare', {
    taskGroups: async () => [],
    experiments: async (_offset, _signal, runIds) => {
      if (!runIds) catalogRequests++
      const runs = [run, { ...run, id: 'run-b', attempt_number: 2 },
        ...(includeNew ? [{ ...run, id: 'run-c', attempt_number: 3 }] : [])]
      const matching = runIds ? runs.filter(item => runIds.includes(item.id)) : runs
      return { items: matching.map(item => ({ task: { ...task }, run: { ...item } })), total: matching.length, limit: 200 }
    },
    runMetrics: async () => { metricRequests++; return [{ epoch: 1, values: { 'metrics/mAP50(B)': score } }] },
  }, 'runs=run-a,run-b')
  try {
    await view.settle()
    const chartValue = () => view.findAll(node => node.type === 'LineChartView')[0].props.option.series[0].data[0][1]
    assert.equal(chartValue(), 0.4)
    const previousRequests = metricRequests
    score = 0.7
    await view.tick()
    assert.equal(chartValue(), 0.7)
    assert.ok(metricRequests > previousRequests, 'selected metrics re-fetch even when run metadata is unchanged')
    includeNew = true
    const beforeOpen = catalogRequests
    await view.click(view.findAll(node => node.type === 'button' && node.props.className === 'button primary')[0])
    assert.ok(catalogRequests > beforeOpen, 'opening picker requests the full catalog')
    assert.equal(view.findAll(node => node.props.className === 'compare-run-select').length, 3)
  } finally { view.dispose() }
})
