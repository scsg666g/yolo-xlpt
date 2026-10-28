import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

function loadModule(name, globals = {}) {
  const source = readFileSync(new URL(`../src/${name}.ts`, import.meta.url), 'utf8')
  const compiled = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText
  const module = { exports: {} }
  vm.runInNewContext(compiled, {
    exports: module.exports,
    require: (path) => loadModule(path.replace('./', ''), globals),
    ...globals,
  })
  return module.exports
}
const { metricProfile, primaryMetric, bestMetricPoint } = loadModule('metricProfiles')
const { comparisonEvidence } = loadModule('comparisonEvidence')
const { applyHistoryRefresh, bridgeHistoryHead, refreshActiveHistoryTail, mergeHistoryHead, retainedHistorySelection, historyNeedsDetailRefresh } = loadModule('modelTestHistory')

const point = (epoch, values) => ({ epoch, values })

test('all task types choose their own primary metric and best evaluation epoch', () => {
  const cases = [
    ['detect', 'metrics/mAP50-95(B)'], ['segment', 'metrics/mAP50-95(M)'],
    ['classify', 'metrics/accuracy_top1'], ['pose', 'metrics/mAP50-95(P)'],
    ['obb', 'metrics/mAP50-95(B)'],
  ]
  for (const [task, key] of cases) {
    assert.equal(primaryMetric(task).key, key)
    assert.ok(metricProfile(task).summaryKeys.includes(key))
    const samples = [point(1, { [key]: 0.4 }), point(2, { [key]: 0.9 }), point(3, { [key]: Number.NaN })]
    assert.equal(bestMetricPoint(samples, task).epoch, 2)
  }
  // Segmentation's best mask epoch is different from its best box epoch.
  const segmentation = [
    point(1, { 'metrics/mAP50-95(M)': 0.8, 'metrics/mAP50-95(B)': 0.3 }),
    point(2, { 'metrics/mAP50-95(M)': 0.4, 'metrics/mAP50-95(B)': 0.9 }),
  ]
  assert.equal(bestMetricPoint(segmentation, 'segment').epoch, 1)
  assert.equal(bestMetricPoint([point(1, { 'metrics/mAP50-95(B)': 0.8 })], 'segment'), undefined)
  assert.equal(metricProfile('unknown'), undefined)
  assert.equal(metricProfile('__proto__'), undefined)
  assert.equal(bestMetricPoint([point(1, { 'metrics/accuracy_top1': null })], 'classify'), undefined)
})

test('background history refresh preserves loaded pages, cursor selection and updated head', () => {
  const initial = Array.from({ length: 100 }, (_, index) => ({ id: `test-${index}`, status: 'queued' }))
  const next = mergeHistoryHead(initial, [{ id: 'new-test', status: 'running' }, { id: 'test-0', status: 'completed' }])
  assert.equal(next.length, 101)
  assert.equal(next[0].id, 'new-test')
  assert.equal(next[1].status, 'completed')
  assert.equal(next[100].id, 'test-99')
  assert.equal(retainedHistorySelection('test-80', next), 'test-80')
  assert.equal(retainedHistorySelection('deleted', next), 'new-test')
  assert.equal(retainedHistorySelection(null, []), null)
})

test('queued-to-running changes and terminal receipts invalidate stale cached model-test details', () => {
  assert.equal(historyNeedsDetailRefresh({ status: 'running' }, { status: 'queued' }), true)
  assert.equal(historyNeedsDetailRefresh({ status: 'completed', finished_at: 'now' }, { status: 'running' }), true)
  assert.equal(historyNeedsDetailRefresh({ status: 'completed', finished_at: 'new' }, { status: 'completed', finished_at: 'old' }), true)
  assert.equal(historyNeedsDetailRefresh({ status: 'completed', finished_at: 'now' }, { status: 'completed', finished_at: 'now' }), false)
})

function runEvidence(overrides = {}, snapshot = {}) {
  return {
    taskType: 'detect', nodeId: 'node-a', dataPath: '/datasets/detection/data.yaml',
    parameters: { dataset_snapshot: {
      version: 1, source: 'managed_catalog', verification: 'catalog_snapshot',
      configuration_scope: 'launch_configuration', node_id: 'node-a', task_type: 'detect',
      fingerprint: 'a'.repeat(64), config_sha256: 'b'.repeat(64),
      checked_at: '2026-10-28T12:00:00Z', classes: ['cat'], stats: { train_images: 10, val_images: 2 },
      ...snapshot,
    } },
    ...overrides,
  }
}

test('comparison requires compatible historical evidence rather than matching path text', () => {
  const original = runEvidence()
  assert.equal(comparisonEvidence([original, runEvidence()]).comparable, true)
  for (const other of [
    runEvidence({ nodeId: 'node-b' }), runEvidence({ dataPath: '/other/data.yaml' }),
    runEvidence({}, { fingerprint: 'c'.repeat(64) }), runEvidence({}, { configuration_scope: 'resume_request' }),
    runEvidence({}, { classes: ['dog'] }), runEvidence({}, { stats: { train_images: 11, val_images: 2 } }),
    runEvidence({}, { verification: 'config_only' }), runEvidence({ parameters: {} }),
    runEvidence({ taskType: 'classify' }), runEvidence({}, { task_type: 'segment' }),
  ]) assert.equal(comparisonEvidence([original, other]).comparable, false)
  // Same pathname on two servers is insufficient even if checksums happen to match.
  assert.equal(comparisonEvidence([original, runEvidence({ nodeId: 'node-b' }, { node_id: 'node-b' })]).comparable, false)
  assert.match(comparisonEvidence([original, runEvidence({ parameters: {} })]).reason, /未核验/)
  assert.match(comparisonEvidence([original, runEvidence()]).reason, /仍可能/)
})

function apiModule(fetch, timers = {}) {
  return loadModule('api', { fetch, AbortController, TypeError, window: { setTimeout, clearTimeout, ...timers } }).api
}

test('concurrent clicks share one write but a subsequent deliberate call remains possible', async () => {
  let requests = 0
  let resolveFetch
  const api = apiModule(() => {
    requests++
    return new Promise((resolve) => { resolveFetch = resolve })
  })
  const first = api.clone('task-1')
  const second = api.clone('task-1')
  assert.equal(requests, 1)
  resolveFetch({ ok: true, status: 200, json: async () => ({ id: 'clone-1' }) })
  assert.equal((await first).id, 'clone-1')
  assert.equal((await second).id, 'clone-1')
  const third = api.clone('task-1')
  assert.equal(requests, 2)
  resolveFetch({ ok: true, status: 200, json: async () => ({ id: 'clone-2' }) })
  assert.equal((await third).id, 'clone-2')
})

test('write timeout and network errors explain uncertain acceptance and never auto-resubmit', async () => {
  let requests = 0
  const api = apiModule((path, { signal }) => {
    requests++
    return new Promise((resolve, reject) => signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError'))))
  }, { setTimeout: (callback) => setTimeout(callback, 5) })
  await assert.rejects(api.start('task-1'), /后台可能已经受理/)
  assert.equal(requests, 1)
  const disconnected = apiModule(async () => { throw new TypeError('Network failed') })
  await assert.rejects(disconnected.create({ name: 'new task' }), /先刷新/)
})

test('previously unbounded task reads abort when the network stalls', async () => {
  const api = apiModule((path, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
  }), { setTimeout: (callback) => setTimeout(callback, 5) })
  await assert.rejects(api.tasks(), /请求超时/)
})


test('filtered history removes active records that leave the filter, including older cursor pages', async () => {
  const initial = Array.from({ length: 100 }, (_, index) => ({ id: `test-${index}`, status: 'queued' }))
  const head = initial.slice(0, 50)
  let inflight = 0; let maxInflight = 0
  const refreshed = await refreshActiveHistoryTail(initial, head, new Set(['queued', 'running']), async (id) => {
    inflight++; maxInflight = Math.max(maxInflight, inflight)
    await Promise.resolve()
    inflight--
    if (id === 'test-50') throw Object.assign(new Error('gone'), { status: 404 })
    return { id, status: id === 'test-51' ? 'running' : 'queued' }
  })
  const next = applyHistoryRefresh(initial, head, refreshed, 'queued')
  assert.equal(maxInflight, 4)
  assert.equal(refreshed.updates.length + refreshed.removed.length, 8)
  assert.equal(next.length, 98)
  assert.equal(next.some((item) => item.id === 'test-50' || item.id === 'test-51'), false)
  assert.equal(retainedHistorySelection('test-80', next), 'test-80')
  assert.notEqual(refreshed.next, 0)
})

test('a large incoming history head bridges the gap while preserving the old tail cursor boundary', async () => {
  const current = Array.from({ length: 100 }, (_, index) => ({ id: `old-${index}`, status: 'completed' }))
  const fresh = Array.from({ length: 60 }, (_, index) => ({ id: `new-${index}`, status: 'queued' }))
  let requests = 0
  const head = await bridgeHistoryHead(current, { items: fresh.slice(0, 50), next_cursor: 'new-49' }, async (cursor) => {
    requests++
    assert.equal(cursor, 'new-49')
    return { items: [...fresh.slice(50), ...current.slice(0, 40)], next_cursor: 'old-39' }
  })
  const merged = applyHistoryRefresh(current, head, { updates: [], removed: [] })
  assert.equal(requests, 1)
  assert.equal(merged.length, 160)
  assert.equal(merged[59].id, 'new-59')
  assert.equal(merged[159].id, 'old-99')
  assert.equal(new Set(merged.map((item) => item.id)).size, merged.length)
})


test('local GPU mapping displays only coherent launch evidence and never guesses historical mappings', () => {
  const { localDeviceBindingLabel } = loadModule('localDeviceBinding')
  const binding = { version: 1, requested_device: '1,0', cuda_visible_devices: 'GPU-bbbb,GPU-aaaa', cuda_device_order: 'PCI_BUS_ID',
    devices: [{ physical_index: 1, logical_index: 0, uuid: 'GPU-bbbb' }, { physical_index: 0, logical_index: 1, uuid: 'GPU-aaaa' }] }
  assert.match(localDeviceBindingLabel({ local_device_binding: binding }), /物理 GPU 1 → 训练逻辑 GPU 0/)
  assert.match(localDeviceBindingLabel({ local_device_binding: binding }), /CUDA_VISIBLE_DEVICES=GPU-bbbb,GPU-aaaa/)
  assert.equal(localDeviceBindingLabel(), null)
  assert.equal(localDeviceBindingLabel({ device: '0' }), null)
  assert.equal(localDeviceBindingLabel({ local_device_binding: { ...binding, requested_device: '0,1' } }), null)
  assert.equal(localDeviceBindingLabel({ local_device_binding: { ...binding, devices: [binding.devices[0], binding.devices[0]] } }), null)
})
