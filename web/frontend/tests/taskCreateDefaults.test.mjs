import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/taskCreateDefaults.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
const module = { exports: {} }
vm.runInNewContext(compiled, { exports: module.exports, require: () => ({}) })
const { applyLocalResourceDefaults, defaultLocalDevice } = module.exports

const resources = {
  gpus: [{ index: 2 }, { index: 3 }],
  models: [{ path: '/models/baseline.pt' }],
  datasets: [{ path: '/datasets/detect.yaml' }],
  classification_datasets: [{ path: '/datasets/classify' }],
}
const edits = { model_path: false, data_path: false, device: false }

test('local defaults select one GPU and the dataset matching the task type', () => {
  const detect = applyLocalResourceDefaults({ task_type: 'detect', execution_node_id: null, model_path: '', data_path: '', device: '0' }, resources, edits)
  assert.equal(defaultLocalDevice(resources), '2')
  assert.equal(detect.device, '2')
  assert.equal(detect.model_path, '/models/baseline.pt')
  assert.equal(detect.data_path, '/datasets/detect.yaml')
  assert.equal(applyLocalResourceDefaults({ ...detect, data_path: '' , task_type: 'classify' }, resources, edits).data_path, '/datasets/classify')
  assert.equal(defaultLocalDevice({ ...resources, gpus: [] }), 'cpu')
})

test('late local responses cannot mutate remote paths or device, including blank remote fields', () => {
  const remote = { task_type: 'detect', execution_node_id: 'agent-a', model_path: '', data_path: '', device: '0' }
  assert.equal(applyLocalResourceDefaults(remote, resources, edits), remote)
  const filled = { ...remote, model_path: '/remote/model.pt', data_path: '/remote/data.yaml', device: '1' }
  assert.equal(applyLocalResourceDefaults(filled, resources, edits), filled)
})

test('late local responses keep explicitly edited or imported values', () => {
  const local = { task_type: 'detect', execution_node_id: null, model_path: '', data_path: '', device: '3' }
  const result = applyLocalResourceDefaults(local, resources, { model_path: true, data_path: true, device: true })
  assert.equal(result.model_path, '')
  assert.equal(result.data_path, '')
  assert.equal(result.device, '3')
  const selected = applyLocalResourceDefaults({ ...local, model_path: '/selected.pt', data_path: '/selected.yaml' }, resources, edits)
  assert.equal(selected.model_path, '/selected.pt')
  assert.equal(selected.data_path, '/selected.yaml')
})
