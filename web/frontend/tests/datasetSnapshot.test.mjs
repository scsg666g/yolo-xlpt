import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/datasetSnapshot.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
const module = { exports: {} }
vm.runInNewContext(compiled, { exports: module.exports })
const { datasetTrainingSnapshot } = module.exports

test('training snapshots retain every backend verification status', () => {
  for (const verification of ['catalog_snapshot', 'config_only', 'path_only', 'unavailable']) {
    assert.equal(datasetTrainingSnapshot({ dataset_snapshot: { version: 1, source: 'runtime_config', verification } })?.verification, verification)
  }
})

test('legacy attempts without a dataset snapshot remain unknown', () => {
  assert.equal(datasetTrainingSnapshot({}), null)
  assert.equal(datasetTrainingSnapshot({ dataset_snapshot: { version: 1, source: 'runtime_config', verification: 'unknown' } }), null)
})

test('resume submission evidence remains distinct from verified training input', () => {
  const value = datasetTrainingSnapshot({ dataset_snapshot: { version: 1, source: 'runtime_config',
    verification: 'config_only', configuration_scope: 'resume_request' } })
  assert.equal(value.configuration_scope, 'resume_request')
  assert.equal(value.fingerprint, null)
})
