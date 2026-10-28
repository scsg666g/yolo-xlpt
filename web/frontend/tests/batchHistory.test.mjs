import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import ts from 'typescript'

const source = readFileSync(new URL('../src/batchHistory.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
const module = { exports: {} }
new Function('exports', 'module', compiled)(module.exports, module)
const { batchHistoryForRun, batchPath } = module.exports

test('successive live updates keep the full reduction path', () => {
  assert.equal(batchPath(10, [10, 5]).cardLabel, 'Batch 10 → 5')
  assert.equal(batchPath(10, [10, 5, 2]).cardLabel, 'Batch 10 → 5 → 2')
  assert.equal(batchPath(10, [10, 5, 2]).submittedBatch, 10)
})

test('old tasks without a parsed path show only the submitted setting', () => {
  const display = batchPath(10, undefined)
  assert.equal(display.cardLabel, '设置 Batch 10')
  assert.equal(display.detailLabel, '未记录自动调整')
})

test('an older Attempt never inherits the current Attempt path', () => {
  const task = { current_run_id: 'current', batch_history: [10, 5, 2] }
  const prior = { id: 'prior', batch_history: [20, 10] }
  assert.deepEqual(batchHistoryForRun(task, prior), [20, 10])
  assert.deepEqual(batchHistoryForRun(task, { id: 'prior' }), [])
  assert.deepEqual(batchHistoryForRun(task, { id: 'current', parameter_snapshot: { batch: 10 }, batch_history: [10, 5] }), [10, 5, 2])
  assert.deepEqual(batchHistoryForRun(task, { id: 'current', parameter_snapshot: { batch: 20 }, batch_history: [] }), [])
})
