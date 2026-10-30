import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/taskOrder.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const module = { exports: {} }
vm.runInNewContext(compiled, { exports: module.exports })
const { reorderVisibleGroupTasks } = module.exports

test('reorder stays within the selected group even when other groups interleave', () => {
  const tasks = [
    { id: 'a', group_id: 'one' }, { id: 'x', group_id: 'two' },
    { id: 'b', group_id: 'one' }, { id: 'y', group_id: 'two' },
  ]
  const reordered = reorderVisibleGroupTasks(tasks, tasks, ['one', 'two'], 'b', 'a', false)
  assert.deepEqual(reordered.map((task) => task.id), ['b', 'x', 'a', 'y'])
  assert.deepEqual(tasks.map((task) => task.id), ['a', 'x', 'b', 'y'])
  assert.equal(reorderVisibleGroupTasks(tasks, tasks, ['one', 'two'], 'x', 'a', false), tasks)
})

test('reorder retains hidden tasks and treats unknown groups as ungrouped', () => {
  const tasks = [
    { id: 'a', group_id: 'unknown' }, { id: 'hidden', group_id: null },
    { id: 'b', group_id: null }, { id: 'c', group_id: 'known' },
  ]
  const visible = tasks.filter((task) => task.id !== 'hidden')
  assert.deepEqual(reorderVisibleGroupTasks(tasks, visible, ['known'], 'b', 'a', false).map((task) => task.id), ['b', 'hidden', 'a', 'c'])
  assert.equal(reorderVisibleGroupTasks(tasks, visible, ['known'], 'missing', 'a', false), tasks)
  assert.equal(reorderVisibleGroupTasks(tasks, visible, ['known'], 'b', 'a', true), tasks)
})
