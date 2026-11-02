import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/taskOrder.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const module = { exports: {} }
vm.runInNewContext(compiled, { exports: module.exports })
const { reorderVisibleGroupTasks, sortOverviewTasks } = module.exports

test('overview keeps groups together and sorts by creation rather than manual or completion order', () => {
  const groups = [{ id: 'smoke' }, { id: 'fire' }]
  const tasks = [
    { id: 'fire-new', group_id: 'fire', created_at: '2026-10-30T10:00:00', display_order: 0 },
    { id: 'smoke-old', group_id: 'smoke', created_at: '2026-10-28T10:00:00', finished_at: '2026-11-02T10:00:00' },
    { id: 'ungrouped', group_id: null, created_at: '2026-11-02T10:00:00' },
    { id: 'smoke-new', group_id: 'smoke', created_at: '2026-10-29T10:00:00', display_order: 20 },
    { id: 'fire-old', group_id: 'fire', created_at: '2026-10-27T10:00:00' },
  ]
  const originalIds = tasks.map((task) => task.id)
  const sorted = sortOverviewTasks(tasks, groups)
  assert.deepEqual(Array.from(sorted, (task) => task.id), ['smoke-new', 'smoke-old', 'fire-new', 'fire-old', 'ungrouped'])
  assert.deepEqual(tasks.map((task) => task.id), originalIds)
  // A status/metric refresh must not move tasks between groups or change ordering.
  assert.deepEqual(Array.from(sortOverviewTasks(tasks.map((task) => ({ ...task, status: 'running', updated_at: '2026-11-03T12:00:00' })), groups), (task) => task.id), Array.from(sorted, (task) => task.id))
})

test('overview compares historical UTC timestamps and explicit offsets as instants', () => {
  const tasks = [
    { id: 'old', created_at: '2026-11-02T08:30:00+08:00' },
    { id: 'new', created_at: '2026-11-02 01:30:00' },
    { id: 'middle', created_at: '2026-11-02T01:00:00Z' },
  ]
  assert.deepEqual(Array.from(sortOverviewTasks(tasks, []), (task) => task.id), ['new', 'middle', 'old'])
})

test('overview retains legacy tasks with unknown groups or missing timestamps and resolves ties consistently', () => {
  const tasks = [
    { id: 'z', group_id: 'deleted-group', created_at: 'invalid' },
    { id: 'known', group_id: 'known', created_at: null },
    { id: 'b', group_id: null, created_at: '2026-11-02T01:00:00Z' },
    { id: 'a', created_at: '2026-11-02T01:00:00' },
    { id: 'missing' },
  ]
  assert.deepEqual(Array.from(sortOverviewTasks(tasks, [{ id: 'known' }]), (task) => task.id), ['known', 'a', 'b', 'missing', 'z'])
  assert.deepEqual(Array.from(sortOverviewTasks([...tasks].reverse(), [{ id: 'known' }]), (task) => task.id), ['known', 'a', 'b', 'missing', 'z'])
})

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
