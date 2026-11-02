import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

function load(name) {
  const compiled = ts.transpileModule(readFileSync(new URL(`../src/${name}.ts`, import.meta.url), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText
  const module = { exports: {} }
  vm.runInNewContext(compiled, { exports: module.exports, require: path => load(path.replace('./', '')) })
  return module.exports
}
const { overviewGroupLayout } = load('taskOverviewLayout')
const groups = [{ id: 'smoke', name: '抽烟检测' }, { id: 'fire', name: '火检测' }]
const tasks = [
  { id: 'fire-old', group_id: 'fire', created_at: '2026-10-28T01:00:00Z' },
  { id: 'smoke-middle', group_id: 'smoke', created_at: '2026-10-29T01:00:00Z' },
  { id: 'fire-new', group_id: 'fire', created_at: '2026-10-30T01:00:00Z' },
  { id: 'smoke-old', group_id: 'smoke', created_at: '2026-10-28T01:00:00Z' },
  { id: 'smoke-new', group_id: 'smoke', created_at: '2026-10-30T01:00:00Z' },
]
const plain = value => JSON.parse(JSON.stringify(value))

test('each group has its own column, heading and chronological chain without cross-group links', () => {
  const layout = overviewGroupLayout(tasks, groups)
  const smoke = layout.filter(item => item.task.group_id === 'smoke')
  const fire = layout.filter(item => item.task.group_id === 'fire')
  assert.equal(new Set(smoke.map(item => item.position.x)).size, 1)
  assert.equal(new Set(fire.map(item => item.position.x)).size, 1)
  assert.ok(smoke[0].position.x < fire[0].position.x)
  assert.equal(smoke[0].position.y, fire[0].position.y)
  assert.ok(smoke[0].position.y < smoke[1].position.y && smoke[1].position.y < smoke[2].position.y)
  assert.deepEqual(plain(layout.map(item => [item.task.id, item.previousId])), [
    ['smoke-new', null], ['smoke-middle', 'smoke-new'], ['smoke-old', 'smoke-middle'],
    ['fire-new', null], ['fire-old', 'fire-new'],
  ])
  assert.equal(smoke[0].groupName, '抽烟检测')
  assert.equal(smoke[0].groupCount, 3)
  assert.equal(layout.filter(item => item.firstInGroup).length, 2)
})

test('deleting a middle or first task closes the gap and reconnects adjacent tasks', () => {
  const before = overviewGroupLayout(tasks, groups)
  const after = overviewGroupLayout(tasks.filter(task => task.id !== 'smoke-middle'), groups)
  const old = after.find(item => item.task.id === 'smoke-old')
  assert.deepEqual(plain(old.position), plain(before.find(item => item.task.id === 'smoke-middle').position))
  assert.equal(old.previousId, 'smoke-new')
  assert.equal(old.groupCount, 2)
  const deletedFirst = overviewGroupLayout(tasks.filter(task => task.id !== 'smoke-new'), groups)
  const first = deletedFirst.find(item => item.task.id === 'smoke-middle')
  assert.deepEqual(plain(first.position), plain(before[0].position))
  assert.equal(first.firstInGroup, true)
  assert.equal(first.previousId, null)
  assert.deepEqual(plain(after.filter(item => item.task.group_id === 'fire').map(item => item.position)), plain(before.filter(item => item.task.group_id === 'fire').map(item => item.position)))
})

test('new and moved tasks reflow their own group, empty groups are omitted and old unknown groups remain visible', () => {
  const items = tasks.map(task => task.id === 'smoke-middle' ? { ...task, group_id: 'fire' } : task)
  items.push({ id: 'newest', group_id: 'smoke', created_at: '2026-11-01T01:00:00Z' })
  items.push({ id: 'legacy', group_id: 'removed', created_at: null })
  const layout = overviewGroupLayout(items, [...groups, { id: 'empty', name: '空组' }])
  assert.equal(layout[0].task.id, 'newest')
  assert.equal(layout.find(item => item.task.id === 'smoke-old').previousId, 'smoke-new')
  const moved = layout.find(item => item.task.id === 'smoke-middle')
  assert.equal(moved.groupName, '火检测')
  assert.equal(moved.previousId, 'fire-new')
  assert.equal(layout.at(-1).groupName, '未分组')
  assert.equal(layout.filter(item => item.firstInGroup).length, 3)
  assert.deepEqual(plain(overviewGroupLayout([], groups)), [])
})
