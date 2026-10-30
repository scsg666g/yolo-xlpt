import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/components/Datasets.tsx', import.meta.url), 'utf8')
const start = source.indexOf('  async function copyDatasetPath(path: string) {')
const end = source.indexOf('\n  const loadDatasets =', start)
assert.ok(start > 0 && end > start, 'the component still defines its copy interaction')
const compiled = ts.transpileModule('async function exercise(path: string) {\n' + source.slice(start, end) + '\nreturn copyDatasetPath(path)\n}', {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText

function copyFixture({ clipboard, legacyCopy = false } = {}) {
  const state = { notice: '', error: '', copied: '', removed: false }
  const field = { value: '', readOnly: false, style: {}, select() { state.copied = this.value }, remove() { state.removed = true } }
  const document = { createElement: () => field, body: { appendChild() {} }, execCommand: () => legacyCopy }
  const context = { navigator: { clipboard }, document, setNotice: value => { state.notice = value }, setPathCopyError: value => { state.error = value } }
  return { state, exercise: vm.runInNewContext(compiled + '\nexercise', context) }
}

test('dataset card copies the exact full path through the browser clipboard', async () => {
  const fixture = copyFixture({ clipboard: { writeText: async value => { fixture.state.copied = value } } })
  await fixture.exercise('/datasets/长目录/数据.yaml')
  assert.equal(fixture.state.copied, '/datasets/长目录/数据.yaml')
  assert.equal(fixture.state.notice, '数据集路径已复制')
  assert.equal(fixture.state.error, '')
  assert.match(source, /<details className="dataset-path-details"><summary>查看完整路径<\/summary><code>\{dataset\.directory\}<\/code><\/details>/)
})

test('dataset path remains manually selectable when clipboard permission is denied', async () => {
  const fixture = copyFixture({ clipboard: { writeText: async () => { throw new Error('denied') } } })
  await fixture.exercise('remote/dataset.yaml')
  assert.match(fixture.state.error, /手动选择复制/)
  assert.equal(fixture.state.notice, '')
})

test('private HTTP deployments can copy via the legacy browser fallback', async () => {
  const fixture = copyFixture({ legacyCopy: true })
  await fixture.exercise('relative/data.yaml')
  assert.equal(fixture.state.copied, 'relative/data.yaml')
  assert.equal(fixture.state.removed, true)
  assert.equal(fixture.state.notice, '数据集路径已复制')
})
