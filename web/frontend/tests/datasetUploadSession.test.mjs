import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { createRequire } from 'node:module'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/datasetUploadSession.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const module = { exports: {} }
vm.runInNewContext(compiled, { exports: module.exports, require: createRequire(import.meta.url), TextEncoder, Uint8Array, DOMException })
const { hashDatasetFile, datasetSessionKey } = module.exports

test('streamed client checksum matches server SHA-256 across multiple chunks', async () => {
  const bytes = Buffer.alloc(3 * 1024 * 1024 + 77, 19)
  const actual = await hashDatasetFile(new Blob([bytes]), new AbortController().signal)
  assert.equal(actual, createHash('sha256').update(bytes).digest('hex'))
})
test('resume identity changes with target server location and actual file contents', () => {
  const manifest = [{ path: 'images/train/a.jpg', size: 123, sha256: 'a'.repeat(64) }]
  const first = datasetSessionKey('local-location', '数据集', manifest)
  assert.equal(first, datasetSessionKey('local-location', '数据集', manifest))
  assert.notEqual(first, datasetSessionKey('remote-location', '数据集', manifest))
  assert.notEqual(first, datasetSessionKey('local-location', '数据集', [{ ...manifest[0], sha256: 'b'.repeat(64) }]))
})
test('stopping checksum computation honours the upload abort signal', async () => {
  const controller = new AbortController(); controller.abort()
  await assert.rejects(hashDatasetFile(new Blob(['bytes']), controller.signal), { name: 'AbortError' })
})
