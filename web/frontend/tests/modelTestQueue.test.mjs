import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'
const source = readFileSync(new URL('../src/modelTestQueue.ts', import.meta.url), 'utf8')
const module = { exports: {} }
vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, { exports: module.exports })
const { bindImageTest, imageResultUrl, imageTestKey, imageTestState, pendingTestImages, refreshImageTest, removeImageTest } = module.exports
const key = imageTestKey('model-a', .25, .7, false)
const image = id => ({ id, file: { name: 'same-name.jpg', size: 10 }, previewUrl: `blob:${id}` })
const record = (id, status = 'completed', task_type = 'detect') => ({ id, status, task_type, model: {}, files: [{ name: 'same-name.jpg' }] })

test('stable IDs preserve source batch URLs when earlier images are removed or names match', () => {
  let images = bindImageTest([image(1), image(2)], [1, 2], key, record('first'))
  images = [...images, image(3)]
  assert.deepEqual(Array.from(pendingTestImages(images, key), item => item.id), [3])
  images = bindImageTest(images, [3], key, record('second'))
  assert.equal(imageResultUrl(images[2]), '/api/model-tests/second/images/0/result')
  images = images.slice(1)
  assert.equal(imageResultUrl(images[0]), '/api/model-tests/first/images/1/result')
  assert.equal(imageResultUrl(images[1]), '/api/model-tests/second/images/0/result')
})

test('failed and cancelled retries preserve old results but remain pending', () => {
  for (const status of ['failed', 'cancelled']) {
    let images = bindImageTest([image(1), image(2)], [1, 2], key, record('first'))
    images = bindImageTest(images, [2], key, record('retry', status))
    assert.equal(imageTestState(images[0], key), 'completed')
    assert.equal(imageTestState(images[1], key), 'failed')
    assert.equal(images[1].result.test.id, 'first')
    assert.equal(pendingTestImages(images, key).length, 1)
  }
})

test('model version and detection thresholds invalidate results; classifying ignores IoU', () => {
  const images = bindImageTest([image(1)], [1], key, record('detect'))
  assert.equal(imageTestState(images[0], imageTestKey('model-b', .25, .7, false)), 'stale')
  assert.equal(imageTestState(images[0], imageTestKey('model-a', .3, .7, false)), 'stale')
  assert.equal(imageTestState(images[0], imageTestKey('model-a', .25, .8, false)), 'stale')
  const classification = bindImageTest([image(1)], [1], key, record('classify', 'completed', 'classify'))
  assert.equal(imageTestState(classification[0], imageTestKey('model-a', .25, .8, true)), 'completed')
  assert.equal(imageTestState(classification[0], imageTestKey('model-a', .3, .8, true)), 'stale')
})

test('background state recovery and deletion affect only the associated batch', () => {
  let images = bindImageTest([image(1), image(2)], [1, 2], key, record('running', 'running'))
  assert.equal(pendingTestImages(images, key).length, 0)
  images = refreshImageTest(images.slice(1), record('running'))
  assert.equal(imageResultUrl(images[0]), '/api/model-tests/running/images/1/result')
  images = bindImageTest([...images, image(3)], [3], key, record('other'))
  images = removeImageTest(images, 'running')
  assert.equal(imageTestState(images[0], key), 'pending')
  assert.equal(imageTestState(images[1], key), 'completed')
})
