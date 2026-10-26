import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/chartMetrics.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
const module = { exports: {} }
vm.runInNewContext(compiled, { exports: module.exports })
const { finiteMetricValue, hasAnyMetricValue, hasMetricValue, metricPointCount, metricSeriesDisplay } = module.exports

test('metric series keeps one or two valid results visible', () => {
  const empty = metricSeriesDisplay([])
  assert.equal(empty.showSymbol, false)
  assert.equal(empty.showAllSymbol, false)
  assert.equal(empty.symbol, 'circle')
  assert.equal(empty.symbolSize, 5)
  assert.equal(empty.connectNulls, false)

  const single = metricSeriesDisplay([0.42])
  assert.equal(single.showSymbol, true)
  assert.equal(single.showAllSymbol, true)
  assert.equal(single.symbol, 'circle')
  assert.equal(single.symbolSize, 8)
  assert.equal(single.connectNulls, false)

  const pair = metricSeriesDisplay([null, 0.42, Number.NaN, 0.51])
  assert.equal(pair.showSymbol, true)
  assert.equal(pair.showAllSymbol, true)
  assert.equal(pair.symbol, 'circle')
  assert.equal(pair.symbolSize, 5)
  assert.equal(pair.connectNulls, false)
  const continuous = metricSeriesDisplay([0.42, 0.51, 0.58])
  assert.equal(continuous.showSymbol, false)
  assert.equal(continuous.showAllSymbol, false)

  const isolated = metricSeriesDisplay([0.42, null, 0.51, null, 0.58])
  assert.equal(isolated.showSymbol, true)
  assert.equal(isolated.showAllSymbol, true)

  const separatedSegments = metricSeriesDisplay([0.42, 0.51, null, 0.58, 0.63])
  assert.equal(separatedSegments.showSymbol, false)
  assert.equal(separatedSegments.showAllSymbol, false)
})

test('only finite numbers count as metric values', () => {
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, null, undefined, '0.5', {}, []]) {
    assert.equal(finiteMetricValue(value), false)
  }
  assert.equal(finiteMetricValue(0), true)
  assert.equal(finiteMetricValue(-0.25), true)

  const metrics = [
    { epoch: 1, values: { map: null, loss: 0.8 } },
    { epoch: 2, values: { map: Number.NaN, loss: '0.7' } },
    { epoch: 3, values: { map: 0.6, loss: 0.5 } },
  ]
  assert.equal(metricPointCount(metrics, 'map'), 1)
  assert.equal(metricPointCount(metrics, 'loss'), 2)
  assert.equal(hasMetricValue(metrics, ['missing', 'map']), true)
  assert.equal(hasMetricValue(metrics, ['missing']), false)
  assert.equal(hasAnyMetricValue(metrics), true)
  assert.equal(hasAnyMetricValue([{ epoch: 1, values: { map: Number.NaN, loss: null } }]), false)
})
