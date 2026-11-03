import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

function harness() {
  const cache = new Map(), events = [], calls = [], uploads = []
  let respond = async () => ({ ok: true, status: 200, json: async () => ({}) })
  const window = { setTimeout, clearTimeout, dispatchEvent: event => events.push(event) }
  class XHR {
    upload = {}
    headers = {}
    status = 204
    responseText = ''
    constructor() { uploads.push(this) }
    open() {}
    setRequestHeader(key, value) { this.headers[key] = value }
    send() {}
    abort() { this.onabort?.() }
  }
  function load(name) {
    if (cache.has(name)) return cache.get(name)
    const source = readFileSync(new URL(`../src/${name}.ts`, import.meta.url), 'utf8')
    const compiled = ts.transpileModule(source, { compilerOptions: {
      target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS,
    } }).outputText
    const module = { exports: {} }
    cache.set(name, module.exports)
    vm.runInNewContext(compiled, { exports: module.exports,
      require: path => load(path.replace('./', '')), window, AbortController, DOMException,
      CustomEvent: class { constructor(type, init) { this.type = type; this.detail = init.detail } },
      XMLHttpRequest: XHR, fetch: async (path, options) => { calls.push({ path, options }); return respond(path, options) },
    })
    return module.exports
  }
  return { auth: load('authSession'), transport: load('api'), events, calls, uploads,
    respond: callback => { respond = callback } }
}

const unauthorized = () => ({ ok: false, status: 401, json: async () => ({ detail: 'session expired' }) })

test('requests and binary uploads carry the current CSRF token without browser storage', async () => {
  const h = harness()
  h.auth.setCsrfToken('csrf-current')
  await h.transport.api.importRuns()
  assert.equal(h.calls[0].options.headers['X-CSRF-Token'], 'csrf-current')
  assert.equal(h.calls[0].options.credentials, 'same-origin')
  const upload = h.transport.uploadDatasetFile('upload-id', 0, { size: 3 }, new AbortController().signal, () => {})
  assert.equal(h.uploads[0].headers['X-CSRF-Token'], 'csrf-current')
  assert.equal(h.uploads[0].headers['Content-Type'], 'application/octet-stream')
  h.uploads[0].onload()
  await upload
})

test('incorrect login stays on the form, while a private 401 invalidates the browser session', async () => {
  const h = harness()
  h.auth.setCsrfToken('current')
  h.respond(unauthorized)
  await assert.rejects(h.transport.api.login('admin', 'incorrect'), /session expired/)
  assert.equal(h.events.length, 0)
  await assert.rejects(h.transport.api.resources(), /session expired/)
  assert.equal(h.events.length, 1)
  assert.equal(h.events[0].type, h.auth.AUTH_EXPIRED_EVENT)
  assert.equal(h.auth.csrfHeaders()['X-CSRF-Token'], undefined)
})

test('a late 401 from an old request cannot dismiss a newly authenticated session', async () => {
  const h = harness()
  let finish
  h.auth.setCsrfToken('old-session')
  h.respond(() => new Promise(resolve => { finish = resolve }))
  const pending = h.transport.api.resources()
  h.auth.setCsrfToken('new-session')
  finish(unauthorized())
  await assert.rejects(pending, /session expired/)
  assert.equal(h.events.length, 0)
  assert.equal(h.auth.csrfHeaders()['X-CSRF-Token'], 'new-session')
})

test('binary upload authentication failure also respects the session generation', async () => {
  const h = harness()
  h.auth.setCsrfToken('old')
  const upload = h.transport.uploadDatasetFile('upload-id', 0, { size: 3 }, new AbortController().signal, () => {})
  h.auth.setCsrfToken('new')
  h.uploads[0].status = 401
  h.uploads[0].responseText = '{"detail":"session expired"}'
  h.uploads[0].onload()
  await assert.rejects(upload, /session expired/)
  assert.equal(h.events.length, 0)
  assert.equal(h.auth.csrfHeaders()['X-CSRF-Token'], 'new')
})
