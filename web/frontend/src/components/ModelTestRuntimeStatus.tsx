import { useEffect, useRef, useState } from 'react'
import { Cpu, LoaderCircle, Power } from 'lucide-react'
import { api } from '../api'
import type { ModelTestRuntime } from '../types'
import '../inference.css'

export function ModelTestRuntimeStatus() {
  const [runtime, setRuntime] = useState<ModelTestRuntime | null>(null)
  const [error, setError] = useState('')
  const [stale, setStale] = useState(false)
  const [releasing, setReleasing] = useState(false)
  const lifecycle = useRef<AbortController | null>(null)
  const revision = useRef(0)

  useEffect(() => {
    const abort = new AbortController()
    lifecycle.current = abort
    let timer = 0
    async function poll() {
      const version = revision.current
      if (!document.hidden) {
        try {
          const value = await api.modelTestRuntime(abort.signal)
          if (!abort.signal.aborted && version === revision.current) {
            setRuntime(value)
            setStale(false)
          }
        } catch {
          if (!abort.signal.aborted && version === revision.current) setStale(true)
        }
      }
      if (!abort.signal.aborted) timer = window.setTimeout(() => void poll(), 2000)
    }
    void poll()
    return () => { abort.abort(); window.clearTimeout(timer); lifecycle.current = null }
  }, [])

  async function release() {
    const signal = lifecycle.current?.signal
    if (!signal || signal.aborted || releasing) return
    revision.current += 1
    setReleasing(true)
    setError('')
    try {
      const value = await api.releaseModelTestRuntime(signal)
      if (!signal.aborted) { setRuntime(value); setStale(false) }
    } catch (reason) {
      if (!signal.aborted) setError(reason instanceof Error ? reason.message : '释放失败，请稍后重试')
    } finally {
      revision.current += 1
      if (!signal.aborted) setReleasing(false)
    }
  }

  const busy = runtime?.state === 'loading' || runtime?.state === 'running' || !!runtime?.active_test_id
  const label = stale ? '模型驻留状态暂不可用'
    : !runtime ? '正在读取模型驻留状态…'
      : runtime.state === 'loading' ? '正在加载模型'
        : busy ? '正在执行测试'
          : runtime.state === 'ready' ? '模型已驻留，可复用'
            : runtime.release_reason === 'idle' ? '空闲模型已自动释放'
              : runtime.release_reason === 'manual' ? '模型已手动释放'
                : runtime.release_reason === 'cancelled' ? '测试已取消，模型已释放'
                  : runtime.release_reason === 'failed' || runtime.release_reason === 'process_exit' ? '测试进程已释放，下次重新加载'
                  : '当前无驻留模型'
  const expiry = runtime?.expires_in_seconds
  return <section className="model-test-runtime" aria-label="中心模型内存状态">
    <div className="model-test-runtime-main">
      {busy && !stale ? <LoaderCircle size={16} className="spin" /> : <Cpu size={16} />}
      <div><strong role="status">{label}</strong>
        <p title={runtime?.model_sha256 ? `SHA-256: ${runtime.model_sha256}` : undefined}>中心共享 CPU · 最多驻留 1 个模型{runtime?.model_name && !stale ? ` · ${runtime.model_name}` : ''}
          {runtime?.model_sha256 && !stale ? ` · 版本 ${runtime.model_sha256.slice(0, 8)}` : ''}
          {runtime && !stale && (runtime.idle_seconds === 0 ? ' · 每批完成后释放' : ` · 空闲 ${runtime.idle_seconds} 秒后释放`)}
          {runtime?.state === 'ready' && !busy && !stale && expiry !== null && expiry !== undefined ? `（约 ${expiry} 秒）` : ''}
          {!!runtime?.waiting_tests && !stale && ` · 待完成 ${runtime.waiting_tests} 批`}
        </p>
      </div>
      <button type="button" className="button ghost compact" disabled={!runtime?.can_release || stale || releasing}
        title="仅释放模型测试进程的内存；已有结果和原权重继续保留。测试执行或排队时不可释放。"
        onClick={() => void release()}>{releasing ? <LoaderCircle size={14} className="spin" /> : <Power size={14} />}立即释放</button>
    </div>
    {error && <p className="model-test-error" role="alert">{error}</p>}
  </section>
}
