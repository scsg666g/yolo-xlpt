import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { api } from '../api'
import type { Artifact } from '../types'

function usePreparedArtifact(artifact: Artifact, active: boolean) {
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    setReady(false)
    if (!active) return
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    let first = true
    const started = Date.now()
    setError('')
    async function prepare() {
      try {
        const result = await api.prepareArtifact(artifact.id, first)
        first = false
        if (cancelled) return
        if (result.status === 'ready') { setReady(true); return }
        if (result.status === 'failed') throw new Error(result.message || '文件传输失败')
        if (Date.now() - started > 180_000) throw new Error('文件尚未就绪，请确认 Agent 和代理在线后重试')
        timer = setTimeout(prepare, 2000)
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : '准备下载失败')
      }
    }
    void prepare()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [artifact.id, artifact.modified_at, artifact.size, active, retry])
  return { ready, error, retry: () => setRetry((value) => value + 1) }
}

export function ArtifactDownload({ artifact, children, className = '', title }: {
  artifact: Artifact; children: ReactNode; className?: string; title?: string
}) {
  const [active, setActive] = useState(false)
  const { ready, error, retry } = usePreparedArtifact(artifact, active)
  useEffect(() => {
    if (ready && active) {
      window.location.assign(`/api/artifacts/${artifact.id}/download`)
      setActive(false)
    }
  }, [ready, active, artifact.id])
  // Once ready the browser streams the cached file without loading weights into JS memory.
  return <span className="artifact-download-control" onDoubleClick={(event) => event.stopPropagation()}>
    {ready ? <a className={className} href={`/api/artifacts/${artifact.id}/download`} title={title} onClick={(event) => event.stopPropagation()}>{children}</a>
      : <button className={className || 'icon-button'} title={error || title || `下载 ${artifact.name}`} disabled={active && !error} onClick={(event) => { event.stopPropagation(); if (active) retry(); else setActive(true) }}>{active && !error ? '准备中…' : error ? '重试下载' : children}</button>}
    {error && <small role="alert">{error}</small>}
  </span>
}

export function ArtifactPreview({ artifact }: { artifact: Artifact }) {
  const { ready, error, retry } = usePreparedArtifact(artifact, true)
  if (error) return <button onClick={retry} title={error}>图片未就绪，点击重试</button>
  if (!ready) return <span>正在获取图片…</span>
  return <a href={`/api/artifacts/${artifact.id}/download`} target="_blank" rel="noreferrer" title={`打开 ${artifact.name}`}><img src={`/api/artifacts/${artifact.id}/download`} alt={artifact.name} loading="lazy" /><span>{artifact.name}</span></a>
}
