import { useEffect, useRef, useState } from 'react'
import { AlertCircle, CheckCircle2, Download, Images, Info, LoaderCircle, Maximize2, Play, ScanEye, Upload, X } from 'lucide-react'
import { api, formatBytes } from '../api'
import type { Artifact, ModelTest } from '../types'
import '../inference.css'

interface Props {
  artifact: Artifact
  taskName: string
  onClose: () => void
}

interface SelectedImage {
  id: number
  file: File
  previewUrl: string
}

const statuses: Record<ModelTest['status'], string> = {
  uploading: '准备图片', preparing: '准备模型', queued: '等待测试', running: '正在检测', completed: '检测完成', failed: '检测失败',
}

function detectionLabel(count: number | null) {
  return count === null ? '结果未就绪' : count > 0 ? `检测到 ${count} 个目标` : '未检测到目标'
}

export function ModelTestModal({ artifact, taskName, onClose }: Props) {
  const [images, setImages] = useState<SelectedImage[]>([])
  const [confidence, setConfidence] = useState(0.25)
  const [test, setTest] = useState<ModelTest | null>(null)
  const [selected, setSelected] = useState(0)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [stage, setStage] = useState('')
  const [error, setError] = useState('')
  const [enlarged, setEnlarged] = useState<{ url: string; title: string } | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const controller = useRef<AbortController | null>(null)
  const testId = useRef<string | null>(null)
  const dialog = useRef<HTMLDivElement>(null)
  const lightbox = useRef<HTMLDivElement>(null)
  const lightboxOpener = useRef<HTMLElement | null>(null)
  const dragDepth = useRef(0)
  const nextImageId = useRef(0)
  const previewUrls = useRef(new Set<string>())
  const files = images.map(({ file }) => file)

  useEffect(() => () => {
    controller.current?.abort()
    const urls = [...previewUrls.current]
    previewUrls.current.clear()
    // Let React detach image elements before invalidating their browser URLs.
    window.setTimeout(() => urls.forEach((url) => URL.revokeObjectURL(url)), 0)
    const id = testId.current
    if (id) {
      const cleanup = async () => {
        for (let retry = 0; retry < 3; retry++) {
          try { await api.deleteModelTest(id); return } catch { await new Promise((resolve) => window.setTimeout(resolve, 1500)) }
        }
      }
      void cleanup()
    }
  }, [])

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    dialog.current?.focus()
    return () => previous?.focus()
  }, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopImmediatePropagation()
        if (enlarged) setEnlarged(null)
        else onClose()
      }
      if (event.key === 'Tab') {
        const focusScope = enlarged ? lightbox.current : dialog.current
        const controls = focusScope?.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], input:not(:disabled), [tabindex="0"]')
        if (!controls?.length) {
          event.preventDefault()
          focusScope?.focus()
          return
        }
        const first = controls[0]; const last = controls[controls.length - 1]
        if (!focusScope?.contains(document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first).focus() }
        else if (event.shiftKey && (document.activeElement === first || document.activeElement === focusScope)) { event.preventDefault(); last.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [busy, enlarged, onClose])

  useEffect(() => {
    if (!enlarged) return
    const opener = lightboxOpener.current
    lightbox.current?.querySelector<HTMLElement>('button:not(:disabled)')?.focus()
    return () => {
      lightboxOpener.current = null
      if (opener?.isConnected) opener.focus()
    }
  }, [enlarged])

  function openLightbox(url: string, title: string, opener: HTMLElement) {
    lightboxOpener.current = opener
    setEnlarged({ url, title })
  }

  function chooseImages(list: FileList | null) {
    const added = Array.from(list || [])
    if (input.current) input.current.value = ''
    if (!added.length || busy) return
    const chosen = [...files, ...added]
    if (chosen.length > 10 || chosen.reduce((sum, file) => sum + file.size, 0) > 100 * 1024 ** 2) {
      setError('每次最多 10 张图片，总大小不超过 100 MB'); return
    }
    if (chosen.some((file) => !/\.(jpe?g|png|webp|bmp)$/i.test(file.name) || !file.size || file.size > 20 * 1024 ** 2)) {
      setError('请选择 JPG、PNG、WEBP 或 BMP 图片，单张不超过 20 MB'); return
    }
    const additions = added.map((file) => {
      const previewUrl = URL.createObjectURL(file)
      previewUrls.current.add(previewUrl)
      return { id: nextImageId.current++, file, previewUrl }
    })
    setImages((current) => [...current, ...additions]); setError(''); setTest(null); setStage(''); setEnlarged(null)
  }

  function removeImage(index: number) {
    if (busy) return
    const removed = images[index]
    const remaining = images.filter((_, current) => current !== index)
    setImages(remaining)
    if (removed) {
      previewUrls.current.delete(removed.previewUrl)
      // The state update removes the image source before the next task revokes it.
      window.setTimeout(() => URL.revokeObjectURL(removed.previewUrl), 0)
    }
    setSelected(Math.max(0, Math.min(selected > index ? selected - 1 : selected, remaining.length - 1)))
    setError(''); setTest(null); setStage(''); setEnlarged(null)
  }

  function resetDrag() {
    dragDepth.current = 0
    setDragging(false)
  }

  function dragEnter(event: React.DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation()
    if (busy) return
    dragDepth.current += 1
    setDragging(true)
  }

  function dragOver(event: React.DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation()
    event.dataTransfer.dropEffect = busy ? 'none' : 'copy'
  }

  function dragLeave(event: React.DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation()
    dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (!dragDepth.current) setDragging(false)
  }

  function dropImages(event: React.DragEvent) {
    event.preventDefault(); event.stopPropagation()
    resetDrag()
    if (busy) return
    if (Array.from(event.dataTransfer.items).some((item) => item.webkitGetAsEntry?.()?.isDirectory)) {
      setError('暂不支持拖入文件夹，请选择图片文件'); return
    }
    chooseImages(event.dataTransfer.files)
  }

  async function run() {
    if (busy || !files.length) return
    const abort = new AbortController()
    controller.current = abort
    const { signal } = abort
    const pause = async () => {
      await new Promise<void>((resolve) => window.setTimeout(resolve, 1500))
      signal.throwIfAborted()
    }
    resetDrag(); setBusy(true); setError(''); setTest(null); setEnlarged(null)
    try {
      if (testId.current) {
        await api.deleteModelTest(testId.current).catch((reason: Error & { status?: number }) => { if (reason.status !== 404) throw reason })
        testId.current = null
      }
      setStage('正在准备模型，远程权重首次使用需要传输…')
      const deadline = Date.now() + 5 * 60_000
      let first = true
      while (true) {
        signal.throwIfAborted()
        const prepared = await api.prepareArtifact(artifact.id, first, signal)
        first = false
        signal.throwIfAborted()
        if (prepared.status === 'ready') break
        if (prepared.status === 'failed') throw new Error(prepared.message || '模型准备失败')
        if (Date.now() > deadline) throw new Error('模型尚未准备完成，请确认远程服务器在线后重试')
        await pause()
      }
      const created = await api.createModelTest(artifact.id, confidence, files, signal)
      testId.current = created.id
      for (let index = 0; index < files.length; index++) {
        setStage(`正在上传 ${index + 1}/${files.length}：${files[index].name}`)
        await api.uploadModelTestImage(created.id, index, files[index], signal)
      }
      setStage('正在准备模型副本…')
      let current = await api.startModelTest(created.id, signal)
      const inferenceDeadline = Date.now() + 15 * 60_000
      while (true) {
        setTest(current); setStage(statuses[current.status])
        if (current.status === 'failed') throw new Error(current.error || '检测失败')
        if (current.status === 'completed') break
        if (Date.now() > inferenceDeadline) throw new Error('等待测试超时，请重试或减少图片数量')
        await pause()
        current = await api.modelTest(created.id, signal)
      }
    } catch (reason) {
      if (!signal.aborted) {
        setError(reason instanceof Error ? reason.message : '模型测试失败')
        setStage(statuses.failed)
      }
    } finally {
      if (!signal.aborted) setBusy(false)
    }
  }

  function close() {
    onClose()
  }

  const completed = test?.status === 'completed'
  const failed = stage === statuses.failed
  const detections = completed ? test.files[selected].detections : null
  const detectedImages = completed ? test.files.filter((file) => file.detections !== null && file.detections > 0).length : 0
  const emptyImages = completed ? test.files.filter((file) => file.detections === 0).length : 0
  const original = completed ? `/api/model-tests/${test.id}/images/${selected}/original` : images[selected]?.previewUrl
  const result = completed ? `/api/model-tests/${test.id}/images/${selected}/result` : ''
  const selectedFile = files[selected]

  return <div className="modal-backdrop model-test-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) close() }} onDragOver={(event) => { if (event.dataTransfer.types.includes('Files')) { event.preventDefault(); event.dataTransfer.dropEffect = 'none' } }} onDrop={(event) => { event.preventDefault(); resetDrag() }}>
    <div ref={dialog} tabIndex={-1} className={`modal model-test-modal${dragging ? ' is-dragging' : ''}`} role="dialog" aria-modal="true" aria-labelledby="model-test-title" onDragEnter={dragEnter} onDragOver={dragOver} onDragLeave={dragLeave} onDrop={dropImages}>
      {dragging && <div className="model-test-drop-indicator" role="status"><Upload size={40} /><strong>松开添加图片</strong><span>{files.length ? '追加到当前图片列表' : '支持一次拖入多张图片'}</span></div>}
      <header><div className="modal-heading"><div className="heading-icon"><ScanEye size={22} /></div><div><span className="eyebrow">MODEL TEST</span><h2 id="model-test-title">测试模型</h2><p>{taskName} · {artifact.name.split('/').pop()}</p></div></div><button className="icon-button" onClick={close} aria-label="关闭模型测试" title={busy ? '取消测试并关闭' : '关闭'}><X size={20} /></button></header>
      <div className="model-test-body">
        <div className="model-test-settings">
          <button className="button ghost" disabled={busy} onClick={() => input.current?.click()}><Upload size={16} />{files.length ? `添加图片（${files.length}/10）` : '上传图片'}</button>
          <input ref={input} hidden disabled={busy} type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" multiple onChange={(event) => chooseImages(event.target.files)} />
          <label htmlFor="test-confidence">置信度阈值 <strong>{confidence.toFixed(2)}</strong></label>
          <input id="test-confidence" type="range" min="0.01" max="1" step="0.01" value={confidence} disabled={busy} onChange={(event) => setConfidence(Number(event.target.value))} />
          <button className="button primary" disabled={busy || !files.length} onClick={() => void run()}>{busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}{completed ? '重新检测' : '运行检测'}</button>
        </div>
        <p className="model-test-help">{busy ? '检测期间不能添加或移除图片' : '可继续拖入图片追加，点击缩略图右上角移除'} · 最多 10 张 · 单张 20 MB / 2000 万像素 · 总大小 100 MB · CPU 检测</p>
        {error && <div className="model-test-error" role="alert">{error}</div>}
        {(stage || busy) && <div className={`model-test-status${failed ? ' failed' : ''}`} role="status">{busy ? <LoaderCircle className="spin" size={14} /> : failed ? <AlertCircle size={14} /> : null}{stage}{completed && ` · 本次阈值 ${test.confidence.toFixed(2)}`}</div>}
        {completed && <div className="model-test-summary" role="status">共 {test.files.length} 张：<strong>{detectedImages} 张检测到目标</strong>，{emptyImages} 张未检测到。</div>}
        {!files.length ? <button className="model-test-empty" disabled={busy} onClick={() => input.current?.click()}><Images size={46} /><strong>点击或拖入图片</strong><span>支持 JPG、PNG、WEBP、BMP，可一次上传多张</span></button> : <div className="model-test-workspace">
          <nav className="model-test-thumbnails" aria-label="测试图片">{images.map(({ id, file, previewUrl }, index) => <div className="model-test-thumbnail" key={id}><button className={`model-test-thumbnail-select${selected === index ? ' selected' : ''}`} onClick={() => setSelected(index)} aria-pressed={selected === index}><img src={previewUrl} alt="" draggable={false} /><span title={file.name}>{file.name}</span><small className={completed && (test.files[index].detections || 0) > 0 ? 'has-detections' : undefined}>{completed ? detectionLabel(test.files[index].detections) : formatBytes(file.size)}</small></button><button className="model-test-remove" disabled={busy} title={`移除 ${file.name}`} aria-label={`移除 ${file.name}`} onClick={() => removeImage(index)}><X size={14} /></button></div>)}</nav>
          <div className="model-test-comparison">{[{ url: original, title: '原图' }, { url: result, title: '检测结果' }].map(({ url, title }) => <section className="model-test-panel" key={title}><div><strong>{title}</strong>{url && <button className="icon-button" title={`放大${title}`} aria-label={`放大${title}`} onClick={(event) => openLightbox(url, title, event.currentTarget)}><Maximize2 size={15} /></button>}</div>{title === '检测结果' && completed && detections !== null && <div className={`model-test-outcome ${detections > 0 ? 'detected' : 'not-detected'}`} role="status">{detections > 0 ? <CheckCircle2 size={19} /> : <Info size={19} />}<div><strong>{detections > 0 ? detectionLabel(detections) : `当前置信度阈值 ${test.confidence.toFixed(2)} 下未检测到目标`}</strong>{detections === 0 && <p>可适当降低阈值后重新检测。这不代表图片中一定没有目标。</p>}</div></div>}{url ? <button className="model-test-image" onDoubleClick={(event) => openLightbox(url, title, event.currentTarget)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openLightbox(url, title, event.currentTarget) } }} aria-label={`${title}，双击放大`}><img src={url} alt={`${selectedFile?.name} ${title}`} /></button> : <div className="model-test-placeholder">{busy ? <LoaderCircle className="spin" size={28} /> : failed ? <AlertCircle size={32} /> : <ScanEye size={32} />}<span>{busy ? '检测结果生成中…' : failed ? '检测失败，请查看上方错误原因' : '运行检测后，查看目标框与置信度'}</span></div>}</section>)}</div>
        </div>}
      </div>
      <footer><span>{completed ? `${selectedFile?.name} · ${detectionLabel(test.files[selected].detections)} · 双击图片放大 · 关闭前请下载结果` : '阈值越高，显示的检测结果越严格'}</span>{result && <a className="button primary" href={`${result}?download=true`} download><Download size={15} />下载结果图</a>}<button className="button ghost" onClick={close}>{busy ? '取消并关闭' : '关闭'}</button></footer>
      {enlarged && <div ref={lightbox} className="model-test-lightbox" role="dialog" aria-modal="true" aria-label={`放大查看${enlarged.title}`} tabIndex={-1} onClick={() => setEnlarged(null)}><button className="icon-button" aria-label="关闭图片放大" onClick={() => setEnlarged(null)}><X size={24} /></button><img src={enlarged.url} alt={enlarged.title} onClick={(event) => event.stopPropagation()} /><span>{enlarged.title} · Esc 关闭</span></div>}
    </div>
  </div>
}
