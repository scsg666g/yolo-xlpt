import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertCircle, Download, FileArchive, HardDrive, History, Info, LoaderCircle, Play,
  RefreshCw, RotateCcw, ScanEye, Trash2, Upload, X,
} from 'lucide-react'
import { api, formatBytes } from '../api'
import type { Artifact, ModelTest, ModelTestSummary } from '../types'
import {
  activeModelTestStatuses, hasExactModelVersion, ModelStrip, modelTestAttemptText,
  modelTestDate, modelTestDetectionLabel, modelTestStatusLabels, modelTestWeightName,
  type SelectedImage, TestWorkspace,
} from './ModelTestShared'

interface Props {
  artifact?: Artifact | null
  taskId: string
  taskName: string
  runId?: string | null
  attemptNumber?: number | null
  bestMetric?: { epoch: number; value: number } | null
  newTestDisabledReason?: string | null
  onClose: () => void
}

type ModalTab = 'new' | 'history'
type HistoryScope = 'all' | 'current'

export function ModelTestModal({ artifact, taskId, taskName, runId, attemptNumber, bestMetric, newTestDisabledReason, onClose }: Props) {
  const [tab, setTab] = useState<ModalTab>(() => artifact && !newTestDisabledReason ? 'new' : 'history')
  const [images, setImages] = useState<SelectedImage[]>([])
  const [confidence, setConfidence] = useState(0.25)
  const [iou, setIou] = useState(0.7)
  const [test, setTest] = useState<ModelTest | null>(null)
  const [selected, setSelected] = useState(0)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [stage, setStage] = useState('')
  const [error, setError] = useState('')
  const [history, setHistory] = useState<ModelTestSummary[]>([])
  const [historyDetails, setHistoryDetails] = useState<Record<string, ModelTest>>({})
  const [historyNextCursor, setHistoryNextCursor] = useState<string | null>(null)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyLoadingMore, setHistoryLoadingMore] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [historyScope, setHistoryScope] = useState<HistoryScope>('all')
  const [historySelectedId, setHistorySelectedId] = useState<string | null>(null)
  const [historyImage, setHistoryImage] = useState(0)
  const [historyDetailLoading, setHistoryDetailLoading] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [reusingId, setReusingId] = useState<string | null>(null)
  const [rerunSource, setRerunSource] = useState<ModelTest | null>(null)
  const [enlarged, setEnlarged] = useState<{ url: string; title: string } | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const runController = useRef<AbortController | null>(null)
  const pendingUploadId = useRef<string | null>(null)
  const historyDetailController = useRef<AbortController | null>(null)
  const reuseController = useRef<AbortController | null>(null)
  const historyRefreshVersion = useRef(0)
  const historyLoadingVersion = useRef(0)
  const dialog = useRef<HTMLDivElement>(null)
  const lightbox = useRef<HTMLDivElement>(null)
  const lightboxOpener = useRef<HTMLElement | null>(null)
  const dragDepth = useRef(0)
  const nextImageId = useRef(0)
  const previewUrls = useRef(new Set<string>())
  const files = images.map(({ file }) => file)
  const historyRunId = historyScope === 'current' ? runId : undefined

  const refreshHistory = useCallback(async (signal?: AbortSignal, quiet = false) => {
    const version = ++historyRefreshVersion.current
    const loadingVersion = quiet ? null : ++historyLoadingVersion.current
    if (loadingVersion !== null) {
      setHistoryLoading(true)
      setHistoryNextCursor(null)
    }
    try {
      const page = await api.modelTests(taskId, signal, undefined, 50, historyRunId)
      if (signal?.aborted || version !== historyRefreshVersion.current) return
      if (quiet) {
        const refreshedIds = new Set(page.items.map((item) => item.id))
        setHistory((current) => [...page.items, ...current.filter((item) => !refreshedIds.has(item.id))])
        setHistorySelectedId((current) => current || page.items[0]?.id || null)
      } else {
        setHistory(page.items)
        setHistoryNextCursor(page.next_cursor)
        setHistorySelectedId((current) => current && page.items.some((item) => item.id === current) ? current : page.items[0]?.id || null)
      }
      setHistoryError('')
    } catch (reason) {
      if (!signal?.aborted && version === historyRefreshVersion.current) setHistoryError(reason instanceof Error ? reason.message : '加载历史记录失败')
    } finally {
      if (!signal?.aborted && loadingVersion !== null && loadingVersion === historyLoadingVersion.current) setHistoryLoading(false)
    }
  }, [historyRunId, taskId])

  const loadHistoryDetail = useCallback(async (testId: string) => {
    historyDetailController.current?.abort()
    const abort = new AbortController()
    historyDetailController.current = abort
    setHistoryDetailLoading(true)
    setHistoryError('')
    try {
      const detail = await api.taskModelTest(taskId, testId, abort.signal)
      if (!abort.signal.aborted) setHistoryDetails((current) => ({ ...current, [detail.id]: detail }))
    } catch (reason) {
      if (!abort.signal.aborted) setHistoryError(reason instanceof Error ? reason.message : '加载检测详情失败')
    } finally {
      if (!abort.signal.aborted) setHistoryDetailLoading(false)
      if (historyDetailController.current === abort) historyDetailController.current = null
    }
  }, [taskId])

  async function loadMoreHistory() {
    if (!historyNextCursor || historyLoading || historyLoadingMore) return
    const cursor = historyNextCursor
    const generation = historyLoadingVersion.current
    setHistoryLoadingMore(true)
    setHistoryError('')
    try {
      const page = await api.modelTests(taskId, undefined, cursor, 50, historyRunId)
      if (generation !== historyLoadingVersion.current) return
      setHistory((current) => {
        const existingIds = new Set(current.map((item) => item.id))
        return [...current, ...page.items.filter((item) => !existingIds.has(item.id))]
      })
      setHistoryNextCursor(page.next_cursor)
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : '加载更多历史记录失败')
    } finally {
      setHistoryLoadingMore(false)
    }
  }

  useEffect(() => {
    const abort = new AbortController()
    void refreshHistory(abort.signal)
    return () => abort.abort()
  }, [refreshHistory])

  const hasActiveHistory = history.some((item) => activeModelTestStatuses.has(item.status))
  useEffect(() => {
    if (tab !== 'history' || !hasActiveHistory) return
    const abort = new AbortController()
    let timer = 0
    const poll = async () => {
      await refreshHistory(abort.signal, true)
      if (!abort.signal.aborted) timer = window.setTimeout(() => void poll(), 3000)
    }
    timer = window.setTimeout(() => void poll(), 3000)
    return () => { window.clearTimeout(timer); abort.abort() }
  }, [hasActiveHistory, refreshHistory, tab])

  useEffect(() => {
    const ownedPreviewUrls = previewUrls.current
    const abandonPendingUpload = () => {
      const testId = pendingUploadId.current
      if (!testId) return
      pendingUploadId.current = null
      void api.abandonModelTest(testId, true).catch(() => undefined)
    }
    const handlePageHide = (event: PageTransitionEvent) => {
      if (event.persisted) return
      abandonPendingUpload()
      runController.current?.abort()
    }
    window.addEventListener('pagehide', handlePageHide)
    return () => {
      window.removeEventListener('pagehide', handlePageHide)
      abandonPendingUpload()
      runController.current?.abort()
      historyDetailController.current?.abort()
      reuseController.current?.abort()
      const urls = [...ownedPreviewUrls]
      ownedPreviewUrls.clear()
      window.setTimeout(() => urls.forEach((url) => URL.revokeObjectURL(url)), 0)
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
        if (!controls?.length) { event.preventDefault(); focusScope?.focus(); return }
        const first = controls[0]; const last = controls[controls.length - 1]
        if (!focusScope?.contains(document.activeElement)) { event.preventDefault(); (event.shiftKey ? last : first).focus() }
        else if (event.shiftKey && (document.activeElement === first || document.activeElement === focusScope)) { event.preventDefault(); last.focus() }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [enlarged, onClose])

  useEffect(() => {
    if (!enlarged) return
    const opener = lightboxOpener.current
    lightbox.current?.querySelector<HTMLElement>('button:not(:disabled)')?.focus()
    return () => { lightboxOpener.current = null; if (opener?.isConnected) opener.focus() }
  }, [enlarged])

  function openLightbox(url: string, title: string, opener: HTMLElement) {
    lightboxOpener.current = opener
    setEnlarged({ url, title })
  }

  function releaseImages(current: SelectedImage[]) {
    for (const image of current) {
      previewUrls.current.delete(image.previewUrl)
      URL.revokeObjectURL(image.previewUrl)
    }
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
      window.setTimeout(() => URL.revokeObjectURL(removed.previewUrl), 0)
    }
    setSelected(Math.max(0, Math.min(selected > index ? selected - 1 : selected, remaining.length - 1)))
    setError(''); setTest(null); setStage(''); setEnlarged(null)
  }

  function resetDrag() { dragDepth.current = 0; setDragging(false) }
  function dragEnter(event: React.DragEvent) {
    if (tab !== 'new' || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation()
    if (busy) return
    dragDepth.current += 1; setDragging(true)
  }
  function dragOver(event: React.DragEvent) {
    if (tab !== 'new' || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation(); event.dataTransfer.dropEffect = busy ? 'none' : 'copy'
  }
  function dragLeave(event: React.DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation(); dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (!dragDepth.current) setDragging(false)
  }
  function dropImages(event: React.DragEvent) {
    if (tab !== 'new') return
    event.preventDefault(); event.stopPropagation(); resetDrag()
    if (busy) return
    if (Array.from(event.dataTransfer.items).some((item) => item.webkitGetAsEntry?.()?.isDirectory)) {
      setError('暂不支持拖入文件夹，请选择图片文件'); return
    }
    chooseImages(event.dataTransfer.files)
  }

  async function run() {
    if (busy || pendingUploadId.current || !files.length) return
    if (rerunSource && !hasExactModelVersion(rerunSource)) {
      setError('该历史记录没有可验证的原权重版本，不能按原权重重测')
      return
    }
    const artifactId = rerunSource?.artifact_id ?? artifact?.id
    if (!artifactId || (!rerunSource && newTestDisabledReason)) {
      setError(newTestDisabledReason || '当前没有可用的模型权重')
      return
    }
    const abort = new AbortController()
    runController.current = abort
    const { signal } = abort
    let createdId: string | null = null
    let lastKnown: ModelTest | null = null
    const pause = async () => { await new Promise<void>((resolve) => window.setTimeout(resolve, 1500)); signal.throwIfAborted() }
    resetDrag(); setBusy(true); setError(''); setTest(null); setEnlarged(null)
    try {
      if (rerunSource) {
        setStage('正在创建重测记录…')
      } else {
        setStage('正在准备模型，远程权重首次使用需要传输…')
        const deadline = Date.now() + 5 * 60_000
        let first = true
        while (true) {
          signal.throwIfAborted()
          const prepared = await api.prepareArtifact(artifactId, first, signal)
          first = false
          if (prepared.status === 'ready') break
          if (prepared.status === 'failed') throw new Error(prepared.message || '模型准备失败')
          if (Date.now() > deadline) throw new Error('模型尚未准备完成，请确认远程服务器在线后重试')
          await pause()
        }
      }
      const created = await api.createModelTest(
        artifactId, confidence, iou, files, signal, runId, rerunSource?.id,
      )
      createdId = created.id
      lastKnown = created
      pendingUploadId.current = created.id
      setTest(created)
      setStage(modelTestStatusLabels[created.status])
      for (let index = 0; index < files.length; index++) {
        setStage(`正在上传 ${index + 1}/${files.length}：${files[index].name}`)
        await api.uploadModelTestImage(created.id, index, files[index], signal)
      }
      setStage('正在准备模型副本…')
      let current = await api.startModelTest(created.id, signal)
      lastKnown = current
      pendingUploadId.current = null
      const inferenceDeadline = Date.now() + 15 * 60_000
      while (true) {
        lastKnown = current
        setTest(current); setStage(modelTestStatusLabels[current.status])
        if (current.status === 'failed') throw new Error(current.error || '检测失败')
        if (current.status === 'cancelled') throw new Error('检测已取消')
        if (current.status === 'completed') break
        if (Date.now() > inferenceDeadline) throw new Error('等待测试超时，请稍后在历史记录中查看')
        await pause()
        current = await api.modelTest(created.id, signal)
      }
      await refreshHistory(signal, true)
    } catch (reason) {
      if (!signal.aborted) {
        const message = reason instanceof Error ? reason.message : '模型测试请求失败'
        if (createdId) {
          try {
            lastKnown = await api.modelTest(createdId, signal)
            setTest(lastKnown)
          } catch {
            // Keep the last confirmed server state when the status request also fails.
          }
        }
        if (lastKnown?.status === 'failed') {
          setStage(modelTestStatusLabels.failed)
          setError(lastKnown.error || message)
        } else if (lastKnown?.status === 'cancelled') {
          pendingUploadId.current = null
          setStage(modelTestStatusLabels.cancelled)
          setError(lastKnown.error || message)
        } else if (lastKnown?.status === 'completed') {
          pendingUploadId.current = null
          setStage(modelTestStatusLabels.completed)
          setError('')
        } else if (lastKnown) {
          const uploadIncomplete = lastKnown.status === 'uploading'
          if (!uploadIncomplete) pendingUploadId.current = null
          setStage(`${modelTestStatusLabels[lastKnown.status]} · 本页状态同步中断`)
          setError(uploadIncomplete
            ? `${message}；图片上传或提交状态待确认，可取消本次上传后重试`
            : `${message}；检测已提交到后台并会继续运行，请在历史记录中刷新查看`)
        } else {
          setError(message)
          setStage(modelTestStatusLabels.failed)
        }
        await refreshHistory(signal, true)
      }
    } finally {
      if (!signal.aborted) setBusy(false)
      if (runController.current === abort) runController.current = null
    }
  }

  async function cancelPendingUpload() {
    const testId = pendingUploadId.current
    if (!testId) return
    runController.current?.abort()
    setBusy(true); setError(''); setStage('正在取消未完成的上传…')
    try {
      const cancelled = await api.abandonModelTest(testId)
      if (pendingUploadId.current === testId) pendingUploadId.current = null
      setTest(cancelled)
      setStage(modelTestStatusLabels[cancelled.status])
      await refreshHistory(undefined, true)
    } catch (reason) {
      const requestError = reason as Error & { status?: number }
      if (requestError.status === 409) {
        pendingUploadId.current = null
        const statusController = new AbortController()
        try {
          const current = await api.modelTest(testId, statusController.signal)
          setTest(current); setStage(modelTestStatusLabels[current.status])
          setError('检测已经提交到后台，可在历史记录中继续查看')
          await refreshHistory(undefined, true)
        } catch (statusReason) {
          setError(statusReason instanceof Error ? statusReason.message : '读取检测状态失败，请在历史记录中确认')
        }
      } else {
        setError(requestError.message || '取消上传失败，请重试')
        setStage(modelTestStatusLabels.failed)
      }
    } finally {
      setBusy(false)
    }
  }

  const visibleHistory = useMemo(() => historyScope === 'current' && runId
    ? history.filter((item) => item.run_id === runId)
    : history, [history, historyScope, runId])
  const selectedSummary = visibleHistory.find((item) => item.id === historySelectedId) || visibleHistory[0] || null
  const selectedHistory = selectedSummary ? historyDetails[selectedSummary.id] || null : null

  useEffect(() => {
    setHistoryImage(0)
  }, [selectedSummary?.id])

  useEffect(() => {
    if (selectedSummary && !selectedHistory) void loadHistoryDetail(selectedSummary.id)
  }, [loadHistoryDetail, selectedHistory, selectedSummary])

  useEffect(() => {
    if (selectedSummary && selectedHistory && activeModelTestStatuses.has(selectedHistory.status)
      && !activeModelTestStatuses.has(selectedSummary.status)) {
      setHistoryDetails((current) => {
        const next = { ...current }
        delete next[selectedSummary.id]
        return next
      })
    }
  }, [selectedHistory, selectedSummary])

  function openHistory(item: ModelTestSummary) {
    setHistorySelectedId(item.id); setHistoryImage(0); setHistoryError('')
  }

  async function deleteHistory(item: ModelTestSummary) {
    const active = activeModelTestStatuses.has(item.status)
    const prompt = active
      ? '这条检测仍在处理中。删除将显式取消它并清理已上传的图片，确定继续吗？'
      : '删除后原图、检测结果和下载文件都无法恢复，确定删除这条记录吗？'
    if (!window.confirm(prompt)) return
    historyRefreshVersion.current += 1
    setDeletingId(item.id); setHistoryError('')
    try {
      await api.deleteModelTest(item.id)
      historyRefreshVersion.current += 1
      historyLoadingVersion.current += 1
      setHistory((current) => current.filter((record) => record.id !== item.id))
      setHistoryDetails((current) => {
        const next = { ...current }
        delete next[item.id]
        return next
      })
      setHistorySelectedId((current) => current === item.id ? null : current)
      setHistoryImage(0)
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : '删除历史记录失败')
    } finally {
      setDeletingId(null)
    }
  }

  async function reuseHistory(item: ModelTest) {
    if (!item.files.length || reusingId) return
    if (!hasExactModelVersion(item)) {
      setHistoryError('该历史记录缺少可验证的原权重版本，仍可查看和下载已有结果')
      return
    }
    setReusingId(item.id); setHistoryError('')
    reuseController.current?.abort()
    const abort = new AbortController()
    reuseController.current = abort
    const prepared: SelectedImage[] = []
    try {
      for (let index = 0; index < item.files.length; index++) {
        const response = await fetch(`/api/model-tests/${item.id}/images/${index}/original`, { signal: abort.signal })
        if (!response.ok) throw new Error(`无法读取第 ${index + 1} 张原图`)
        const blob = await response.blob()
        const file = new File([blob], item.files[index].name || `image-${index + 1}.jpg`, { type: blob.type || 'image/jpeg' })
        const previewUrl = URL.createObjectURL(file)
        previewUrls.current.add(previewUrl)
        prepared.push({ id: nextImageId.current++, file, previewUrl })
      }
      releaseImages(images)
      setImages(prepared); setConfidence(item.confidence); setIou(item.iou); setSelected(0)
      setTest(null); setStage(''); setError(''); setRerunSource(item); setTab('new'); setEnlarged(null)
    } catch (reason) {
      releaseImages(prepared)
      if (!abort.signal.aborted) setHistoryError(reason instanceof Error ? reason.message : '准备重测图片失败')
    } finally {
      if (!abort.signal.aborted) setReusingId(null)
      if (reuseController.current === abort) reuseController.current = null
    }
  }

  const completed = test?.status === 'completed'
  const failed = stage === modelTestStatusLabels.failed
  const parametersChanged = completed && (confidence !== test.confidence || iou !== test.iou)
  const currentName = test?.files[selected]?.name || files[selected]?.name
  const currentResult = completed ? `/api/model-tests/${test.id}/images/${selected}/result` : ''
  const canOpenNewTab = !!test || (!!rerunSource && hasExactModelVersion(rerunSource)) || (!!artifact && !newTestDisabledReason)
  const hasPendingUpload = !!pendingUploadId.current

  useEffect(() => {
    if (tab === 'new' && !canOpenNewTab && !busy) setTab('history')
  }, [busy, canOpenNewTab, tab])

  return <div className="modal-backdrop model-test-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }} onDragOver={(event) => { if (tab !== 'new' && event.dataTransfer.types.includes('Files')) { event.preventDefault(); event.dataTransfer.dropEffect = 'none' } }} onDrop={(event) => { if (tab !== 'new') { event.preventDefault(); resetDrag() } }}>
    <div ref={dialog} tabIndex={-1} className={`modal model-test-modal${dragging ? ' is-dragging' : ''}`} role="dialog" aria-modal="true" aria-labelledby="model-test-title" onDragEnter={dragEnter} onDragOver={dragOver} onDragLeave={dragLeave} onDrop={dropImages}>
      {dragging && <div className="model-test-drop-indicator" role="status"><Upload size={40} /><strong>松开添加图片</strong><span>{files.length ? '追加到当前图片列表' : '支持一次拖入多张图片'}</span></div>}
      <header><div className="modal-heading"><div className="heading-icon"><ScanEye size={22} /></div><div><span className="eyebrow">MODEL TEST</span><h2 id="model-test-title">测试模型</h2><p>{taskName} · 检测记录保存在中心服务器</p></div></div><button className="icon-button" onClick={onClose} aria-label="关闭模型测试" title="关闭"><X size={20} /></button></header>
      <nav className="model-test-tabs" role="tablist" aria-label="模型测试视图">
        <button role="tab" aria-selected={tab === 'new'} className={tab === 'new' ? 'active' : ''} disabled={!canOpenNewTab} title={!canOpenNewTab ? newTestDisabledReason || '当前没有可用的模型权重' : '新建检测'} onClick={() => setTab('new')}><ScanEye size={15} />新建检测</button>
        <button role="tab" aria-selected={tab === 'history'} className={tab === 'history' ? 'active' : ''} onClick={() => setTab('history')}><History size={15} />历史记录 <span>{history.length}</span></button>
      </nav>
      <div className={`model-test-body${tab === 'history' ? ' history-mode' : ''}`}>
        {tab === 'new' ? <>
          <ModelStrip test={test || rerunSource} artifact={artifact} fallbackAttempt={attemptNumber} bestMetric={bestMetric} />
          {rerunSource && !test && <div className="model-test-rerun-notice"><RotateCcw size={15} /><span>已载入历史图片和参数；将固定使用原检测的权重版本，并创建一条新记录。</span><button className="icon-button" title={artifact && !newTestDisabledReason ? '改用当前权重' : '取消重测'} aria-label={artifact && !newTestDisabledReason ? '改用当前权重' : '取消重测'} onClick={() => { setRerunSource(null); if (!artifact || newTestDisabledReason) setTab('history') }}><X size={14} /></button></div>}
          <div className="model-test-settings">
            <button className="button ghost" disabled={busy} onClick={() => input.current?.click()}><Upload size={16} />{files.length ? `添加图片（${files.length}/10）` : '上传图片'}</button>
            <input ref={input} hidden disabled={busy} type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" multiple onChange={(event) => chooseImages(event.target.files)} />
            <div className="model-test-slider"><label htmlFor="test-confidence">置信度 Conf <strong>{confidence.toFixed(2)}</strong></label><input id="test-confidence" type="range" min="0.01" max="1" step="0.01" value={confidence} disabled={busy} onChange={(event) => setConfidence(Number(event.target.value))} /></div>
            <div className="model-test-slider"><label htmlFor="test-iou">NMS IoU <strong>{iou.toFixed(2)}</strong></label><input id="test-iou" type="range" min="0.01" max="1" step="0.01" value={iou} disabled={busy} onChange={(event) => setIou(Number(event.target.value))} /></div>
            <button className="button primary" disabled={busy || hasPendingUpload || !files.length} onClick={() => void run()}>{busy ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}{completed ? '再次检测' : '运行检测'}</button>
          </div>
          <p className="model-test-help">{busy ? '检测期间不能添加或移除图片' : '可继续拖入图片追加，点击缩略图右上角移除'} · 最多 10 张 · 单张 20 MB / 2000 万像素 · 总大小 100 MB · CPU 检测</p>
          {error && <div className="model-test-error" role="alert">{error}</div>}
          {(stage || busy) && <div className={`model-test-status${failed ? ' failed' : ''}`} role="status">{busy ? <LoaderCircle className="spin" size={14} /> : failed ? <AlertCircle size={14} /> : null}{stage}{completed && ` · Conf ${test.confidence.toFixed(2)} · IoU ${test.iou.toFixed(2)}`}</div>}
          {parametersChanged && <div className="model-test-parameter-notice"><Info size={15} />当前显示的是 Conf {test.confidence.toFixed(2)} · IoU {test.iou.toFixed(2)} 的历史结果；再次检测会新建一条记录。</div>}
          {!images.length ? <button className="model-test-empty" onClick={() => input.current?.click()}><Upload size={32} /><strong>点击或拖入图片</strong><span>支持 JPG、PNG、WEBP、BMP，可多选</span></button>
            : <TestWorkspace test={test} images={images} selected={selected} onSelect={setSelected} onRemove={removeImage} busy={busy} failed={failed} openLightbox={openLightbox} />}
        </> : <div className="model-test-history-layout">
          <aside className="model-test-history-sidebar">
            <div className="model-test-history-toolbar">
              <div className="model-test-history-scope" role="group" aria-label="历史范围">
                <button className={historyScope === 'all' ? 'active' : ''} onClick={() => setHistoryScope('all')}>全部</button>
                {runId && <button className={historyScope === 'current' ? 'active' : ''} onClick={() => setHistoryScope('current')}>当前 Attempt</button>}
              </div>
              <button className="icon-button" disabled={historyLoading} title="刷新历史记录" aria-label="刷新历史记录" onClick={() => void refreshHistory()}>{historyLoading ? <LoaderCircle className="spin" size={15} /> : <RefreshCw size={15} />}</button>
            </div>
            {historyLoading && !history.length ? <div className="model-test-history-empty"><LoaderCircle className="spin" size={24} /><span>正在加载历史记录…</span></div>
              : visibleHistory.length ? <div className="model-test-history-list">{visibleHistory.map((item) => {
                const objects = item.files.reduce((sum, file) => sum + (file.detections || 0), 0)
                const detected = item.files.filter((file) => (file.detections || 0) > 0).length
                return <button key={item.id} className={`model-test-history-item${selectedSummary?.id === item.id ? ' selected' : ''}`} onClick={() => openHistory(item)}>
                  <div><span className={`model-test-history-status ${item.status}`}><i />{modelTestStatusLabels[item.status]}</span><time>{modelTestDate(item.created_at)}</time></div>
                  <strong>{modelTestWeightName(item)} <small>{modelTestAttemptText(item)}</small></strong>
                  <p>{item.files.length} 张图片 · {detected} 张命中 · {objects} 个目标</p>
                  <footer><span>Conf {item.confidence.toFixed(2)} · IoU {item.iou.toFixed(2)}</span>{typeof item.storage_bytes === 'number' && <span><HardDrive size={11} />{formatBytes(item.storage_bytes)}</span>}</footer>
                </button>
              })}{historyNextCursor && <button className="button ghost model-test-history-more" disabled={historyLoading || historyLoadingMore} onClick={() => void loadMoreHistory()}>{historyLoadingMore ? <LoaderCircle className="spin" size={14} /> : <History size={14} />}加载更多</button>}</div> : <div className="model-test-history-empty"><History size={28} /><strong>暂无检测记录</strong><span>{historyScope === 'current' ? '当前 Attempt 还没有检测记录' : '完成一次检测后会保存在这里'}</span></div>}
          </aside>
          <section className="model-test-history-detail">
            {historyError && <div className="model-test-error" role="alert">{historyError}</div>}
            {selectedHistory ? <>
              <div className="model-test-history-heading"><div><span>{modelTestDate(selectedHistory.created_at)}</span><h3>{modelTestWeightName(selectedHistory)}</h3><p>{modelTestAttemptText(selectedHistory)} · Conf {selectedHistory.confidence.toFixed(2)} · IoU {selectedHistory.iou.toFixed(2)}</p></div>{historyDetailLoading && <LoaderCircle className="spin" size={18} />}</div>
              <ModelStrip test={selectedHistory} artifact={artifact} history />
              {selectedHistory.error && <div className="model-test-error" role="alert">{selectedHistory.error}</div>}
              {selectedHistory.status === 'completed' && !hasExactModelVersion(selectedHistory) && <div className="model-test-parameter-notice" role="status"><Info size={15} />该历史记录缺少可验证的原权重版本，已有结果仍可查看和下载，但不能按原权重重测。</div>}
              <TestWorkspace test={selectedHistory} selected={historyImage} onSelect={setHistoryImage} busy={activeModelTestStatuses.has(selectedHistory.status)} failed={selectedHistory.status === 'failed' || selectedHistory.status === 'cancelled'} openLightbox={openLightbox} />
            </> : selectedSummary && (historyDetailLoading || !historyError) ? <div className="model-test-history-detail-empty"><LoaderCircle className="spin" size={24} /><span>正在加载检测详情…</span></div> : !historyLoading && <div className="model-test-history-detail-empty"><ScanEye size={34} /><span>选择左侧记录查看详情</span></div>}
          </section>
        </div>}
      </div>
      <footer>
        {tab === 'new' ? <>
          <span>{completed ? `${currentName || '当前图片'} · ${modelTestDetectionLabel(test.files[selected]?.detections ?? null)} · 已保存到历史记录` : hasPendingUpload && !busy ? '本次上传尚未完成，请取消后再创建检测' : hasPendingUpload ? '正在上传或提交；关闭后请在历史记录确认最终状态' : busy ? '关闭只停止本页等待，已提交的检测会继续运行' : '阈值越高，显示的检测结果越严格'}</span>
          {hasPendingUpload && !busy && <button className="button danger" onClick={() => void cancelPendingUpload()}><X size={15} />取消本次上传</button>}
          {completed && <a className="button ghost" href={`/api/model-tests/${test.id}/download`} download><FileArchive size={15} />下载全部</a>}
          {currentResult && <a className="button primary" href={`${currentResult}?download=true`} download><Download size={15} />当前结果图</a>}
        </> : <>
          <span>{selectedHistory ? `${modelTestDate(selectedHistory.created_at)} · ${modelTestStatusLabels[selectedHistory.status]}` : `共 ${visibleHistory.length} 条记录`}</span>
          {selectedHistory?.status === 'completed' && <a className="button ghost" href={`/api/model-tests/${selectedHistory.id}/download`} download><FileArchive size={15} />下载全部</a>}
          {selectedHistory?.status === 'completed' && <a className="button ghost" href={`/api/model-tests/${selectedHistory.id}/images/${historyImage}/result?download=true`} download><Download size={15} />当前结果图</a>}
          {selectedHistory?.status === 'completed' && <button className="button primary" disabled={reusingId === selectedHistory.id || !hasExactModelVersion(selectedHistory)} title={hasExactModelVersion(selectedHistory) ? '使用原图和原权重版本创建新记录' : '历史记录缺少可验证的原权重版本'} onClick={() => void reuseHistory(selectedHistory)}>{reusingId === selectedHistory.id ? <LoaderCircle className="spin" size={15} /> : <RotateCcw size={15} />}使用这些图片重测</button>}
          {selectedHistory && <button className="button danger" disabled={deletingId === selectedHistory.id} onClick={() => void deleteHistory(selectedHistory)}>{deletingId === selectedHistory.id ? <LoaderCircle className="spin" size={15} /> : <Trash2 size={15} />}删除记录</button>}
        </>}
        <button className="button ghost" onClick={onClose}>关闭</button>
      </footer>
      {enlarged && <div ref={lightbox} className="model-test-lightbox" role="dialog" aria-modal="true" aria-label={`放大查看${enlarged.title}`} tabIndex={-1} onClick={() => setEnlarged(null)}><button className="icon-button" aria-label="关闭图片放大" onClick={() => setEnlarged(null)}><X size={24} /></button><img src={enlarged.url} alt={enlarged.title} onClick={(event) => event.stopPropagation()} /><span>{enlarged.title} · Esc 关闭</span></div>}
    </div>
  </div>
}
