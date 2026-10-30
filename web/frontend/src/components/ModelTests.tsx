import { applyHistoryRefresh, bridgeHistoryHead, historyNeedsDetailRefresh, refreshActiveHistoryTail, retainedHistorySelection } from '../modelTestHistory'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertCircle, Box, CheckCircle2, ChevronDown, Cpu, Download, FileArchive, FolderOpen,
  HardDrive, History, ImagePlus, Images, Info, LoaderCircle, Play, RefreshCw,
  RotateCcw, ScanEye, Server, SlidersHorizontal, Sparkles, Trash2, Upload, X,
} from 'lucide-react'
import { api, formatBytes, statusLabel } from '../api'
import type {
  Artifact, ComputeNode, ModelAsset, ModelTest, ModelTestSummary,
  TaskGroup, TrainingRun, TrainingTask,
} from '../types'
import {
  activeModelTestStatuses, createModelTestRequestId, createOrRecoverModelTest, modelTestDate, modelTestParameterText,
  modelTestExecutionLabel, modelTestRequestMayHaveReachedServer, modelTestResultLabel, modelTestStatusLabels, modelTestTaskLabel, modelTestTaskType, modelTestWeightName,
  startOrRecoverModelTest, type SelectedImage, TestWorkspace, uploadPendingModelTestImages,
} from './ModelTestShared'
import { RegisteredModelLibrary, TrainingTaskLibrary } from './ModelLibrary'
import '../model-tests.css'

type SourceType = 'local_path' | 'training_artifact' | 'remote_path'
type PageView = 'new' | 'history'

const sourceLabels: Record<SourceType, string> = {
  local_path: '本机模型',
  training_artifact: '训练产物',
  remote_path: '远程模型',
}

const collapsedHistoryGroupsKey = 'yolo-model-test-history-collapsed-groups-v1'

function initialCollapsedHistoryGroups() {
  try {
    const value = JSON.parse(window.localStorage.getItem(collapsedHistoryGroupsKey) || '[]')
    return new Set<string>(Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [])
  } catch { return new Set<string>() }
}

const assetStatusLabels: Record<ModelAsset['status'], string> = {
  preparing: '准备中', ready: '已就绪', failed: '准备失败',
}

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException('操作已取消', 'AbortError')); return }
    const timer = window.setTimeout(() => { cleanup(); resolve() }, ms)
    const abort = () => { window.clearTimeout(timer); cleanup(); reject(new DOMException('操作已取消', 'AbortError')) }
    const cleanup = () => signal.removeEventListener('abort', abort)
    signal.addEventListener('abort', abort, { once: true })
  })
}

function upsertAsset(items: ModelAsset[], asset: ModelAsset) {
  return [asset, ...items.filter((item) => item.id !== asset.id)]
}

function modelSourceType(test: ModelTestSummary): SourceType {
  if (test.model_source_type === 'remote_path') return 'remote_path'
  if (test.model_source_type === 'local_path') return 'local_path'
  if (test.model_asset_id && test.execution_node_id) return 'remote_path'
  if (test.model_asset_id) return 'local_path'
  return 'training_artifact'
}

function sourceDescription(test: ModelTestSummary) {
  const sourceType = modelSourceType(test)
  if (sourceType === 'training_artifact') {
    const attempt = test.source_attempt_number ?? test.model.attempt_number ?? test.attempt_number
    const task = test.task_name?.trim()
    const source = task ? `训练任务 · ${task}` : '训练产物'
    return attempt ? `${source} · Attempt ${String(attempt).padStart(3, '0')}` : source
  }
  if (sourceType === 'remote_path') return `远程服务器 · ${test.execution_node_name || '已登记节点'}`
  return '本机服务器 · 模型目录'
}

function historyGroupFor(
  test: ModelTestSummary,
  tasksById: Map<string, TrainingTask>,
  groupsById: Map<string, TaskGroup>,
  nodesById: Map<string, ComputeNode>,
) {
  const snapshot = test.model.history_group
  if (test.task_id) {
    const task = tasksById.get(test.task_id)
    if (task && !task.group_id) return { key: 'ungrouped:tasks', name: '未分组', order: 1_000_000 }
    const groupId = task?.group_id
    const group = groupId ? groupsById.get(groupId) : undefined
    if (group) return { key: `task_group:${group.id}`, name: group.name, order: group.display_order }
    if (groupId && snapshot?.kind === 'task_group' && snapshot.id === groupId) {
      return { key: `task_group:${groupId}`, name: snapshot.name, order: 500_000 }
    }
    if (!task && snapshot?.kind === 'task_group' && snapshot.name) {
      return { key: `task_group:${snapshot.id || snapshot.name}`, name: snapshot.name, order: 500_000 }
    }
    return { key: task ? `task_group:${groupId}` : 'ungrouped:missing-task', name: task ? '任务分组不可用' : '已删除或不可用任务', order: 1_000_000 }
  }
  if (modelSourceType(test) === 'remote_path') {
    const nodeId = test.execution_node_id || snapshot?.execution_node_id || null
    const node = (nodeId ? nodesById.get(nodeId)?.name : null)
      || test.execution_node_name || snapshot?.execution_node_name || test.model.execution_node_name || '已登记节点'
    return { key: `standalone_model:remote:${nodeId || node}`, name: `独立模型 · ${node}`, order: 1_000_001 }
  }
  return { key: 'standalone_model:local', name: '独立模型 · 本机模型', order: 1_000_001 }
}

function testHasExactModel(test: ModelTest) {
  return Boolean(test.artifact_id || test.model_asset_id)
    && typeof test.model_sha256 === 'string'
    && /^[0-9a-f]{64}$/i.test(test.model_sha256)
    && typeof test.model_size === 'number'
    && Number.isSafeInteger(test.model_size)
    && test.model_size >= 0
}

function HistoryImagePreview({ test }: { test: ModelTestSummary }) {
  const [failed, setFailed] = useState(false)
  const index = test.files.findIndex((file) => file.uploaded)
  if (index < 0 || failed) {
    return <span className="model-test-history-preview placeholder" aria-hidden="true"><Images size={18} /></span>
  }
  return <span className="model-test-history-preview">
    <img
      src={`/api/model-tests/${test.id}/images/${index}/original`}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
    />
  </span>
}

export function ModelTests() {
  const [view, setView] = useState<PageView>('new')
  const [sourceType, setSourceType] = useState<SourceType>('local_path')
  const [assets, setAssets] = useState<ModelAsset[]>([])
  const [localRoots, setLocalRoots] = useState<string[]>([])
  const [assetsLoading, setAssetsLoading] = useState(true)
  const [sourceBusy, setSourceBusy] = useState(false)
  const [sourceError, setSourceError] = useState('')
  const [sourceNotice, setSourceNotice] = useState('')
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null)
  const [removalAsset, setRemovalAsset] = useState<ModelAsset | null>(null)
  const [removingAssetId, setRemovingAssetId] = useState<string | null>(null)
  const [removalError, setRemovalError] = useState('')
  const [localPath, setLocalPath] = useState('')
  const [localName, setLocalName] = useState('')
  const [remotePath, setRemotePath] = useState('')
  const [remoteName, setRemoteName] = useState('')
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [remoteNodeId, setRemoteNodeId] = useState('')

  const [tasks, setTasks] = useState<TrainingTask[]>([])
  const [taskGroups, setTaskGroups] = useState<TaskGroup[]>([])
  const [trainingTaskId, setTrainingTaskId] = useState('')
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [trainingRunId, setTrainingRunId] = useState('')
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [artifactId, setArtifactId] = useState<number | null>(null)
  const [trainingSourceLoading, setTrainingSourceLoading] = useState(false)

  const [images, setImages] = useState<SelectedImage[]>([])
  const [confidence, setConfidence] = useState(0.25)
  const [iou, setIou] = useState(0.70)
  const [selectedImage, setSelectedImage] = useState(0)
  const [test, setTest] = useState<ModelTest | null>(null)
  const [running, setRunning] = useState(false)
  const [runStage, setRunStage] = useState('')
  const [runError, setRunError] = useState('')
  const [dragging, setDragging] = useState(false)

  const [history, setHistory] = useState<ModelTestSummary[]>([])
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyLoadingMore, setHistoryLoadingMore] = useState(false)
  const [historyCursor, setHistoryCursor] = useState<string | null>(null)
  const [historyError, setHistoryError] = useState('')
  const [historySource, setHistorySource] = useState<SourceType | ''>('')
  const [historyStatus, setHistoryStatus] = useState<ModelTest['status'] | ''>('')
  const [historySelectedId, setHistorySelectedId] = useState<string | null>(null)
  const [historyDetails, setHistoryDetails] = useState<Record<string, ModelTest>>({})
  const [historyDetailLoading, setHistoryDetailLoading] = useState(false)
  const [historyImage, setHistoryImage] = useState(0)
  const [collapsedHistoryGroups, setCollapsedHistoryGroups] = useState<Set<string>>(initialCollapsedHistoryGroups)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [reusingId, setReusingId] = useState<string | null>(null)
  const [rerunSource, setRerunSource] = useState<ModelTest | null>(null)

  const [enlarged, setEnlarged] = useState<{ url: string; title: string } | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const nextImageId = useRef(0)
  const previewUrls = useRef(new Set<string>())
  const runController = useRef<AbortController | null>(null)
  const sourceController = useRef<AbortController | null>(null)
  const removalController = useRef<AbortController | null>(null)
  const assetsRefreshVersion = useRef(0)
  const removalDialog = useRef<HTMLDialogElement>(null)
  const historyController = useRef<AbortController | null>(null)
  const detailController = useRef<AbortController | null>(null)
  const reuseController = useRef<AbortController | null>(null)
  const historyLoadMoreController = useRef<AbortController | null>(null)
  const historyDetailRef = useRef<HTMLElement>(null)
  const historyItemsRef = useRef<ModelTestSummary[]>([])
  const historyTailCursor = useRef(0)
  const historyRefreshVersion = useRef(0)
  const historyLoadingVersion = useRef(0)
  const historySourceRef = useRef<SourceType | ''>('')
  const historyStatusRef = useRef<ModelTest['status'] | ''>('')
  const pendingUploadId = useRef<string | null>(null)
  const dragDepth = useRef(0)
  const lightbox = useRef<HTMLDivElement>(null)
  const lightboxOpener = useRef<HTMLElement | null>(null)

  const files = useMemo(() => images.map(({ file }) => file), [images])
  const selectedAsset = assets.find((asset) => asset.id === selectedAssetId) || null
  const selectedArtifact = artifacts.find((artifact) => artifact.id === artifactId) || null
  const selectedRun = runs.find((run) => run.id === trainingRunId) || null
  const selectedTask = tasks.find((task) => task.id === trainingTaskId) || null
  const remoteNodes = nodes.filter((node) => node.kind === 'agent' && node.enabled)
  const tasksById = useMemo(() => new Map(tasks.map((task) => [task.id, task])), [tasks])
  const groupsById = useMemo(() => new Map(taskGroups.map((group) => [group.id, group])), [taskGroups])
  const nodesById = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes])
  const groupedHistory = useMemo(() => {
    const sections = new Map<string, { key: string; name: string; order: number; items: ModelTestSummary[] }>()
    for (const item of history) {
      const group = historyGroupFor(item, tasksById, groupsById, nodesById)
      const section = sections.get(group.key) || { ...group, items: [] }
      section.items.push(item)
      sections.set(group.key, section)
    }
    return [...sections.values()].sort((left, right) => left.order - right.order || left.name.localeCompare(right.name, 'zh-CN'))
  }, [groupsById, history, nodesById, tasksById])
  historySourceRef.current = historySource
  historyStatusRef.current = historyStatus

  useEffect(() => {
    try { window.localStorage.setItem(collapsedHistoryGroupsKey, JSON.stringify([...collapsedHistoryGroups])) } catch { /* Storage is optional. */ }
  }, [collapsedHistoryGroups])

  const refreshAssets = useCallback(async (signal?: AbortSignal) => {
    const version = ++assetsRefreshVersion.current
    setAssetsLoading(true)
    try {
      const page = await api.modelAssets(signal)
      if (signal?.aborted || version !== assetsRefreshVersion.current) return
      const items = page.items
      setAssets(items)
      setLocalRoots(page.local_roots?.length ? page.local_roots : page.local_root ? [page.local_root] : [])
      setSelectedAssetId((current) => current && items.some((asset) => asset.id === current) ? current : null)
    } catch (reason) {
      if (!signal?.aborted && version === assetsRefreshVersion.current) setSourceError(reason instanceof Error ? reason.message : '读取已登记模型失败')
    } finally {
      if (!signal?.aborted && version === assetsRefreshVersion.current) setAssetsLoading(false)
    }
  }, [])

  const assetsNeedRefresh = assets.some((asset) => asset.status === 'preparing' || asset.removal_blocked_reason)
  useEffect(() => {
    if (!assetsNeedRefresh || sourceBusy || removingAssetId) return
    const abort = new AbortController()
    let timer = 0
    const poll = async () => {
      await refreshAssets(abort.signal)
      if (!abort.signal.aborted) timer = window.setTimeout(() => void poll(), 3000)
    }
    timer = window.setTimeout(() => void poll(), 3000)
    return () => { window.clearTimeout(timer); abort.abort() }
  }, [assetsNeedRefresh, sourceBusy, removingAssetId, refreshAssets])

  historyItemsRef.current = history

  const refreshHistory = useCallback(async (signal?: AbortSignal, quiet = false) => {
    const version = ++historyRefreshVersion.current
    const loadingVersion = quiet ? null : ++historyLoadingVersion.current
    if (loadingVersion !== null) {
      historyLoadMoreController.current?.abort()
      setHistoryLoading(true)
      setHistoryLoadingMore(false)
    }
    try {
      const page = await api.globalModelTests(
        signal, undefined, 50, historySourceRef.current, historyStatusRef.current,
      )
      const head = quiet ? await bridgeHistoryHead(historyItemsRef.current, page, (cursor) => api.globalModelTests(signal, cursor, 50, historySourceRef.current, historyStatusRef.current)) : page.items
      const tail = quiet ? await refreshActiveHistoryTail(historyItemsRef.current, head,
        activeModelTestStatuses, (id) => api.modelTest(id, signal || new AbortController().signal), historyTailCursor.current)
        : { updates: [], removed: [], next: 0 }
      if (signal?.aborted || version !== historyRefreshVersion.current) return
      historyTailCursor.current = tail.next
      if (quiet) {
        setHistory((current) => applyHistoryRefresh(current, head, tail, historyStatusRef.current))
        setHistorySelectedId((current) => current || page.items[0]?.id || null)
      } else {
        setHistory(page.items)
        setHistoryCursor(page.next_cursor)
        setHistorySelectedId((current) => retainedHistorySelection(current, page.items))
      }
      setHistoryError('')
    } catch (reason) {
      if (!signal?.aborted && version === historyRefreshVersion.current) setHistoryError(reason instanceof Error ? reason.message : '加载全局历史失败')
    } finally {
      if (!signal?.aborted && loadingVersion !== null && loadingVersion === historyLoadingVersion.current) setHistoryLoading(false)
    }
  }, [])

  useEffect(() => {
    const abort = new AbortController()
    void refreshAssets(abort.signal)
    void api.tasks(abort.signal).then((items) => {
      if (abort.signal.aborted) return
      setTasks(items)
      const latest = [...items].sort((left, right) => right.updated_at.localeCompare(left.updated_at))[0]
      setTrainingTaskId((current) => current && items.some((item) => item.id === current) ? current : latest?.id || '')
    }).catch((reason) => { if (!abort.signal.aborted) setSourceError(reason instanceof Error ? reason.message : '读取训练任务失败') })
    void api.taskGroups(abort.signal).then((items) => {
      if (!abort.signal.aborted) setTaskGroups(items)
    }).catch((reason) => { if (!abort.signal.aborted) setSourceError(reason instanceof Error ? reason.message : '读取任务分组失败') })
    void api.computeNodes(abort.signal).then((items) => {
      if (abort.signal.aborted) return
      setNodes(items)
      const first = items.find((node) => node.kind === 'agent' && node.enabled)
      setRemoteNodeId((current) => current || first?.id || '')
    }).catch((reason) => { if (!abort.signal.aborted) setSourceError(reason instanceof Error ? reason.message : '读取远程节点失败') })
    return () => abort.abort()
  }, [refreshAssets])

  useEffect(() => {
    if (!trainingTaskId) { setRuns([]); setTrainingRunId(''); return }
    const abort = new AbortController()
    setTrainingSourceLoading(true)
    void api.runs(trainingTaskId, abort.signal).then((items) => {
      if (abort.signal.aborted) return
      const ordered = [...items].sort((left, right) => right.attempt_number - left.attempt_number)
      setRuns(ordered)
      setTrainingRunId((current) => current && ordered.some((run) => run.id === current) ? current : ordered[0]?.id || '')
      setSourceError('')
    }).catch((reason) => { if (!abort.signal.aborted) setSourceError(reason instanceof Error ? reason.message : '读取 Attempt 失败') })
      .finally(() => { if (!abort.signal.aborted) setTrainingSourceLoading(false) })
    return () => abort.abort()
  }, [trainingTaskId])

  useEffect(() => {
    if (!trainingTaskId || !trainingRunId) { setArtifacts([]); setArtifactId(null); return }
    const abort = new AbortController()
    setTrainingSourceLoading(true)
    void api.runArtifacts(trainingTaskId, trainingRunId, abort.signal).then((items) => {
      if (abort.signal.aborted) return
      const weights = items.filter((item) => /\.pt$/i.test(item.name) || /\.pt$/i.test(item.path))
        .sort((left, right) => Number(/best\.pt$/i.test(right.name)) - Number(/best\.pt$/i.test(left.name)))
      setArtifacts(weights)
      setArtifactId((current) => current && weights.some((item) => item.id === current) ? current : weights[0]?.id || null)
      setSourceError('')
    }).catch((reason) => { if (!abort.signal.aborted) setSourceError(reason instanceof Error ? reason.message : '读取模型产物失败') })
      .finally(() => { if (!abort.signal.aborted) setTrainingSourceLoading(false) })
    return () => abort.abort()
  }, [trainingRunId, trainingTaskId])

  useEffect(() => {
    historyController.current?.abort()
    const abort = new AbortController()
    historyController.current = abort
    void refreshHistory(abort.signal)
    return () => abort.abort()
  }, [historySource, historyStatus, refreshHistory])

  const hasActiveHistory = history.some((item) => activeModelTestStatuses.has(item.status))
  useEffect(() => {
    if (!hasActiveHistory) return
    const abort = new AbortController()
    let timer = 0
    const poll = async () => {
      await refreshHistory(abort.signal, true)
      if (!abort.signal.aborted) timer = window.setTimeout(() => void poll(), 3000)
    }
    timer = window.setTimeout(() => void poll(), 3000)
    return () => { window.clearTimeout(timer); abort.abort() }
  }, [hasActiveHistory, refreshHistory])

  const selectedSummary = history.find((item) => item.id === historySelectedId) || history[0] || null
  const selectedHistory = selectedSummary ? historyDetails[selectedSummary.id] || null : null

  useEffect(() => {
    setHistoryImage(0)
  }, [selectedSummary?.id])

  const loadHistoryDetail = useCallback(async (id: string) => {
    detailController.current?.abort()
    const abort = new AbortController()
    detailController.current = abort
    setHistoryDetailLoading(true)
    try {
      const detail = await api.modelTest(id, abort.signal)
      if (!abort.signal.aborted) {
        setHistoryDetails((current) => ({ ...current, [detail.id]: detail }))
        setHistory((current) => current.flatMap((item) => item.id !== detail.id ? [item] : historyStatusRef.current && detail.status !== historyStatusRef.current ? [] : [{ ...item, ...detail }]))
      }
    } catch (reason) {
      if (!abort.signal.aborted) setHistoryError(reason instanceof Error ? reason.message : '加载测试详情失败')
    } finally {
      if (!abort.signal.aborted) setHistoryDetailLoading(false)
    }
  }, [])

  useEffect(() => {
    if (selectedSummary && historyNeedsDetailRefresh(selectedSummary, selectedHistory)) void loadHistoryDetail(selectedSummary.id)
  }, [loadHistoryDetail, selectedHistory, selectedSummary])

  useEffect(() => () => detailController.current?.abort(), [selectedSummary?.id])

  const selectedHistoryId = selectedSummary?.id
  const selectedHistoryActive = !!selectedSummary && (activeModelTestStatuses.has(selectedSummary.status)
    || !!selectedHistory && activeModelTestStatuses.has(selectedHistory.status))
  useEffect(() => {
    if (!selectedHistoryId || !selectedHistoryActive) return
    let cancelled = false
    let timer = 0
    const poll = async () => {
      await loadHistoryDetail(selectedHistoryId)
      if (!cancelled) timer = window.setTimeout(() => void poll(), 3000)
    }
    timer = window.setTimeout(() => void poll(), 3000)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [loadHistoryDetail, selectedHistoryActive, selectedHistoryId])

  useEffect(() => {
    const ownedUrls = previewUrls.current
    const abandonPending = () => {
      const id = pendingUploadId.current
      if (!id) return
      pendingUploadId.current = null
      void api.abandonModelTest(id, true).catch(() => undefined)
    }
    const pageHide = (event: PageTransitionEvent) => {
      if (event.persisted) return
      abandonPending()
      runController.current?.abort()
    }
    window.addEventListener('pagehide', pageHide)
    return () => {
      window.removeEventListener('pagehide', pageHide)
      abandonPending()
      runController.current?.abort()
      sourceController.current?.abort()
      removalController.current?.abort()
      historyController.current?.abort()
      detailController.current?.abort()
      reuseController.current?.abort()
      historyLoadMoreController.current?.abort()
      const urls = [...ownedUrls]
      ownedUrls.clear()
      window.setTimeout(() => urls.forEach((url) => URL.revokeObjectURL(url)), 0)
    }
  }, [])

  useEffect(() => {
    if (!enlarged) return
    const opener = lightboxOpener.current
    lightbox.current?.querySelector<HTMLElement>('button:not(:disabled)')?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        setEnlarged(null)
      } else if (event.key === 'Tab') {
        event.preventDefault()
        lightbox.current?.querySelector<HTMLElement>('button:not(:disabled)')?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      lightboxOpener.current = null
      if (opener?.isConnected) opener.focus()
    }
  }, [enlarged])

  useEffect(() => {
    const dialog = removalDialog.current
    if (!removalAsset || !dialog) return
    dialog.showModal()
    return () => dialog.close()
  }, [removalAsset])

  async function removeRegisteredModel() {
    if (!removalAsset || removingAssetId) return
    const asset = removalAsset
    const abort = new AbortController()
    removalController.current = abort
    assetsRefreshVersion.current += 1
    setAssetsLoading(false)
    setRemovingAssetId(asset.id)
    setRemovalError('')
    try {
      await api.removeModelAsset(asset.id, abort.signal)
      if (abort.signal.aborted) return
      assetsRefreshVersion.current += 1
      setAssets((current) => current.filter((item) => item.id !== asset.id))
      if (selectedAssetId === asset.id || rerunSource?.model_asset_id === asset.id) {
        setSelectedAssetId(null)
        resetCurrentResult()
      }
      setSourceError('')
      setSourceNotice(`已移除“${asset.name}”的模型登记；原始权重和测试历史已保留。可重新登记；原模型版本仍可用时，可从历史重测。`)
      setRemovalAsset(null)
    } catch (reason) {
      if (!abort.signal.aborted) {
        setRemovalError(reason instanceof Error ? reason.message : '移除模型登记失败，请重试')
        void refreshAssets(abort.signal)
      }
    } finally {
      if (!abort.signal.aborted) setRemovingAssetId(null)
      if (removalController.current === abort) removalController.current = null
    }
  }

  function resetCurrentResult() {
    setTest(null)
    setRunError('')
    setRunStage('')
    setEnlarged(null)
    setRerunSource(null)
  }

  function chooseSource(next: SourceType) {
    if (running) return
    sourceController.current?.abort()
    sourceController.current = null
    setSourceBusy(false)
    setSourceType(next); setSelectedAssetId(null); setSourceError(''); setSourceNotice('')
    resetCurrentResult()
  }

  async function waitForAssetReady(initial: ModelAsset, signal: AbortSignal, retry = false) {
    let current = initial
    const deadline = Date.now() + 10 * 60_000
    if (current.status === 'ready') return current
    if (current.status === 'failed' && !retry) throw new Error(current.error || '模型准备失败')
    const preparation = await api.prepareModelAsset(current.id, retry, signal)
    if (preparation.status === 'failed') throw new Error(preparation.error || current.error || '模型准备失败')
    while (current.status !== 'ready') {
      if (Date.now() > deadline) throw new Error('模型准备时间较长，请稍后刷新已登记模型查看状态')
      await sleep(1500, signal)
      const version = ++assetsRefreshVersion.current
      const page = await api.modelAssets(signal)
      if (signal.aborted) throw new DOMException('操作已取消', 'AbortError')
      if (version === assetsRefreshVersion.current) {
        setAssets(page.items)
        setLocalRoots(page.local_roots?.length ? page.local_roots : page.local_root ? [page.local_root] : [])
        setAssetsLoading(false)
      }
      const refreshed = page.items.find((asset) => asset.id === current.id)
      if (!refreshed) throw new Error('已登记模型不存在，请重新选择')
      current = refreshed
      if (current.status === 'failed') throw new Error(current.error || '模型准备失败')
    }
    return current
  }

  async function registerPathModel() {
    if (sourceBusy || (sourceType !== 'local_path' && sourceType !== 'remote_path')) return
    const path = (sourceType === 'local_path' ? localPath : remotePath).trim()
    const name = (sourceType === 'local_path' ? localName : remoteName).trim()
    if (!path) { setSourceError('请输入所选服务器上的 .pt 模型路径'); return }
    if (sourceType === 'remote_path' && !remoteNodeId) { setSourceError('请先选择远程服务器'); return }
    sourceController.current?.abort()
    const abort = new AbortController()
    sourceController.current = abort
    setSourceBusy(true); setSourceError(''); setSourceNotice(sourceType === 'remote_path' ? '正在校验并从远程服务器准备模型…' : '正在校验本机模型…')
    try {
      let asset = sourceType === 'local_path'
        ? await api.createLocalModelAsset(path, name || undefined, abort.signal)
        : await api.createRemoteModelAsset(remoteNodeId, path, name || undefined, abort.signal)
      if (abort.signal.aborted) return
      assetsRefreshVersion.current += 1
      setAssetsLoading(false)
      setAssets((items) => upsertAsset(items, asset)); setSelectedAssetId(asset.id)
      if (asset.status !== 'ready') asset = await waitForAssetReady(asset, abort.signal, asset.status === 'failed')
      setSelectedAssetId(asset.id)
      setSourceNotice(`模型已就绪 · ${formatBytes(asset.size || 0)} · SHA-256 ${asset.sha256?.slice(0, 12) || '待校验'}…`)
      resetCurrentResult()
    } catch (reason) {
      if (!abort.signal.aborted) { setSourceError(reason instanceof Error ? reason.message : '模型校验失败'); setSourceNotice('') }
    } finally {
      if (!abort.signal.aborted) setSourceBusy(false)
      if (sourceController.current === abort) sourceController.current = null
    }
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
    if (!added.length || running) return
    const chosen = [...files, ...added]
    if (chosen.length > 10 || chosen.reduce((sum, file) => sum + file.size, 0) > 100 * 1024 ** 2) {
      setRunError('每次最多 10 张图片，总大小不超过 100 MB'); return
    }
    if (chosen.some((file) => !/\.(jpe?g|png|webp|bmp)$/i.test(file.name) || !file.size || file.size > 20 * 1024 ** 2)) {
      setRunError('请选择 JPG、PNG、WEBP 或 BMP 图片，单张不超过 20 MB'); return
    }
    const additions = added.map((file) => {
      const previewUrl = URL.createObjectURL(file)
      previewUrls.current.add(previewUrl)
      return { id: nextImageId.current++, file, previewUrl }
    })
    setImages((current) => [...current, ...additions])
    setRunError(''); setTest(null); setRunStage(''); setEnlarged(null)
  }

  function removeImage(index: number) {
    if (running) return
    const removed = images[index]
    const remaining = images.filter((_, current) => current !== index)
    setImages(remaining)
    if (removed) {
      previewUrls.current.delete(removed.previewUrl)
      window.setTimeout(() => URL.revokeObjectURL(removed.previewUrl), 0)
    }
    setSelectedImage(Math.max(0, Math.min(selectedImage > index ? selectedImage - 1 : selectedImage, remaining.length - 1)))
    setRunError(''); setTest(null); setRunStage(''); setEnlarged(null)
  }

  function resetDrag() { dragDepth.current = 0; setDragging(false) }
  function dragEnter(event: React.DragEvent) {
    if (view !== 'new' || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation()
    if (running) return
    dragDepth.current += 1; setDragging(true)
  }
  function dragOver(event: React.DragEvent) {
    if (view !== 'new' || !event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation(); event.dataTransfer.dropEffect = running ? 'none' : 'copy'
  }
  function dragLeave(event: React.DragEvent) {
    if (!event.dataTransfer.types.includes('Files')) return
    event.preventDefault(); event.stopPropagation(); dragDepth.current = Math.max(0, dragDepth.current - 1)
    if (!dragDepth.current) setDragging(false)
  }
  function dropImages(event: React.DragEvent) {
    if (view !== 'new') return
    event.preventDefault(); event.stopPropagation(); resetDrag()
    if (running) return
    if (Array.from(event.dataTransfer.items).some((item) => item.webkitGetAsEntry?.()?.isDirectory)) {
      setRunError('暂不支持拖入文件夹，请选择图片文件'); return
    }
    chooseImages(event.dataTransfer.files)
  }

  async function prepareTrainingArtifact(signal: AbortSignal) {
    if (!selectedArtifact) throw new Error('请选择可用的 .pt 训练产物')
    const deadline = Date.now() + 5 * 60_000
    let first = true
    while (true) {
      const prepared = await api.prepareArtifact(selectedArtifact.id, first, signal)
      first = false
      if (prepared.status === 'ready') return
      if (prepared.status === 'failed') throw new Error(prepared.message || '训练产物准备失败')
      if (Date.now() > deadline) throw new Error('远程权重尚未准备完成，请确认 Agent 在线后重试')
      await sleep(1500, signal)
    }
  }

  async function runModelTest() {
    if (running || removingAssetId || pendingUploadId.current || !files.length) return
    const rerun = rerunSource
    if (rerun && !testHasExactModel(rerun)) { setRunError('该历史记录没有可验证的原模型版本'); return }
    if (!rerun && sourceType === 'training_artifact' && !selectedArtifact) { setRunError('请先选择训练产物'); return }
    if (!rerun && sourceType !== 'training_artifact' && !selectedAsset) { setRunError('请先校验并选择模型路径'); return }
    runController.current?.abort()
    const abort = new AbortController()
    runController.current = abort
    const { signal } = abort
    let createdId: string | null = null
    let lastKnown: ModelTest | null = null
    resetDrag(); setRunning(true); setRunError(''); setTest(null); setEnlarged(null)
    try {
      let asset = selectedAsset
      if (rerun) {
        setRunStage('正在固定历史记录的原模型版本…')
      } else if (sourceType === 'training_artifact') {
        setRunStage('正在准备训练产物，远程权重首次使用需要传输…')
        await prepareTrainingArtifact(signal)
      } else if (asset && asset.status !== 'ready') {
        setRunStage('正在准备模型副本…')
        asset = await waitForAssetReady(asset, signal, asset.status === 'failed')
      }
      setRunStage('正在创建测试记录…')
      const requestId = createModelTestRequestId()
      createdId = requestId
      pendingUploadId.current = requestId
      const created = await createOrRecoverModelTest(requestId, files, signal, () => (
        rerun?.artifact_id
          ? api.createModelTest(
            rerun.artifact_id, confidence, iou, files, requestId, signal, rerun.run_id || undefined, rerun.id,
          )
          : rerun?.model_asset_id
            ? api.createStandaloneModelTest(
              rerun.model_asset_id, confidence, iou, files, requestId, signal, rerun.id,
            )
            : sourceType === 'training_artifact' && selectedArtifact
              ? api.createModelTest(selectedArtifact.id, confidence, iou, files, requestId, signal, trainingRunId || undefined)
              : api.createStandaloneModelTest(asset!.id, confidence, iou, files, requestId, signal)
      ))
      lastKnown = created
      setTest(created); setRunStage(modelTestStatusLabels[created.status])
      await refreshHistory(signal, true)
      let current = await uploadPendingModelTestImages(created, files, signal, (index, file) => {
        setRunStage(`正在上传 ${index + 1}/${files.length}：${file.name}`)
      })
      lastKnown = current
      if (current.status === 'uploading') {
        setRunStage('正在提交测试…')
        current = await startOrRecoverModelTest(current, files, signal)
        lastKnown = current
      }
      if (current.status !== 'uploading') pendingUploadId.current = null
      await refreshHistory(signal, true)
      const deadline = Date.now() + 15 * 60_000
      while (true) {
        lastKnown = current
        setTest(current); setRunStage(modelTestStatusLabels[current.status])
        if (current.status === 'failed') throw new Error(current.error || '测试失败')
        if (current.status === 'cancelled') throw new Error('测试已取消')
        if (current.status === 'completed') break
        if (Date.now() > deadline) throw new Error('等待测试超时，任务会继续运行，请稍后在历史记录中查看')
        await sleep(1500, signal)
        current = await api.modelTest(requestId, signal)
      }
      await refreshHistory(signal, true)
    } catch (reason) {
      if (!signal.aborted) {
        const message = reason instanceof Error ? reason.message : '模型测试请求失败'
        const uncertainCreate = createdId !== null && lastKnown === null && modelTestRequestMayHaveReachedServer(reason)
        if (createdId && (lastKnown !== null || uncertainCreate)) {
          try {
            lastKnown = await api.modelTest(createdId, signal)
            setTest(lastKnown)
          } catch {
            // Keep the last confirmed server state when the status request also fails.
          }
        }
        if (createdId && lastKnown === null && !uncertainCreate && pendingUploadId.current === createdId) {
          pendingUploadId.current = null
        }
        if (lastKnown?.status === 'failed') {
          setRunStage(modelTestStatusLabels.failed)
          setRunError(lastKnown.error || message)
        } else if (lastKnown?.status === 'cancelled') {
          pendingUploadId.current = null
          setRunStage(modelTestStatusLabels.cancelled)
          setRunError(lastKnown.error || message)
        } else if (lastKnown?.status === 'completed') {
          pendingUploadId.current = null
          setRunStage(modelTestStatusLabels.completed)
          setRunError('')
        } else if (lastKnown) {
          const uploadIncomplete = lastKnown.status === 'uploading'
          if (!uploadIncomplete) pendingUploadId.current = null
          setRunStage(`${modelTestStatusLabels[lastKnown.status]} · 本页状态同步中断`)
          setRunError(uploadIncomplete
            ? `${message}；图片上传或提交状态待确认，可取消本次上传后重试`
            : `${message}；测试已提交到后台并会继续运行，请在全局历史中刷新查看`)
        } else {
          setRunError(message)
          setRunStage(modelTestStatusLabels.failed)
        }
        await refreshHistory(signal, true)
      }
    } finally {
      if (!signal.aborted) {
        setRunning(false)
        void refreshAssets(signal)
      }
      if (runController.current === abort) runController.current = null
    }
  }

  async function cancelPendingUpload() {
    const id = pendingUploadId.current
    if (!id) return
    runController.current?.abort()
    setRunning(true); setRunError(''); setRunStage('正在取消未完成的上传…')
    try {
      const cancelled = await api.abandonModelTest(id)
      pendingUploadId.current = null
      setTest(cancelled); setRunStage(modelTestStatusLabels[cancelled.status])
      await refreshHistory(undefined, true)
      void refreshAssets()
    } catch (reason) {
      const requestError = reason as Error & { status?: number }
      if (requestError.status === 409) {
        pendingUploadId.current = null
        const statusController = new AbortController()
        try {
          const current = await api.modelTest(id, statusController.signal)
          setTest(current); setRunStage(modelTestStatusLabels[current.status])
          setRunError('测试已经提交到后台，可在全局历史中继续查看')
          await refreshHistory(undefined, true)
        } catch (statusReason) {
          setRunError(statusReason instanceof Error ? statusReason.message : '读取测试状态失败，请刷新历史记录确认')
        }
      } else if (requestError.status === 404) {
        pendingUploadId.current = null
        setTest(null); setRunStage('')
        setRunError('服务器暂未找到本次测试记录，已解除本页占用；请刷新历史记录确认后再重试')
        await refreshHistory(undefined, true)
      } else {
        setRunError(requestError.message || '取消上传失败，请到历史记录确认状态')
      }
    } finally {
      setRunning(false)
    }
  }

  async function loadMoreHistory() {
    if (!historyCursor || historyLoading || historyLoadingMore) return
    historyLoadMoreController.current?.abort()
    const abort = new AbortController()
    historyLoadMoreController.current = abort
    const generation = historyLoadingVersion.current
    setHistoryLoadingMore(true)
    try {
      const page = await api.globalModelTests(abort.signal, historyCursor, 50, historySource, historyStatus)
      if (abort.signal.aborted || generation !== historyLoadingVersion.current) return
      setHistory((current) => {
        const ids = new Set(current.map((item) => item.id))
        return [...current, ...page.items.filter((item) => !ids.has(item.id))]
      })
      setHistoryCursor(page.next_cursor)
    } catch (reason) {
      if (!abort.signal.aborted && generation === historyLoadingVersion.current) setHistoryError(reason instanceof Error ? reason.message : '加载更多历史记录失败')
    } finally {
      if (historyLoadMoreController.current === abort) {
        historyLoadMoreController.current = null
        setHistoryLoadingMore(false)
      }
    }
  }

  async function deleteHistory(item: ModelTestSummary) {
    const active = activeModelTestStatuses.has(item.status)
    if (!window.confirm(active
      ? '这条测试仍在处理中。删除将取消测试并清理已上传的图片，确定继续吗？'
      : '删除后原图、测试结果和下载文件都无法恢复，确定删除吗？')) return
    historyRefreshVersion.current += 1
    historyLoadingVersion.current += 1
    historyLoadMoreController.current?.abort()
    setHistoryLoadingMore(false)
    setDeletingId(item.id); setHistoryError('')
    try {
      await api.deleteModelTest(item.id)
      historyRefreshVersion.current += 1
      historyLoadingVersion.current += 1
      setHistory((current) => current.filter((record) => record.id !== item.id))
      setHistoryDetails((current) => {
        const next = { ...current }; delete next[item.id]; return next
      })
      setHistorySelectedId((current) => current === item.id ? null : current)
      setHistoryImage(0)
    } catch (reason) {
      setHistoryError(reason instanceof Error ? reason.message : '删除测试记录失败')
    } finally {
      setDeletingId(null)
    }
  }

  async function reuseHistory(item: ModelTest) {
    if (!item.files.length || reusingId) return
    if (!testHasExactModel(item)) {
      setHistoryError('该历史记录缺少可验证的原模型版本，仍可查看和下载已有结果')
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
      const nextSource = modelSourceType(item)
      setSourceType(nextSource)
      // Reruns use the immutable history version, including removed registrations.
      // Do not reinsert a removed model into the library just to display a rerun.
      setSelectedAssetId(assets.some((asset) => asset.id === item.model_asset_id) ? item.model_asset_id || null : null)
      if (nextSource === 'remote_path' && item.execution_node_id) setRemoteNodeId(item.execution_node_id)
      if (nextSource === 'training_artifact') {
        setTrainingTaskId(item.task_id || '')
        setTrainingRunId(item.run_id || item.source_run_id || '')
        setArtifactId(item.artifact_id ?? null)
      }
      setImages(prepared); setConfidence(item.confidence); setIou(item.iou); setSelectedImage(0)
      setTest(null); setRunStage(''); setRunError(''); setRerunSource(item); setView('new'); setEnlarged(null)
    } catch (reason) {
      releaseImages(prepared)
      if (!abort.signal.aborted) setHistoryError(reason instanceof Error ? reason.message : '准备重测图片失败')
    } finally {
      if (!abort.signal.aborted) setReusingId(null)
      if (reuseController.current === abort) reuseController.current = null
    }
  }

  function openLightbox(url: string, title: string, opener: HTMLElement) {
    lightboxOpener.current = opener
    setEnlarged({ url, title })
  }

  const modelReady = rerunSource
    ? testHasExactModel(rerunSource)
    : sourceType === 'training_artifact' ? Boolean(selectedArtifact) : selectedAsset?.status === 'ready'
  const completed = test?.status === 'completed'
  const failed = runStage === modelTestStatusLabels.failed
  const currentTaskType = modelTestTaskType(test || rerunSource) || (sourceType === 'training_artifact' ? selectedTask?.task_type : null)
  const classification = currentTaskType === 'classify'
  const parametersChanged = completed && (confidence !== test.confidence || (!classification && iou !== test.iou))
  const hasPendingUpload = Boolean(pendingUploadId.current)
  const currentResult = completed ? `/api/model-tests/${test.id}/images/${selectedImage}/result` : ''
  const displayedSource = test || rerunSource
  const displayedSourceType = displayedSource ? modelSourceType(displayedSource) : sourceType
  const displayedModelName = displayedSource
    ? displayedSource.model_source_name || displayedSource.model_name || modelTestWeightName(displayedSource)
    : sourceType === 'training_artifact'
      ? selectedArtifact?.name.split('/').pop() || '尚未选择权重'
      : selectedAsset?.name || '尚未选择模型'
  const localRoot = localRoots[0] || ''
  const removalDetails = removalAsset ? assets.find((asset) => asset.id === removalAsset.id) || removalAsset : null

  function renderSelectedModel() {
    if (sourceType === 'training_artifact') {
      return <div className={`model-source-selection${selectedArtifact ? ' ready' : ''}`}>
        <Box size={20} />
        <div><span>当前模型</span><strong>{selectedArtifact?.name.split('/').pop() || '请选择 .pt 训练产物'}</strong><small>{selectedTask?.name || '未选择任务'}{selectedRun ? ` · Attempt ${String(selectedRun.attempt_number).padStart(3, '0')}` : ''}{selectedArtifact ? ` · ${formatBytes(selectedArtifact.size)}` : ''}</small></div>
        {selectedArtifact && <CheckCircle2 size={18} />}
      </div>
    }
    return <div className={`model-source-selection ${selectedAsset?.status || ''}`}>
      {sourceType === 'remote_path' ? <Server size={20} /> : <HardDrive size={20} />}
      <div><span>当前模型</span><strong>{selectedAsset?.name || '请校验或选择已登记模型'}</strong><small title={selectedAsset?.path}>{selectedAsset ? `${assetStatusLabels[selectedAsset.status]} · ${selectedAsset.path}` : sourceType === 'remote_path' ? '路径属于所选远程服务器' : '路径属于中心服务器'}</small></div>
      {selectedAsset?.status === 'ready' ? <CheckCircle2 size={18} /> : selectedAsset?.status === 'preparing' ? <LoaderCircle className="spin" size={18} /> : selectedAsset?.status === 'failed' ? <AlertCircle size={18} /> : null}
    </div>
  }

  return <div className={`page model-tests-page${dragging ? ' is-dragging' : ''}`} onDragEnter={dragEnter} onDragOver={dragOver} onDragLeave={dragLeave} onDrop={dropImages}>
    {dragging && <div className="model-tests-drop-indicator"><Upload size={42} /><strong>松开添加图片</strong><span>图片将加入当前测试</span></div>}
    <header className="page-header model-tests-header">
      <div><span className="eyebrow">MODEL TEST CENTER</span><h1>模型测试</h1><p>选择本机路径、训练产物或远程 Agent 模型，上传图片完成独立测试。</p></div>
      <div className="model-tests-view-tabs" role="tablist" aria-label="模型测试视图">
        <button type="button" role="tab" aria-selected={view === 'new'} className={view === 'new' ? 'active' : ''} onClick={() => setView('new')}><Sparkles size={16} />新建测试</button>
        <button type="button" role="tab" aria-selected={view === 'history'} className={view === 'history' ? 'active' : ''} onClick={() => setView('history')}><History size={16} />全局历史<span title={`当前筛选已加载 ${history.length} 条${historyCursor ? '，可继续加载更多' : ''}`}>已加载 {history.length}</span></button>
      </div>
    </header>

    {view === 'new' ? <div className="model-tests-new-layout">
      <aside className="model-source-panel">
        <div className="model-tests-section-heading"><div><span>01</span><h2>选择模型来源</h2></div><p>路径始终属于当前选择的服务器。</p></div>
        <div className="model-source-tabs" role="tablist" aria-label="模型来源">
          <button type="button" role="tab" aria-selected={sourceType === 'local_path'} disabled={running} className={sourceType === 'local_path' ? 'active' : ''} onClick={() => chooseSource('local_path')}><HardDrive size={16} /><span>本机路径<small>中心服务器</small></span></button>
          <button type="button" role="tab" aria-selected={sourceType === 'training_artifact'} disabled={running} className={sourceType === 'training_artifact' ? 'active' : ''} onClick={() => chooseSource('training_artifact')}><Box size={16} /><span>训练产物<small>best.pt / last.pt</small></span></button>
          <button type="button" role="tab" aria-selected={sourceType === 'remote_path'} disabled={running} className={sourceType === 'remote_path' ? 'active' : ''} onClick={() => chooseSource('remote_path')}><Server size={16} /><span>远程路径<small>Agent 服务器</small></span></button>
        </div>

        {sourceType === 'local_path' && <section className="model-source-form">
          <label>模型路径<input value={localPath} disabled={running || sourceBusy} placeholder={`例如：${localRoot ? `${localRoot.replace(/\/$/, '')}/` : 'pt/'}my-model.pt`} onChange={(event) => { setLocalPath(event.target.value); setSelectedAssetId(null); setSourceNotice(''); resetCurrentResult() }} /></label>
          <small><FolderOpen size={13} />这是中心服务器路径。{localRoots.length ? <>当前允许目录：<code>{localRoots.join('、')}</code></> : '相对路径由平台配置的 pt 模型目录解析。'}</small>
          <label>显示名称（可选）<input value={localName} disabled={running || sourceBusy} placeholder="例如：yan-v2-best" onChange={(event) => setLocalName(event.target.value)} /></label>
          <button type="button" className="button primary" disabled={running || sourceBusy || !localPath.trim()} onClick={() => void registerPathModel()}>{sourceBusy ? <LoaderCircle className="spin" size={16} /> : <ScanEye size={16} />}校验并使用</button>
        </section>}

        {sourceType === 'remote_path' && <section className="model-source-form">
          <label>远程服务器<select value={remoteNodeId} disabled={running || sourceBusy} onChange={(event) => { setRemoteNodeId(event.target.value); setSelectedAssetId(null); setSourceNotice(''); resetCurrentResult() }}><option value="">请选择 Agent 节点</option>{remoteNodes.map((node) => <option key={node.id} value={node.id}>{node.name} · {node.status === 'online' ? '在线' : '离线'}</option>)}</select></label>
          <label>远程模型路径<input value={remotePath} disabled={running || sourceBusy} placeholder="输入所选服务器上的 .pt 路径" onChange={(event) => { setRemotePath(event.target.value); setSelectedAssetId(null); setSourceNotice(''); resetCurrentResult() }} /></label>
          <small><Server size={13} />这是所选远程服务器上的路径，需位于 Agent 上报的 model_test_roots 目录中。</small>
          <label>显示名称（可选）<input value={remoteName} disabled={running || sourceBusy} placeholder="例如：远程 yan-best" onChange={(event) => setRemoteName(event.target.value)} /></label>
          <button type="button" className="button primary" disabled={running || sourceBusy || !remoteNodeId || !remotePath.trim()} onClick={() => void registerPathModel()}>{sourceBusy ? <LoaderCircle className="spin" size={16} /> : <Server size={16} />}校验并准备</button>
        </section>}

        {sourceType === 'training_artifact' && <section className="model-source-form training-artifact-form">
          <label>模型库 · 训练任务</label>
          <TrainingTaskLibrary
            tasks={tasks}
            groups={taskGroups}
            nodes={nodes}
            selectedTaskId={trainingTaskId}
            disabled={running}
            onSelect={(task) => {
              if (task.id === trainingTaskId) return
              setTrainingTaskId(task.id)
              setTrainingRunId('')
              setArtifactId(null)
              resetCurrentResult()
            }}
          />
          <label>训练轮次<select value={trainingRunId} disabled={running || !trainingTaskId || trainingSourceLoading} onChange={(event) => { setTrainingRunId(event.target.value); setArtifactId(null); resetCurrentResult() }}><option value="">{trainingSourceLoading ? '正在读取…' : '请选择 Attempt'}</option>{runs.map((run) => <option key={run.id} value={run.id}>Attempt {String(run.attempt_number).padStart(3, '0')} · {statusLabel(run.status)} · {run.completed_epochs} Epoch</option>)}</select></label>
          <label>模型权重<select value={artifactId ?? ''} disabled={running || !trainingRunId || trainingSourceLoading} onChange={(event) => { setArtifactId(event.target.value ? Number(event.target.value) : null); resetCurrentResult() }}><option value="">{trainingSourceLoading ? '正在读取…' : artifacts.length ? '请选择权重' : '当前 Attempt 没有 .pt 产物'}</option>{artifacts.map((artifact) => <option key={artifact.id} value={artifact.id}>{artifact.name.split(/[\\/]/).pop()} · {formatBytes(artifact.size)}{artifact.kind ? ` · ${artifact.kind}` : ''}</option>)}</select></label>
          <small><Info size={13} />远程训练产物会在第一次测试前安全传输到中心模型缓存。</small>
        </section>}

        {sourceError && <div className="model-test-error" role="alert">{sourceError}</div>}
        {sourceNotice && <div className="model-source-notice" role="status"><CheckCircle2 size={15} />{sourceNotice}</div>}
        {renderSelectedModel()}

        {sourceType !== 'training_artifact' && <>
          <section className="registered-models">
            <div><h3>模型库 · 已登记模型</h3></div>
            <RegisteredModelLibrary
              assets={assets}
              sourceType={sourceType}
              nodes={nodes}
              selectedAssetId={selectedAssetId}
              disabled={running || sourceBusy}
              loading={assetsLoading}
              removingAssetId={removingAssetId}
              onRefresh={() => void refreshAssets()}
              onRemove={(asset) => { setRemovalError(''); setRemovalAsset(asset) }}
              onSelect={(asset) => {
                setSelectedAssetId(asset.id)
                if (asset.source_type === 'remote_path' && asset.execution_node_id) setRemoteNodeId(asset.execution_node_id)
                setSourceError(''); setSourceNotice(''); resetCurrentResult()
              }}
            />
          </section>
        </>}
      </aside>

      <main className="model-test-run-panel">
        <div className="model-tests-section-heading"><div><span>02</span><h2>上传图片并测试</h2></div><p>本机和远程模型均由中心服务器 CPU 执行测试；远程模型首次使用需传输权重。关闭页面不会删除已提交任务。</p></div>
        <section className={`model-test-run-source${modelReady ? ' ready' : ''}`}>
          {displayedSourceType === 'remote_path' ? <Server size={19} /> : displayedSourceType === 'training_artifact' ? <Box size={19} /> : <HardDrive size={19} />}
          <div><span>{sourceLabels[displayedSourceType]} · {modelTestTaskLabel(currentTaskType)}</span><strong>{displayedModelName}</strong><small>{displayedSource ? `当前结果固定于 SHA-256 ${displayedSource.model_sha256?.slice(0, 12) || '待记录'}…` : modelReady ? '模型来源已确认，可以提交测试' : '请先在左侧完成模型选择与校验'}</small></div>
          {modelReady ? <CheckCircle2 size={19} /> : <AlertCircle size={19} />}
        </section>
        {rerunSource && !test && <div className="model-test-rerun-notice"><RotateCcw size={15} /><span>已载入历史图片和参数；将固定使用原测试的模型版本，并创建一条新记录。</span><button type="button" className="icon-button" title="取消重测" aria-label="取消重测" disabled={running} onClick={() => { setRerunSource(null); resetCurrentResult() }}><X size={14} /></button></div>}
        <div className="model-test-run-controls">
          <button type="button" className="button ghost" disabled={running} onClick={() => input.current?.click()}><ImagePlus size={16} />{files.length ? `添加图片（${files.length}/10）` : '选择图片'}</button>
          <input ref={input} hidden disabled={running} type="file" accept=".jpg,.jpeg,.png,.webp,.bmp" multiple onChange={(event) => chooseImages(event.target.files)} />
          <div className="model-test-slider"><label htmlFor="standalone-confidence">置信度 Conf <strong>{confidence.toFixed(2)}</strong></label><input id="standalone-confidence" type="range" min="0.01" max="1" step="0.01" value={confidence} disabled={running} onChange={(event) => setConfidence(Number(event.target.value))} /></div>
          <div className="model-test-slider" title={classification ? '分类模型没有检测框，不使用 IoU' : undefined}><label htmlFor="standalone-iou">NMS IoU <strong>{classification ? '不适用' : iou.toFixed(2)}</strong></label><input id="standalone-iou" type="range" min="0.01" max="1" step="0.01" value={iou} disabled={running || classification} aria-describedby="standalone-parameters-help" onChange={(event) => setIou(Number(event.target.value))} /></div>
          <button type="button" className="button primary" disabled={running || sourceBusy || Boolean(removingAssetId) || hasPendingUpload || !modelReady || !files.length} onClick={() => void runModelTest()}>{running ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}{completed ? '再次测试' : '运行测试'}</button>
        </div>
        <p id="standalone-parameters-help" className="model-test-help">最多 10 张 · 单张 20 MB / 2000 万像素 · 总大小 100 MB · 可拖入 JPG、PNG、WEBP、BMP。{classification ? '分类结果按置信度筛选 Top-5；分类模型没有检测框，IoU 不适用。' : '任务类型在加载模型后自动识别；置信度筛选预测结果，IoU 仅对检测、分割、姿态和 OBB 框类任务生效。'}</p>
        {runError && <div className="model-test-error" role="alert">{runError}</div>}
        {(runStage || running) && <div className={`model-test-status${failed ? ' failed' : ''}`} role="status">{running ? <LoaderCircle className="spin" size={14} /> : failed ? <AlertCircle size={14} /> : completed ? <CheckCircle2 size={14} /> : null}{runStage}{completed && ` · ${modelTestParameterText(test)}`}</div>}
        {parametersChanged && <div className="model-test-parameter-notice"><Info size={15} />当前显示的是 {modelTestParameterText(test)} 的历史结果；再次测试会用新阈值创建记录。</div>}

        {!images.length ? <button type="button" className="model-test-empty standalone" onClick={() => input.current?.click()}><Upload size={34} /><strong>点击或拖入测试图片</strong><span>选好模型后，这里会显示原图和测试结果</span></button>
          : <TestWorkspace test={test} images={images} selected={selectedImage} onSelect={setSelectedImage} onRemove={removeImage} busy={running} failed={failed} openLightbox={openLightbox} />}

        <footer className="model-test-run-footer">
          <span>{completed ? `${test.files[selectedImage]?.name || '当前图片'} · ${modelTestResultLabel(modelTestTaskType(test), test.files[selectedImage]?.prediction_count ?? test.files[selectedImage]?.detections ?? null)} · 已保存到全局历史` : hasPendingUpload ? '图片尚未全部提交，可取消本次上传' : '每次测试都会保存模型摘要、参数、原图和结果图'}</span>
          {hasPendingUpload && !running && <button type="button" className="button danger" onClick={() => void cancelPendingUpload()}><X size={15} />取消上传</button>}
          {completed && <a className="button ghost" href={`/api/model-tests/${test.id}/download`} download><FileArchive size={15} />下载全部</a>}
          {currentResult && <a className="button primary" href={`${currentResult}?download=true`} download><Download size={15} />当前结果图</a>}
        </footer>
      </main>
    </div> : <section className="model-tests-history-card">
      <header className="model-tests-history-toolbar">
        <div><SlidersHorizontal size={16} /><strong>筛选历史</strong></div>
        <label>模型来源<select value={historySource} onChange={(event) => setHistorySource(event.target.value as SourceType | '')}><option value="">全部来源</option><option value="local_path">本机路径</option><option value="training_artifact">训练产物</option><option value="remote_path">远程路径</option></select></label>
        <label>测试状态<select value={historyStatus} onChange={(event) => setHistoryStatus(event.target.value as ModelTest['status'] | '')}><option value="">全部状态</option><option value="uploading">准备图片</option><option value="preparing">准备模型</option><option value="queued">等待测试</option><option value="running">正在测试</option><option value="cancel_requested">正在取消</option><option value="completed">测试完成</option><option value="failed">测试失败</option><option value="cancelled">已取消</option></select></label>
        <button type="button" className="button ghost compact" disabled={historyLoading} onClick={() => void refreshHistory()}>{historyLoading ? <LoaderCircle className="spin" size={15} /> : <RefreshCw size={15} />}刷新</button>
      </header>
      {historyError && <div className="model-test-error model-tests-history-error" role="alert">{historyError}</div>}
      <div className="model-tests-history-layout">
        <aside className="model-test-history-sidebar">
          {historyLoading && !history.length ? <div className="model-test-history-empty"><LoaderCircle className="spin" size={25} /><span>正在加载全局测试历史…</span></div>
            : history.length ? <div className="model-test-history-list">{groupedHistory.map((section) => {
              const collapsed = collapsedHistoryGroups.has(section.key)
              return <section className={`model-test-history-group${collapsed ? ' collapsed' : ''}`} key={section.key}>
                <header><button type="button" aria-expanded={!collapsed} onClick={() => setCollapsedHistoryGroups((current) => {
                  const next = new Set(current)
                  if (next.has(section.key)) next.delete(section.key)
                  else next.add(section.key)
                  return next
                })}><ChevronDown size={14} /><FolderOpen size={15} /><strong>{section.name}</strong><span>{section.items.length}</span></button></header>
                {!collapsed && <div className="model-test-history-group-items">{section.items.map((item) => {
                  const objects = item.files.reduce((sum, file) => sum + (file.prediction_count ?? file.detections ?? 0), 0)
                  const detected = item.files.filter((file) => (file.prediction_count ?? file.detections ?? 0) > 0).length
                  return <button type="button" key={item.id} className={`model-test-history-item${selectedSummary?.id === item.id ? ' selected' : ''}`} onClick={() => { setHistorySelectedId(item.id); setHistoryImage(0); setHistoryError(''); if (window.matchMedia('(max-width: 680px)').matches) window.requestAnimationFrame(() => historyDetailRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })) }}>
                    <HistoryImagePreview test={item} />
                    <div><span className={`model-test-history-status ${item.status}`}><i />{modelTestStatusLabels[item.status]}</span><time>{modelTestDate(item.created_at)}</time></div>
                    <strong>{modelTestWeightName(item)}<small>{sourceDescription(item)}</small></strong>
                    <p>{modelTestTaskLabel(modelTestTaskType(item))} · {item.files.length} 张图片 · {detected} 张有结果 · {modelTestResultLabel(modelTestTaskType(item), item.status === 'completed' ? objects : null)}</p>
                    <footer><span>{modelTestParameterText(item)}</span>{typeof item.storage_bytes === 'number' && <span><HardDrive size={11} />{formatBytes(item.storage_bytes)}</span>}</footer>
                  </button>
                })}</div>}
              </section>
            })}{historyCursor && <button type="button" className="button ghost model-test-history-more" disabled={historyLoading || historyLoadingMore} onClick={() => void loadMoreHistory()}>{historyLoadingMore ? <LoaderCircle className="spin" size={14} /> : <History size={14} />}加载更多</button>}</div>
              : <div className="model-test-history-empty"><History size={30} /><strong>暂无匹配的测试记录</strong><span>新建测试或调整上方筛选条件</span></div>}
        </aside>
        <section ref={historyDetailRef} className="model-test-history-detail" aria-label="测试记录详情">
          {selectedHistory ? <>
            <div className="model-test-history-heading"><div><span>{modelTestDate(selectedHistory.created_at)}</span><h3>{modelTestWeightName(selectedHistory)}</h3><p>{sourceDescription(selectedHistory)} · {modelTestParameterText(selectedHistory)}</p></div>{historyDetailLoading && <LoaderCircle className="spin" size={18} />}</div>
            <div className="model-test-history-source-strip">
              <div><Box size={17} /><span>模型<strong>{selectedHistory.model_source_name || modelTestWeightName(selectedHistory)}</strong></span></div>
              <div><Server size={17} /><span>来源<strong>{sourceDescription(selectedHistory)}</strong></span></div>
              <div><HardDrive size={17} /><span>模型版本<strong>{selectedHistory.model_sha256 ? `${selectedHistory.model_sha256.slice(0, 12)}…` : '旧记录未保存摘要'}{typeof selectedHistory.model_size === 'number' ? ` · ${formatBytes(selectedHistory.model_size)}` : ''}</strong></span></div>
              <div><Cpu size={17} /><span>执行环境<strong>{modelTestExecutionLabel(selectedHistory)}</strong></span></div>
            </div>
            {selectedHistory.error && <div className="model-test-error" role="alert">{selectedHistory.error}</div>}
            {selectedHistory.status === 'completed' && !testHasExactModel(selectedHistory) && <div className="model-test-parameter-notice"><Info size={15} />这是旧版测试记录，没有完整模型摘要；现有结果仍可查看和下载，但不能按原模型重测。</div>}
            <TestWorkspace test={selectedHistory} selected={historyImage} onSelect={setHistoryImage} busy={activeModelTestStatuses.has(selectedHistory.status)} failed={selectedHistory.status === 'failed' || selectedHistory.status === 'cancelled'} openLightbox={openLightbox} />
            <footer className="model-tests-history-actions">
              <span>{modelTestStatusLabels[selectedHistory.status]}{selectedHistory.finished_at ? ` · ${modelTestDate(selectedHistory.finished_at)}` : ''}</span>
              {selectedHistory.status === 'completed' && <a className="button ghost" href={`/api/model-tests/${selectedHistory.id}/download`} download><FileArchive size={15} />下载全部</a>}
              {selectedHistory.status === 'completed' && <a className="button primary" href={`/api/model-tests/${selectedHistory.id}/images/${historyImage}/result?download=true`} download><Download size={15} />当前结果图</a>}
              {selectedHistory.status === 'completed' && <button type="button" className="button primary" disabled={reusingId === selectedHistory.id || !testHasExactModel(selectedHistory)} title={testHasExactModel(selectedHistory) ? '使用原图和原模型版本创建新记录' : '历史记录缺少可验证的原模型版本'} onClick={() => void reuseHistory(selectedHistory)}>{reusingId === selectedHistory.id ? <LoaderCircle className="spin" size={15} /> : <RotateCcw size={15} />}使用这些图片重测</button>}
              <button type="button" className="button danger" disabled={deletingId === selectedHistory.id} onClick={() => void deleteHistory(selectedHistory)}>{deletingId === selectedHistory.id ? <LoaderCircle className="spin" size={15} /> : <Trash2 size={15} />}删除记录</button>
            </footer>
          </> : selectedSummary && historyDetailLoading ? <div className="model-test-history-detail-empty"><LoaderCircle className="spin" size={26} /><span>正在加载测试详情…</span></div>
            : <div className="model-test-history-detail-empty"><ScanEye size={36} /><span>选择左侧记录查看原图、测试结果和统计信息</span></div>}
        </section>
      </div>
    </section>}

    {removalDetails && <dialog ref={removalDialog} className="model-asset-remove-dialog" aria-labelledby="model-asset-remove-title" aria-describedby="model-asset-remove-description" onCancel={(event) => { event.preventDefault(); if (!removingAssetId) setRemovalAsset(null) }}>
      <header><div><Trash2 size={20} /><h2 id="model-asset-remove-title">移除模型登记</h2></div><button type="button" className="icon-button" aria-label="关闭移除确认" disabled={Boolean(removingAssetId)} onClick={() => setRemovalAsset(null)}><X size={19} /></button></header>
      <div className="model-asset-remove-body">
        <p id="model-asset-remove-description">该模型将从模型库中移除。原始 .pt 权重、训练任务和已保存的测试历史都会保留。</p>
        <dl>
          <div><dt>模型名称</dt><dd>{removalDetails.name}</dd></div>
          <div><dt>所属服务器</dt><dd>{removalDetails.source_type === 'local_path' ? '本机 · 中心服务器' : `远程 · ${nodesById.get(removalDetails.execution_node_id || '')?.name || removalDetails.execution_node_name || '已登记节点'}`}</dd></div>
          <div><dt>源文件路径</dt><dd className="model-asset-remove-path">{removalDetails.path}</dd></div>
          <div><dt>关联测试历史</dt><dd>{typeof removalDetails.history_count === 'number' ? `${removalDetails.history_count} 条（保留）` : '已有历史均保留'}</dd></div>
        </dl>
        <p className="model-asset-remove-hint">以后可重新登记同一路径；已有历史仍可查看和下载。原模型版本仍可用时，也可以从历史重测。</p>
        {removalDetails.removal_blocked_reason && <div className="model-test-parameter-notice" role="status"><Info size={15} />{removalDetails.removal_blocked_reason}</div>}
        {removalError && <div className="model-test-error" role="alert">{removalError}</div>}
      </div>
      <footer><button type="button" className="button ghost" autoFocus disabled={Boolean(removingAssetId)} onClick={() => setRemovalAsset(null)}>取消</button><button type="button" className="button danger" disabled={Boolean(removingAssetId || removalDetails.removal_blocked_reason)} onClick={() => void removeRegisteredModel()}>{removingAssetId ? <LoaderCircle className="spin" size={15} /> : <Trash2 size={15} />}{removingAssetId ? '正在移除…' : '确认移除登记'}</button></footer>
    </dialog>}
    {enlarged && <div ref={lightbox} className="model-test-lightbox" role="dialog" aria-modal="true" aria-label={`放大查看${enlarged.title}`} tabIndex={-1} onClick={() => setEnlarged(null)}><button type="button" className="icon-button" aria-label="关闭图片放大" onClick={() => setEnlarged(null)}><X size={24} /></button><img src={enlarged.url} alt={enlarged.title} onClick={(event) => event.stopPropagation()} /><span>{enlarged.title} · Esc 关闭</span></div>}
  </div>
}
