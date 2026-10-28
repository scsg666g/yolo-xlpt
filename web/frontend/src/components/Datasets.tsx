import { DatasetDiscoveryModal } from './DatasetDiscoveryModal'
import { DatasetStorageModal, DatasetActionModal } from './DatasetStorage'
import { DatasetOperationDrawer } from './DatasetOperationDrawer'
import { DatasetServerSelect } from './DatasetServerSelect'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  History,
  Settings2,
  CheckCircle2,
  Database,
  FolderOpen,
  HardDrive,
  Images,
  Link2,
  RefreshCw,
  Search,
  Tags,
  Trash2,
  Upload,
  X,
} from 'lucide-react'
import { api, formatBytes, statusLabel } from '../api'
import type { ComputeNode, DatasetOperationSummary, DatasetUploadResult, ManagedDataset } from '../types'
import { UploadDatasetModal } from './UploadDatasetModal'

type DatasetFilter = 'all' | 'deletable' | 'referenced'

function valueOrZero(value: number | undefined) {
  return value ?? 0
}

function formatCreatedAt(value: string | null) {
  if (!value) return '时间未记录'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '时间未记录' : date.toLocaleString('zh-CN')
}

function refreshRequestId() {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID().replaceAll('-', '')
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), value => value.toString(16).padStart(2, '0')).join('')
}

function pendingRefreshStorageKey(datasetId: string) {
  return `dataset-refresh-request:${encodeURIComponent(datasetId)}`
}

function readPendingRefreshRequest(datasetId: string) {
  try { return window.sessionStorage.getItem(pendingRefreshStorageKey(datasetId)) || null } catch { return null }
}

function savePendingRefreshRequest(datasetId: string, requestId: string | null) {
  try {
    const key = pendingRefreshStorageKey(datasetId)
    if (requestId) window.sessionStorage.setItem(key, requestId)
    else window.sessionStorage.removeItem(key)
  } catch { /* Retry safety remains within this page when browser storage is unavailable. */ }
}

export function Datasets() {
  const [datasets, setDatasets] = useState<ManagedDataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [loadError, setLoadError] = useState('')
  const [notice, setNotice] = useState('')
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<DatasetFilter>('all')
  const [uploading, setUploading] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<ManagedDataset | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [refreshing, setRefreshing] = useState<string | null>(null)
  const refreshSubmitting = useRef(false)
  // Keep uncertain submissions identifiable; retrying must not create a second job.
  const pendingRefreshRequests = useRef(new Map<string, string>())
  const [actionTarget, setActionTarget] = useState<ManagedDataset | 'register' | null>(null)
  const [serverFilter, setServerFilter] = useState('all')
  const [discoveryOpen, setDiscoveryOpen] = useState(false)
  const [storageOpen, setStorageOpen] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [operationSummary, setOperationSummary] = useState<DatasetOperationSummary | null>(null)
  const [historyError, setHistoryError] = useState('')
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [summaryRefresh, setSummaryRefresh] = useState(0)
  const previousSignature = useRef('')
  const loadRequest = useRef<{ version: number; controller: AbortController } | null>(null)
  const loadVersion = useRef(0)
  const catalogueLoaded = useRef(false)
  const deleteDialog = useRef<HTMLDivElement>(null)
  const deleteCancelButton = useRef<HTMLButtonElement>(null)
  const deleteOpener = useRef<HTMLButtonElement | null>(null)

  const loadDatasets = useCallback(async () => {
    loadRequest.current?.controller.abort()
    const version = ++loadVersion.current
    const controller = new AbortController()
    loadRequest.current = { version, controller }
    setLoading(true)
    setLoadError('')
    try {
      const incoming = await api.datasetEntries(controller.signal)
      if (version !== loadVersion.current) return false
      setDatasets(incoming)
      catalogueLoaded.current = true
      return true
    } catch (reason) {
      if (controller.signal.aborted || version !== loadVersion.current) return false
      setLoadError(reason instanceof Error ? reason.message : '数据集列表加载失败')
      return false
    } finally {
      if (loadRequest.current?.version === version) loadRequest.current = null
      if (version === loadVersion.current) setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadDatasets()
    return () => {
      loadVersion.current += 1
      loadRequest.current?.controller.abort()
      loadRequest.current = null
    }
  }, [loadDatasets])

  useEffect(() => {
    let active = true; let running = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let controller: AbortController | null = null
    let forceReload = false
    async function load() {
      if (!active || running || document.hidden) return
      running = true; controller = new AbortController()
      const result = await Promise.allSettled([api.datasetOperationSummary(controller.signal), api.computeNodes(controller.signal)])
      if (active && !controller.signal.aborted) {
        const [history, compute] = result
        if (history.status === 'fulfilled') {
          setOperationSummary(history.value); setHistoryError('')
          const signature = history.value.revision
          if (forceReload || !catalogueLoaded.current && !loadRequest.current || previousSignature.current && signature !== previousSignature.current) {
            // Advance the acknowledged revision only after the matching list loaded.
            // Otherwise one failed request leaves references stale indefinitely.
            if (await loadDatasets() && active) { previousSignature.current = signature; forceReload = false }
          } else previousSignature.current = signature
        } else {
          setHistoryError(history.reason instanceof Error ? history.reason.message : '操作统计暂不可用')
          if ((forceReload || !catalogueLoaded.current && !loadRequest.current) && await loadDatasets()) forceReload = false
        }
        if (active && compute.status === 'fulfilled') setNodes(compute.value)
      }
      running = false
      if (active) timer = setTimeout(load, 5000)
    }
    function visible() { if (!document.hidden) { forceReload = true; clearTimeout(timer); void load() } }
    void load(); document.addEventListener('visibilitychange', visible); window.addEventListener('focus', visible)
    return () => { active = false; controller?.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible); window.removeEventListener('focus', visible) }
  }, [loadDatasets, summaryRefresh])

  const servers = useMemo(() => {
    const options = new Map<string, { id: string; name: string; status?: string; count?: number }>()
    options.set('local', { id: 'local', name: '本机（中台）' })
    for (const n of operationSummary?.nodes || []) options.set(n.id, { id: n.id, name: n.name })
    for (const d of datasets) if (!options.has(d.node_id)) options.set(d.node_id, { id: d.node_id, name: d.node_name || d.node_id })
    for (const n of nodes) options.set(n.id, { id: n.id, name: n.kind === 'local' ? '本机（中台）' : n.name, status: !n.enabled ? '已禁用' : n.status === 'online' ? '在线' : n.status === 'pending' ? '待连接' : '离线' })
    return [{ id: 'all', name: '全部服务器', count: datasets.length }, ...Array.from(options.values()).map(n => ({ ...n, count: datasets.filter(d => d.node_id === n.id).length }))]
  }, [nodes, datasets, operationSummary])
  function operationsChanged() { setSummaryRefresh(v => v + 1); void loadDatasets() }

  async function refreshDataset(dataset: ManagedDataset) {
    if (refreshSubmitting.current || dataset.blocking_operation_id) return
    refreshSubmitting.current = true
    setRefreshing(dataset.id); setError(''); setNotice('')
    const requestId = pendingRefreshRequests.current.get(dataset.id) || readPendingRefreshRequest(dataset.id) || refreshRequestId()
    pendingRefreshRequests.current.set(dataset.id, requestId)
    savePendingRefreshRequest(dataset.id, requestId)
    try {
      let operation: Awaited<ReturnType<typeof api.refreshDatasetEntry>> | null = null
      try { operation = await api.datasetOperation(requestId) } catch (reason) {
        if ((reason as { status?: number } | null)?.status !== 404) throw reason
      }
      if (operation && (operation.kind !== 'refresh' || operation.node_id !== dataset.node_id)) {
        throw new Error('重新加载请求标识与服务器已有操作不匹配，请查看操作记录后再试')
      }
      operation ||= await api.refreshDatasetEntry(dataset.id, requestId)
      pendingRefreshRequests.current.delete(dataset.id)
      savePendingRefreshRequest(dataset.id, null)
      if (operation.status === 'failed' || operation.status === 'cancelled') {
        setError(operation.error || '此前提交的重新加载已失败或取消，请在操作记录中查看，或重新发起加载。')
      } else setNotice(operation.status === 'completed'
        ? `“${dataset.name}”先前提交的重新加载已完成，已同步最新统计。`
        : `“${dataset.name}”已提交重新加载，校验成功后更新图片、标签、类别、空间和警告。支持并行读取的节点可在训练期间校验；请在操作记录中查看执行或等待原因。`)
      operationsChanged()
    } catch (reason) {
      const status = (reason as { status?: number } | null)?.status
      if (status !== undefined && status < 500 && status !== 408) {
        pendingRefreshRequests.current.delete(dataset.id)
        savePendingRefreshRequest(dataset.id, null)
      }
      setError(reason instanceof Error ? reason.message : '重新加载提交失败')
      void loadDatasets()
    } finally {
      refreshSubmitting.current = false
      setRefreshing(null)
    }
  }

  useEffect(() => {
    if (!deleteTarget) return
    deleteCancelButton.current?.focus()
    return () => {
      const opener = deleteOpener.current
      deleteOpener.current = null
      if (opener?.isConnected) opener.focus()
    }
  }, [deleteTarget])

  useEffect(() => {
    if (deleteTarget && deleting) deleteDialog.current?.focus()
  }, [deleteTarget, deleting])

  const summary = useMemo(() => ({
    count: datasets.length,
    images: datasets.reduce((sum, item) => sum + valueOrZero(item.stats.train_images) + valueOrZero(item.stats.val_images), 0),
    bytes: datasets.reduce((sum, item) => sum + item.total_bytes, 0),
    referenced: datasets.filter((item) => item.references.length > 0).length,
  }), [datasets])

  const visibleDatasets = useMemo(() => {
    const query = search.trim().toLocaleLowerCase()
    return datasets.filter((dataset) => {
      if (serverFilter !== 'all' && dataset.node_id !== serverFilter) return false
      if (filter === 'deletable' && !dataset.can_delete) return false
      if (filter === 'referenced' && !dataset.references.length) return false
      if (!query) return true
      return [dataset.name, dataset.directory, dataset.config_path, ...dataset.classes]
        .some((value) => value.toLocaleLowerCase().includes(query))
    })
  }, [datasets, filter, search, serverFilter])

  async function handleUploaded(result: DatasetUploadResult) {
    setUploading(false)
    setNotice(`数据集“${result.name}”已上传并可用于训练。`)
    await loadDatasets()
  }

  async function deleteDataset() {
    if (!deleteTarget || deleteTarget.references.length) return
    setDeleting(true)
    loadRequest.current?.controller.abort()
    setError('')
    setNotice('')
    const name = deleteTarget.name
    try {
      await api.deleteDatasetEntry(deleteTarget.id)
      operationsChanged()
      setDeleteTarget(null)
      setNotice(`数据集“${name}”已提交永久删除，请在操作历史中查看完成状态。`)
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : '数据集删除失败'
      setDeleteTarget(null)
      await loadDatasets()
      setError(message)
    } finally {
      setDeleting(false)
    }
  }

  function openDeleteDialog(dataset: ManagedDataset, opener: HTMLButtonElement) {
    setError('')
    setNotice('')
    deleteOpener.current = opener
    void api.datasetDeletePreview(dataset.id).then(preview => {
      if (!preview.can_delete) { setError(preview.delete_block_reason || '该数据集仍被引用、未接管或正在处理，请刷新后检查'); return }
      setDeleteTarget(preview)
    }).catch((e: Error) => setError(e.message))
  }

  function closeDeleteDialog() {
    if (!deleting) setDeleteTarget(null)
  }

  function handleDeleteDialogKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.nativeEvent.isComposing) return
    if (event.key === 'Escape') {
      event.preventDefault()
      closeDeleteDialog()
      return
    }
    if (event.key !== 'Tab') return
    const controls = deleteDialog.current?.querySelectorAll<HTMLElement>(
      'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]',
    )
    if (!controls?.length) {
      event.preventDefault()
      deleteDialog.current?.focus()
      return
    }
    const first = controls[0]
    const last = controls[controls.length - 1]
    if (event.shiftKey && (document.activeElement === first || document.activeElement === deleteDialog.current)) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  return <div className="page datasets-page">
    <header className="page-header datasets-header">
      <div>
        <span className="eyebrow">DATASET LIBRARY</span>
        <h1>数据集管理</h1>
        <p>统一管理本机和远程数据集：保存位置、上传、已有目录登记、复制迁移及安全删除。当前托管校验支持 YOLO 目标检测。</p>
      </div>
      <div className="header-actions datasets-header-actions">
        <button type="button" className="button ghost" onClick={() => setStorageOpen(true)}><Settings2 size={16} />保存位置</button>
        <button type="button" className="button ghost" title={historyError || '查看全部数据集操作记录'} onClick={() => setHistoryOpen(true)}><History size={16} />操作记录{operationSummary && operationSummary.active > 0 && <span className="dataset-count-badge">{operationSummary.active} 进行中</span>}{operationSummary && operationSummary.failed > 0 && <span className="dataset-count-badge failed">{operationSummary.failed} 失败</span>}{historyError && <AlertTriangle size={14} />}</button>
        <button type="button" className="button ghost" onClick={() => setDiscoveryOpen(true)}><Search size={16} />扫描已有数据集</button>
        <button type="button" className="button ghost" onClick={() => setActionTarget('register')}><FolderOpen size={16} />登记已有目录</button>
        <button type="button" className="button ghost" disabled={loading || deleting} onClick={() => { setNotice(''); operationsChanged() }}>
          <RefreshCw size={16} className={loading ? 'spin' : ''} />刷新
        </button>
        <button type="button" className="button primary" disabled={deleting} onClick={() => { setError(''); setNotice(''); setUploading(true) }}>
          <Upload size={16} />上传数据集
        </button>
      </div>
    </header>

    {notice && <div className="dataset-feedback success" role="status"><CheckCircle2 size={17} />{notice}</div>}
    {(error || loadError) && <div className="dataset-feedback error" role="alert"><AlertTriangle size={17} /><span>{error || loadError}</span><button type="button" onClick={() => { setError(''); setLoadError('') }} aria-label="关闭错误提示"><X size={15} /></button></div>}

    <section className="dataset-overview" aria-label="数据集概览">
      <article><Database size={19} /><div><strong>{summary.count}</strong><span>平台数据集</span></div></article>
      <article><Images size={19} /><div><strong>{summary.images.toLocaleString('zh-CN')}</strong><span>训练与验证图片</span></div></article>
      <article><HardDrive size={19} /><div><strong>{formatBytes(summary.bytes)}</strong><span>数据集总大小</span></div></article>
      <article><Link2 size={19} /><div><strong>{summary.referenced}</strong><span>已被任务引用</span></div></article>
    </section>

    <section className="dataset-toolbar" aria-label="数据集筛选">
      <label className="dataset-search">
        <Search size={16} />
        <input aria-label="搜索数据集" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索名称、路径或类别" />
        {search && <button type="button" onClick={() => setSearch('')} aria-label="清空搜索"><X size={14} /></button>}
      </label>
      <div className="dataset-filters" role="group" aria-label="按引用状态筛选">
        {([['all', '全部'], ['deletable', '可删除'], ['referenced', '已引用']] as Array<[DatasetFilter, string]>).map(([value, label]) =>
          <button key={value} type="button" className={filter === value ? 'active' : ''} onClick={() => setFilter(value)} aria-pressed={filter === value}>{label}</button>)}
      </div>
      <DatasetServerSelect options={servers} value={serverFilter} onChange={setServerFilter} />
      <span className="dataset-result-count">当前显示 {visibleDatasets.length} / {datasets.length}</span>
    </section>

    {loading && !datasets.length ? <div className="empty dataset-empty"><RefreshCw className="spin" size={24} /><strong>正在加载数据集…</strong></div>
      : visibleDatasets.length ? <section className="dataset-grid" aria-label="数据集列表">
        {visibleDatasets.map((dataset) => {
          const referenced = dataset.references.length > 0
          const classCount = dataset.stats.classes ?? dataset.classes.length
          return <article className="dataset-card" key={dataset.id}>
            <header>
              <div className="dataset-card-icon"><Database size={19} /></div>
              <div className="dataset-card-title"><h2 title={dataset.name}>{dataset.name}</h2><span>{dataset.node_name || (dataset.node_id === 'local' ? '本机' : '远程')} · {dataset.ownership === 'managed' ? '平台管理' : '外部登记'} · {formatCreatedAt(dataset.created_at)}</span></div>
              <div className="dataset-card-actions">
                <button type="button" className="dataset-reload" aria-label={`重新加载数据集：${dataset.name}`} title={dataset.blocking_operation_id ? '请先完成或在操作记录中重试／取消未完成操作' : '重新读取文件并校验统计，成功后更新卡片；不会修改源文件'} disabled={!!refreshing || !!dataset.blocking_operation_id || dataset.status !== 'ready'} onClick={() => void refreshDataset(dataset)}><RefreshCw size={14} className={refreshing === dataset.id || (dataset.blocking_operation_kind === 'refresh' && dataset.refresh_status !== 'failed') ? 'spin' : ''} /><span>重新加载</span></button>
                <span className={`dataset-reference-state ${dataset.blocking_operation_id ? 'busy' : referenced ? 'referenced' : 'free'}`}><i />{dataset.blocking_operation_id ? '操作未完成' : referenced ? `${dataset.references.length} 个引用` : dataset.ownership === 'managed' ? '无引用' : '外部文件'}</span>
              </div>
            </header>

            <div className="dataset-check-status"><small>{dataset.checked_at ? `最近完整校验：${formatCreatedAt(dataset.checked_at)}` : '统计来自登记记录，完整校验时间未记录'}</small>
              {dataset.blocking_operation_kind === 'refresh' && <button type="button" onClick={() => setHistoryOpen(true)}>{dataset.refresh_status === 'failed' ? '重新加载失败，保留上次统计；查看处理' : dataset.refresh_status === 'pending' ? '重新加载已提交，保留上次统计；查看等待原因' : '重新加载校验中，保留上次统计；查看进度'}</button>}
              {dataset.refresh_status === 'cancelled' && <small>上次重新加载已取消，保留上次统计</small>}
              {dataset.refresh_error && dataset.refresh_status === 'failed' && <small role="alert">{dataset.refresh_error}</small>}
            </div>

            <div className="dataset-metrics">
              <div><span>训练图片</span><strong>{valueOrZero(dataset.stats.train_images).toLocaleString('zh-CN')}</strong></div>
              <div><span>验证图片</span><strong>{valueOrZero(dataset.stats.val_images).toLocaleString('zh-CN')}</strong></div>
              <div><span>类别数</span><strong>{classCount}</strong></div>
              <div><span>占用空间</span><strong>{formatBytes(dataset.total_bytes)}</strong></div>
            </div>

            <div className="dataset-path" title={dataset.directory}><FolderOpen size={14} /><code>{dataset.directory}</code></div>

            <div className="dataset-classes">
              <div className="dataset-section-heading"><Tags size={14} /><strong>类别</strong><span>{dataset.classes.length}</span></div>
              <div className="dataset-class-list">{dataset.classes.length
                ? <>{dataset.classes.slice(0, 8).map((name, index) => <span key={`${index}-${name}`} title={`${index}: ${name}`}>{index}: {name}</span>)}{dataset.classes.length > 8 && <span>+{dataset.classes.length - 8}</span>}</>
                : <small>未记录类别名称</small>}</div>
            </div>

            {dataset.references.length > 0 && <div className="dataset-references">
              <div className="dataset-section-heading"><Link2 size={14} /><strong>引用任务</strong><span>{dataset.references.length}</span></div>
              <div>{dataset.references.map((reference) => <span key={reference.id} title={reference.deleted ? '该任务位于回收站' : undefined}>
                <strong>{reference.name}</strong><em>{reference.deleted ? '回收站' : statusLabel(reference.status)}</em>
              </span>)}</div>
            </div>}

            {dataset.warnings.length > 0 && <div className="dataset-warnings"><AlertTriangle size={14} /><span>{dataset.warnings.join('；')}</span></div>}

            <footer>
              <span>{valueOrZero(dataset.stats.labels).toLocaleString('zh-CN')} 个标签文件{valueOrZero(dataset.stats.background_images) > 0 ? ` · ${dataset.stats.background_images} 张背景图` : ''}</span>
              <button type="button" className="button ghost compact" disabled={!!dataset.blocking_operation_id} onClick={() => setActionTarget(dataset)}>复制 / 迁移</button>
              {dataset.ownership === 'external' && <button type="button" className="button ghost compact" disabled={!!dataset.blocking_operation_id} onClick={() => { const path = window.prompt(`接管后平台将有权删除文件，请输入完整目录确认：\n${dataset.directory}`); if (path) void api.adoptDataset(dataset.id, path).then(loadDatasets).catch((e: Error) => setError(e.message)) }}>接管文件</button>}
              <button type="button" className="button danger compact" disabled={!dataset.can_delete || deleting} title={dataset.delete_block_reason || (dataset.blocking_operation_id ? '请先完成或取消该数据集的未完成操作' : referenced ? '请先更换或永久删除引用该数据集的任务' : '永久删除数据集')} onClick={(event) => openDeleteDialog(dataset, event.currentTarget)}>
                <Trash2 size={14} />{dataset.blocking_operation_id ? '操作未完成' : referenced ? '仍被引用' : '删除'}
              </button>
            </footer>
          </article>
        })}
      </section> : <div className="empty dataset-empty"><Database size={28} /><strong>{datasets.length ? '没有匹配的数据集' : '还没有平台数据集'}</strong><span>{datasets.length ? '请调整搜索关键词或筛选条件。' : '上传 YOLO 数据集文件夹后，可在这里统一管理。'}</span>{datasets.length > 0 && <button type="button" className="button ghost" onClick={() => { setSearch(''); setFilter('all'); setServerFilter('all') }}>重置筛选</button>}{!datasets.length && <button type="button" className="button primary" onClick={() => setUploading(true)}><Upload size={15} />上传数据集</button>}</div>}

    {discoveryOpen && <DatasetDiscoveryModal onClose={() => { setDiscoveryOpen(false); operationsChanged() }} onChanged={operationsChanged} />}
    {storageOpen && <DatasetStorageModal onClose={() => { setStorageOpen(false); setSummaryRefresh(v => v + 1) }} />}
    {historyOpen && <DatasetOperationDrawer onClose={() => setHistoryOpen(false)} onChanged={operationsChanged} summary={operationSummary} servers={servers.map(({ id, name, status }) => ({ id, name, status }))} />}
    {actionTarget && <DatasetActionModal dataset={actionTarget === 'register' ? undefined : actionTarget} onClose={() => setActionTarget(null)} onDone={operationsChanged} />}
    {uploading && !deleting && <UploadDatasetModal onClose={() => setUploading(false)} onUploaded={(result) => { void handleUploaded(result) }} />}

    {deleteTarget && <div className="modal-backdrop dataset-delete-backdrop" onMouseDown={closeDeleteDialog}>
      <div ref={deleteDialog} className="modal dataset-delete-modal" role="alertdialog" aria-modal="true" aria-labelledby="dataset-delete-title" aria-busy={deleting} tabIndex={-1} onKeyDown={handleDeleteDialogKeyDown} onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">DELETE DATASET</span><h2 id="dataset-delete-title">永久删除数据集</h2></div><button type="button" className="icon-button" disabled={deleting} onClick={closeDeleteDialog} aria-label="关闭删除确认"><X size={19} /></button></header>
        <div className="dataset-delete-body">
          <div className="dataset-delete-warning"><AlertTriangle size={21} /><div><strong>删除后无法恢复</strong><span>将永久删除数据集“{deleteTarget.name}”的图片、标签和配置文件。</span></div></div>
          <div className="dataset-delete-detail"><span>目录</span><code>{deleteTarget.directory}</code></div>
          <p>当前没有任务引用此数据集。基础模型、训练记录和其他数据集不受影响。</p>
        </div>
        <footer><button ref={deleteCancelButton} type="button" className="button ghost" disabled={deleting} onClick={closeDeleteDialog}>取消</button><button type="button" className="button danger" disabled={deleting} onClick={() => void deleteDataset()}>{deleting ? <RefreshCw size={15} className="spin" /> : <Trash2 size={15} />}{deleting ? '正在删除…' : '确认永久删除'}</button></footer>
      </div>
    </div>}
  </div>
}
