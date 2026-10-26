import { DatasetStorageSettings, DatasetOperationHistory, DatasetActionModal } from './DatasetStorage'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
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
import type { DatasetUploadResult, ManagedDataset } from '../types'
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

export function Datasets() {
  const [datasets, setDatasets] = useState<ManagedDataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<DatasetFilter>('all')
  const [uploading, setUploading] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<ManagedDataset | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [actionTarget, setActionTarget] = useState<ManagedDataset | 'register' | null>(null)
  const [serverFilter, setServerFilter] = useState('all')
  const loadRequest = useRef<{ version: number; controller: AbortController } | null>(null)
  const loadVersion = useRef(0)
  const deleteDialog = useRef<HTMLDivElement>(null)
  const deleteCancelButton = useRef<HTMLButtonElement>(null)
  const deleteOpener = useRef<HTMLButtonElement | null>(null)

  const loadDatasets = useCallback(async () => {
    loadRequest.current?.controller.abort()
    const version = ++loadVersion.current
    const controller = new AbortController()
    loadRequest.current = { version, controller }
    setLoading(true)
    setError('')
    try {
      const incoming = await api.datasetEntries(controller.signal)
      if (version === loadVersion.current) setDatasets(incoming)
    } catch (reason) {
      if (controller.signal.aborted || version !== loadVersion.current) return
      setError(reason instanceof Error ? reason.message : '数据集列表加载失败')
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
      if (filter === 'deletable' && dataset.references.length) return false
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
      setDatasets((current) => current.filter((dataset) => dataset.id !== deleteTarget.id))
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
      if (!preview.can_delete) { setError('该数据集仍被引用、未接管或正在处理，请刷新后检查'); return }
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
        <p>统一管理本机和远程数据集：保存位置、上传、已有目录登记、复制迁移及安全删除。</p>
      </div>
      <div className="header-actions datasets-header-actions">
        <button type="button" className="button ghost" disabled={loading || deleting} onClick={() => { setNotice(''); void loadDatasets() }}>
          <RefreshCw size={16} className={loading ? 'spin' : ''} />刷新
        </button>
        <button type="button" className="button primary" disabled={deleting} onClick={() => { setError(''); setNotice(''); setUploading(true) }}>
          <Upload size={16} />上传数据集
        </button>
      </div>
    </header>

    {notice && <div className="dataset-feedback success" role="status"><CheckCircle2 size={17} />{notice}</div>}
    {error && <div className="dataset-feedback error" role="alert"><AlertTriangle size={17} /><span>{error}</span><button type="button" onClick={() => setError('')} aria-label="关闭错误提示"><X size={15} /></button></div>}

    <details className="dataset-storage-panel"><summary>保存位置配置</summary><DatasetStorageSettings /></details>
    <DatasetOperationHistory onChanged={loadDatasets} />
    <button type="button" className="button ghost" onClick={() => setActionTarget('register')}>登记已有服务器目录</button>

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
      <select aria-label="按服务器筛选" value={serverFilter} onChange={e => setServerFilter(e.target.value)}><option value="all">全部服务器</option>{Array.from(new Set(datasets.map(d => d.node_id))).map(id => <option key={id} value={id}>{datasets.find(d => d.node_id === id)?.node_name || (id === 'local' ? '本机' : `远程 ${id.slice(0, 8)}`)}</option>)}</select>
      <span className="dataset-result-count">当前显示 {visibleDatasets.length} / {datasets.length}</span>
    </section>

    {loading && !datasets.length ? <div className="empty dataset-empty"><RefreshCw className="spin" size={24} /><strong>正在加载数据集…</strong></div>
      : visibleDatasets.length ? <section className="dataset-grid" aria-label="数据集列表">
        {visibleDatasets.map((dataset) => {
          const referenced = dataset.references.length > 0
          const classCount = dataset.stats.classes ?? dataset.classes.length
          return <article className="dataset-card" key={dataset.directory}>
            <header>
              <div className="dataset-card-icon"><Database size={19} /></div>
              <div className="dataset-card-title"><h2 title={dataset.name}>{dataset.name}</h2><span>{dataset.node_name || (dataset.node_id === 'local' ? '本机' : '远程')} · {dataset.ownership === 'managed' ? '平台管理' : '外部登记'} · {formatCreatedAt(dataset.created_at)}</span></div>
              <span className={`dataset-reference-state ${referenced ? 'referenced' : 'free'}`}><i />{referenced ? `${dataset.references.length} 个引用` : dataset.ownership === 'managed' ? '无引用' : '外部文件'}</span>
            </header>

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
              <button type="button" className="button ghost compact" onClick={() => setActionTarget(dataset)}>复制 / 迁移</button>
              {dataset.ownership === 'external' && <button type="button" className="button ghost compact" onClick={() => { const path = window.prompt(`接管后平台将有权删除文件，请输入完整目录确认：\n${dataset.directory}`); if (path) void api.adoptDataset(dataset.id, path).then(loadDatasets).catch((e: Error) => setError(e.message)) }}>接管文件</button>}
              <button type="button" className="button danger compact" disabled={referenced || deleting || dataset.ownership !== 'managed'} title={referenced ? '请先更换或永久删除引用该数据集的任务' : '永久删除数据集'} onClick={(event) => openDeleteDialog(dataset, event.currentTarget)}>
                <Trash2 size={14} />{referenced ? '仍被引用' : '删除'}
              </button>
            </footer>
          </article>
        })}
      </section> : <div className="empty dataset-empty"><Database size={28} /><strong>{datasets.length ? '没有匹配的数据集' : '还没有平台数据集'}</strong><span>{datasets.length ? '请调整搜索关键词或筛选条件。' : '上传 YOLO 数据集文件夹后，可在这里统一管理。'}</span>{!datasets.length && <button type="button" className="button primary" onClick={() => setUploading(true)}><Upload size={15} />上传数据集</button>}</div>}

    {actionTarget && <DatasetActionModal dataset={actionTarget === 'register' ? undefined : actionTarget} onClose={() => setActionTarget(null)} onDone={() => { void loadDatasets() }} />}
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
