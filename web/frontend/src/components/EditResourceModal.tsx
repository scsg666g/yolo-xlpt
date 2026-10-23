import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Box, Database, FileCheck2, FolderSearch, LoaderCircle, X } from 'lucide-react'
import { api, formatBytes } from '../api'
import type { ResourceFile, Resources, TrainingTask } from '../types'
import { useModalFocus } from '../useModalFocus'
import { remoteTrainingDataPathError } from '../trainingDataPaths'

interface Props {
  task: TrainingTask | null
  mode: 'dataset' | 'model' | null
  onClose: () => void
  onSave: (mode: 'dataset' | 'model', path: string) => Promise<void>
}

export function EditResourceModal({ task, mode, onClose, onSave }: Props) {
  const [resources, setResources] = useState<Resources | null>(null)
  const [path, setPath] = useState('')
  const [originalPath, setOriginalPath] = useState('')
  const [search, setSearch] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const latestTask = useRef(task)
  const sessionKey = task && mode ? `${task.id}:${mode}` : ''
  const initializedSession = useRef('')
  latestTask.current = task
  const normalizedPath = path.trim()
  const dirty = normalizedPath !== originalPath
  const requestClose = useCallback(() => {
    if (saving) return
    if (dirty && !window.confirm('当前路径尚未保存，确定放弃修改吗？')) return
    onClose()
  }, [dirty, onClose, saving])
  const dialogRef = useModalFocus<HTMLFormElement>(Boolean(sessionKey), requestClose, {
    canClose: !saving,
    initialFocusSelector: '[data-modal-initial-focus]',
  })

  useEffect(() => {
    const currentTask = latestTask.current
    if (!currentTask || !mode) {
      initializedSession.current = ''
      return
    }
    const key = `${currentTask.id}:${mode}`
    if (initializedSession.current === key) return
    initializedSession.current = key
    let active = true
    setResources(null)
    const initialPath = mode === 'dataset' ? currentTask.data_path : currentTask.model_path
    setPath(initialPath)
    setOriginalPath(initialPath.trim())
    setSearch(''); setError('')
    if (!currentTask.execution_node_id) {
      api.resources().then((data) => { if (active) setResources(data) }).catch((reason) => { if (active) setError(reason.message) })
    }
    return () => { active = false }
  }, [task?.id, mode])

  const files = useMemo(() => {
    const source = mode === 'dataset'
      ? (task?.task_type === 'classify' ? resources?.classification_datasets : resources?.datasets)
      : resources?.models
    return (source || []).filter((file) => `${file.name} ${file.path}`.toLowerCase().includes(search.toLowerCase()))
  }, [resources, mode, search, task?.task_type])

  if (!task || !mode) return null
  const isDataset = mode === 'dataset'
  const isClassificationDataset = isDataset && task.task_type === 'classify'
  const locked = ['queued', 'starting', 'running', 'stopping'].includes(task.status)
  const pathError = task.execution_node_id && isDataset ? remoteTrainingDataPathError(task.task_type, path) : null
  const currentMissing = !task.execution_node_id && !files.some((file) => file.path === path) && path === (isDataset ? task.data_path : task.model_path)

  async function save(event: React.FormEvent) {
    event.preventDefault()
    if (!task || !mode || !normalizedPath || !dirty || pathError) return
    setSaving(true); setError('')
    try {
      await onSave(mode, normalizedPath)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保存失败')
    } finally { setSaving(false) }
  }

  function selectFile(file: ResourceFile) { setPath(file.path) }

  const titleId = `resource-modal-title-${mode}`
  return <div className="modal-backdrop resource-backdrop" onMouseDown={requestClose}>
    <form ref={dialogRef} className="modal resource-modal" role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1} onSubmit={save} onMouseDown={(event) => event.stopPropagation()}>
      <header><div className="modal-heading"><div className={`heading-icon ${isDataset ? 'dataset-icon' : 'model-icon'}`}>{isDataset ? <Database size={18} /> : <Box size={18} />}</div><div><span className="eyebrow">{isDataset ? 'DATASET SOURCE' : 'MODEL WEIGHTS'}</span><h2 id={titleId}>{isDataset ? '更换训练数据集' : '更换基础模型'}</h2><p>双击节点可随时重新绑定有效资源</p></div></div><button type="button" className="icon-button" data-modal-initial-focus aria-label={`关闭${isDataset ? '更换数据集' : '更换模型'}窗口`} disabled={saving} onClick={requestClose}><X size={20} /></button></header>
      <div className="modal-body">
        {locked && <div className="alert warning">任务正在运行或排队，请停止任务后再更换资源。</div>}
        {currentMissing && <div className="alert error">当前路径已经失效，请从下方选择服务器中存在的文件。</div>}
        {pathError && <div className="alert error">{pathError}</div>}
        {error && <div className="alert error">{error}</div>}
        <label className="resource-path-label">当前选择<input value={path} onChange={(event) => setPath(event.target.value)} placeholder={isClassificationDataset ? '分类数据集目录的绝对路径' : isDataset ? '数据集 YAML 的绝对路径' : '模型权重的绝对路径'} /></label>
        {task.execution_node_id && <div className="node-info">填写执行服务器上的绝对路径；{isClassificationDataset ? '分类任务需要包含 train/val 的目录。' : ''}Agent 将在启动训练前检查路径和允许目录。</div>}
        {!task.execution_node_id && <><div className="resource-toolbar"><div><FolderSearch size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索名称或路径" /></div><span>发现 {files.length} 个可用文件</span></div>
        <div className="resource-list">
          {files.map((file) => <button type="button" key={file.path} className={path === file.path ? 'selected' : ''} onClick={() => selectFile(file)}><span className="resource-file-icon">{isDataset ? <Database size={16} /> : <Box size={16} />}</span><span><strong>{file.name}</strong><small>{file.path}</small></span><em>{formatBytes(file.size)}</em>{path === file.path && <FileCheck2 className="selected-check" size={16} />}</button>)}
          {!files.length && <div className="resource-empty">没有找到匹配的文件</div>}
        </div></>}
      </div>
      <footer><div className="resource-footer-note"><span>{task.execution_node_id ? '远程 Agent 将验证路径及权限' : '仅允许选择平台批准目录中的文件'}</span>{dirty && <strong>有未保存的修改</strong>}</div><button type="button" className="button ghost" disabled={saving} onClick={requestClose}>取消</button><button className="button primary" disabled={saving || locked || Boolean(pathError) || !normalizedPath || !dirty}>{saving && <LoaderCircle size={16} className="spin" />}{saving ? '保存中' : `保存${isDataset ? '数据集' : '模型'}`}</button></footer>
    </form>
  </div>
}
