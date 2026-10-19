import { useEffect, useMemo, useRef, useState } from 'react'
import { FolderOpen, LoaderCircle, Upload, X } from 'lucide-react'
import { api, formatBytes, uploadDatasetFile } from '../api'
import type { DatasetUploadResult } from '../types'
import { useModalFocus } from '../useModalFocus'
import '../upload.css'

interface Props {
  onClose: () => void
  onUploaded: (result: DatasetUploadResult) => void
}

const imageExtensions = /\.(jpg|jpeg|png|bmp|webp|tif|tiff)$/i
const ignoredFiles = /(^|\/)(\.[^/]+|__MACOSX|Thumbs\.db|desktop\.ini)(\/|$)|\.cache$/i
const directoryAttributes = { webkitdirectory: '', directory: '' }

export function UploadDatasetModal({ onClose, onUploaded }: Props) {
  const [files, setFiles] = useState<File[]>([])
  const [folderName, setFolderName] = useState('')
  const [namesText, setNamesText] = useState('')
  const [error, setError] = useState('')
  const [selectionError, setSelectionError] = useState('')
  const [ignoredCount, setIgnoredCount] = useState(0)
  const [busy, setBusy] = useState(false)
  const [finalizing, setFinalizing] = useState(false)
  const [stage, setStage] = useState('')
  const [bytes, setBytes] = useState(0)
  const [completedFiles, setCompletedFiles] = useState(0)
  const [destination, setDestination] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const uploadId = useRef<string | null>(null)
  const controller = useRef<AbortController | null>(null)
  const selectionVersion = useRef(0)
  const active = useRef(true)
  const dialogRef = useModalFocus<HTMLFormElement>(true, onClose, {
    canClose: !busy,
    initialFocusSelector: '[data-modal-initial-focus]',
  })

  useEffect(() => {
    active.current = true
    return () => {
      active.current = false
      controller.current?.abort()
      if (uploadId.current) void api.cancelDatasetUpload(uploadId.current).catch(() => undefined)
    }
  }, [])

  const { totalBytes, trainCount, valCount } = useMemo(() => {
    let totalBytes = 0; let trainCount = 0; let valCount = 0
    for (const file of files) {
      totalBytes += file.size
      if (!imageExtensions.test(file.name)) continue
      if (file.webkitRelativePath.startsWith(`${folderName}/images/train/`)) trainCount++
      if (file.webkitRelativePath.startsWith(`${folderName}/images/val/`)) valCount++
    }
    return { totalBytes, trainCount, valCount }
  }, [files, folderName])
  const names = namesText.split(/\r?\n/).map((name) => name.trim()).filter(Boolean)
  const percentage = totalBytes ? Math.min(100, Math.round(bytes / totalBytes * 100)) : 0

  async function selectFolder(list: FileList | null) {
    if (!list?.length) return
    const version = ++selectionVersion.current
    if (uploadId.current) void api.cancelDatasetUpload(uploadId.current).catch(() => undefined)
    uploadId.current = null
    setError(''); setSelectionError(''); setNamesText(''); setDestination(''); setBytes(0); setCompletedFiles(0); setStage('')
    const all = Array.from(list)
    const selected = all.filter((file) => !ignoredFiles.test(file.webkitRelativePath))
    const name = all[0].webkitRelativePath.split('/')[0]
    setFiles(selected); setFolderName(name); setIgnoredCount(all.length - selected.length)
    const relative = (file: File) => file.webkitRelativePath.slice(name.length + 1)
    const invalid = selected.find((file) => {
      const path = relative(file)
      return !['data.yaml', 'data.yml', 'classes.txt'].includes(path) &&
        !(/^images\/(train|val)\/.+/.test(path) && imageExtensions.test(path)) &&
        !(/^labels\/(train|val)\/.+\.txt$/.test(path))
    })
    if (!name || selected.some((file) => !file.webkitRelativePath.startsWith(`${name}/`))) {
      setSelectionError('请选择包含 images 和 labels 的最外层文件夹。'); return
    }
    if (invalid) { setSelectionError(`不支持的文件或目录：${relative(invalid)}。请按下方目录格式整理。`); return }
    if (!selected.some((file) => /^images\/train\//.test(relative(file))) || !selected.some((file) => /^images\/val\//.test(relative(file)))) {
      setSelectionError('images/train 和 images/val 都需要至少一张图片，请选择最外层数据集文件夹。'); return
    }
    if (selected.length > 200000 || selected.reduce((sum, file) => sum + file.size, 0) > 50 * 1024 ** 3) {
      setSelectionError('数据集最多支持 20 万个文件，总大小不能超过 50 GB。'); return
    }
    const oversized = selected.find((file) => file.size > (imageExtensions.test(file.name) ? 256 : 4) * 1024 ** 2)
    if (oversized) { setSelectionError(`文件过大：${relative(oversized)}。单张图片最大 256 MB，标签或配置文件最大 4 MB。`); return }
    const classes = selected.find((file) => relative(file) === 'classes.txt')
    if (classes) {
      try {
        const text = (await classes.text()).replace(/^\uFEFF/, '')
        if (active.current && selectionVersion.current === version) setNamesText(text)
      } catch { if (active.current && selectionVersion.current === version) setError('无法读取 classes.txt，请手动填写类别名称。') }
    }
  }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (busy || !files.length || selectionError) return
    const hasConfig = files.some((file) => ['data.yaml', 'data.yml'].includes(file.webkitRelativePath.slice(folderName.length + 1)))
    if (!names.length && !hasConfig) { setError('请填写类别名称，每行一个；类别编号按顺序从 0 开始。'); return }
    if (new Set(names).size !== names.length) { setError('类别名称不能重复。'); return }
    const abortController = new AbortController()
    controller.current = abortController
    const signal = abortController.signal
    setBusy(true); setError(''); setStage('准备上传…')
    try {
      if (!uploadId.current) {
        const session = await api.createDatasetUpload(folderName, files.map((file) => ({ path: file.webkitRelativePath.slice(folderName.length + 1), size: file.size })), signal)
        uploadId.current = session.upload_id
        setDestination(session.destination)
      }
      const id = uploadId.current
      const status = await api.datasetUploadStatus(id, signal)
      if (status.result) { uploadId.current = null; if (active.current) onUploaded(status.result); return }
      const uploaded = new Set(status.uploaded_indices)
      let sent = files.reduce((sum, file, index) => sum + (uploaded.has(index) ? file.size : 0), 0)
      let count = uploaded.size
      setBytes(sent); setCompletedFiles(count)
      for (let index = 0; index < files.length; index++) {
        if (uploaded.has(index)) continue
        if (signal.aborted) throw new DOMException('上传已取消', 'AbortError')
        const file = files[index]
        setStage(`正在上传：${file.webkitRelativePath.slice(folderName.length + 1)}`)
        await uploadDatasetFile(id, index, file, signal, (loaded) => { if (active.current) setBytes(sent + loaded) })
        sent += file.size; count += 1
        if (active.current) { setBytes(sent); setCompletedFiles(count) }
      }
      setFinalizing(true)
      setStage('上传完成，正在检查图片和标签并生成配置，请稍候…')
      const result = await api.completeDatasetUpload(id, names, signal)
      uploadId.current = null
      if (active.current) onUploaded(result)
    } catch (reason) {
      if (active.current) setError(reason instanceof DOMException && reason.name === 'AbortError'
        ? '上传已暂停，点击继续上传可复用已完成的文件。'
        : reason instanceof Error ? reason.message : '上传失败，请重试')
    } finally {
      if (active.current) { setBusy(false); setFinalizing(false) }
      controller.current = null
    }
  }

  return <div className="modal-backdrop dataset-upload-backdrop" onMouseDown={() => { if (!busy) onClose() }}>
    <form ref={dialogRef} className="modal dataset-upload-modal" role="dialog" aria-modal="true" aria-labelledby="dataset-upload-title" tabIndex={-1} onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="eyebrow">UPLOAD DATASET</span><h2 id="dataset-upload-title">上传数据集文件夹</h2></div><button type="button" className="icon-button" disabled={busy} onClick={onClose} aria-label="关闭上传窗口"><X size={20} /></button></header>
      <div className="modal-body">
        <p className="upload-intro">保留你选择的文件夹原名，上传后自动生成 data.yaml 并选中该数据集。</p>
        <div className="upload-format">
          <strong>目录格式要求 · YOLO 目标检测</strong>
          <p>请选择包含 images 和 labels 的最外层文件夹。</p>
          <pre>{`${folderName || '我的数据集'}/\n├── images/\n│   ├── train/   # 训练图片\n│   └── val/     # 验证图片\n└── labels/\n    ├── train/   # 训练标签 .txt\n    └── val/     # 验证标签 .txt`}</pre>
          <ul><li>图片与标签同名：images/train/a.jpg 对应 labels/train/a.txt。</li><li>标签每行：类别编号 中心x 中心y 宽 高，坐标在 0～1 之间，宽高大于 0。</li><li>类别编号从 0 开始；无标签或空标签的图片会作为背景图提示。</li></ul>
          <details><summary>标签示例与文件限制</summary><pre>0 0.5 0.5 0.2 0.3</pre><p>支持 JPG、JPEG、PNG、BMP、WEBP、TIF、TIFF；单张图片最大 256 MB，标签或配置最大 4 MB，总计最大 50 GB / 20 万个文件。第一版仅支持目标检测。</p></details>
        </div>
        <input ref={inputRef} type="file" multiple {...directoryAttributes} hidden disabled={busy} onChange={(event) => { void selectFolder(event.target.files); event.target.value = '' }} aria-label="选择数据集文件夹" />
        <button type="button" className="upload-folder-picker" data-modal-initial-focus disabled={busy} onClick={() => inputRef.current?.click()}><FolderOpen size={26} /><span><strong>{folderName || '选择数据集文件夹'}</strong><small>{files.length ? `${files.length} 个文件 · ${formatBytes(totalBytes)} · 训练 ${trainCount} 张 / 验证 ${valCount} 张` : '点击选择本地文件夹，支持中文目录名'}</small></span><span className="button compact">{folderName ? '重新选择' : '选择文件夹'}</span></button>
        {ignoredCount > 0 && <p className="upload-help">已忽略 {ignoredCount} 个隐藏文件、系统文件或缓存文件。</p>}
        {folderName && <p className="upload-help">服务器目录名：<strong>{folderName}</strong>。存在同名目录时会提示，不覆盖已有数据。</p>}
        {destination && <p className="upload-destination">data.yaml 中的 path：<code>{destination}</code></p>}
        <label className="upload-classes">类别名称（每行一个）<textarea rows={3} value={namesText} disabled={busy} onChange={(event) => setNamesText(event.target.value)} placeholder={'人员\n安全帽'} /><small>第 1 行对应编号 0，第 2 行对应编号 1，依次排列。若文件夹已有带 names 的 data.yaml，可留空自动读取。</small></label>
        {names.length > 0 && <div className="upload-class-preview">{names.slice(0, 20).map((name, index) => <span key={index}>{index}：{name}</span>)}{names.length > 20 && <span>共 {names.length} 个类别</span>}</div>}
        {(selectionError || error) && <div className="alert error" role="alert">{selectionError || error}</div>}
        {stage && <div className="upload-progress" role="status" aria-live="polite"><div><span>{stage}</span><strong>{percentage}%</strong></div><progress max={100} value={percentage} /><small>{completedFiles} / {files.length} 个文件 · {formatBytes(bytes)} / {formatBytes(totalBytes)}</small></div>}
      </div>
      <footer>{busy ? <button type="button" className="button ghost" disabled={finalizing} onClick={() => controller.current?.abort()}>{finalizing ? '正在校验并保存' : '暂停上传'}</button> : <button type="button" className="button ghost" onClick={onClose}>取消</button>}<button type="submit" className="button primary" disabled={busy || !files.length || !!selectionError}>{busy ? <LoaderCircle size={16} className="spin" /> : <Upload size={16} />}{busy ? '正在处理…' : uploadId.current ? '继续上传 / 重试' : '上传并使用'}</button></footer>
    </form>
  </div>
}
