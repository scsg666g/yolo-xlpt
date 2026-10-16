import { useEffect, useState } from 'react'
import { Braces, Check, ChevronDown, Clipboard, Code2, LoaderCircle, Upload, X } from 'lucide-react'
import { api } from '../api'
import type { ComputeNode, DatasetUploadResult, Resources, TaskCreatePayload, TrainingTask } from '../types'
import { UploadDatasetModal } from './UploadDatasetModal'
import { analyzeImportedParameters, definitionsFromCatalog, importDefinitions, ParameterImportResults, type DetectedParameter, type ImportDiagnostic } from './EditParametersModal'

interface Props {
  open: boolean
  onClose: () => void
  onCreated: (task: TrainingTask) => void
}

const initial: TaskCreatePayload = {
  name: '', task_type: 'detect', model_path: '', data_path: '', epochs: 100, batch: 16,
  execution_node_id: null, remote_output_root: '',
  imgsz: 640, device: '0', workers: 8, optimizer: 'auto', patience: 50, lr0: null,
  pretrained: true, cache: false, amp: true, close_mosaic: 10, advanced: {},
}

export function CreateTaskModal({ open, onClose, onCreated }: Props) {
  const [resources, setResources] = useState<Resources | null>(null)
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [form, setForm] = useState<TaskCreatePayload>(initial)
  const [advanced, setAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploaded, setUploaded] = useState<DatasetUploadResult | null>(null)
  const [importOpen, setImportOpen] = useState(false)
  const [jsonOpen, setJsonOpen] = useState(false)
  const [jsonCopied, setJsonCopied] = useState(false)
  const [importText, setImportText] = useState('')
  const [detectedParameters, setDetectedParameters] = useState<DetectedParameter[]>([])
  const [importDiagnostics, setImportDiagnostics] = useState<ImportDiagnostic[]>([])
  const [selectedImportKeys, setSelectedImportKeys] = useState<string[]>([])
  const [parameterDefinitions, setParameterDefinitions] = useState(importDefinitions)

  useEffect(() => {
    if (!open) return
    api.resources().then((data) => {
      setResources(data)
      setForm((value) => ({
        ...value,
        model_path: value.model_path || data.models[0]?.path || '',
        data_path: value.data_path || data.datasets[0]?.path || '',
        device: data.gpus.length > 1 ? data.gpus.map((gpu) => gpu.index).join(',') : String(data.gpus[0]?.index ?? 'cpu'),
      }))
    }).catch((reason) => setError(reason.message))
    api.computeNodes().then(setNodes).catch(() => setNodes([]))
  }, [open])

  useEffect(() => {
    if (!open) return
    let active = true
    api.trainingParameters().then((result) => {
      if (active) setParameterDefinitions(result.parameters.length ? definitionsFromCatalog(result.parameters) : importDefinitions)
    }).catch(() => {
      if (active) setParameterDefinitions(importDefinitions)
    })
    return () => { active = false }
  }, [open])

  if (!open) return null

  if (uploading) return <UploadDatasetModal onClose={() => setUploading(false)} onUploaded={(result) => {
    setResources((current) => current ? { ...current, datasets: [...current.datasets.filter((file) => file.path !== result.resource.path), result.resource] } : { gpus: [], models: [], datasets: [result.resource] })
    setForm((current) => ({ ...current, data_path: result.resource.path }))
    setUploaded(result); setError(''); setUploading(false)
  }} />

  const update = <K extends keyof TaskCreatePayload>(key: K, value: TaskCreatePayload[K]) =>
    setForm((current) => ({ ...current, [key]: value }))

  function openImport() { setImportText(''); setDetectedParameters([]); setImportDiagnostics([]); setSelectedImportKeys([]); setImportOpen(true) }
  function changeImportText(value: string) {
    const analysis = analyzeImportedParameters(value, parameterDefinitions)
    setImportText(value); setDetectedParameters(analysis.parameters); setImportDiagnostics(analysis.diagnostics); setSelectedImportKeys(analysis.parameters.map((item) => item.key))
  }
  function applyImportedParameters() {
    const selected = detectedParameters.filter((item) => selectedImportKeys.includes(item.key))
    setForm((current) => {
      const next = { ...current, advanced: { ...current.advanced } }
      for (const item of selected) {
        if (item.key === 'epochs' || item.key === 'batch' || item.key === 'imgsz' || item.key === 'workers' || item.key === 'patience' || item.key === 'close_mosaic') (next as Record<string, unknown>)[item.key] = Number(item.value)
        else if (item.key === 'lr0') next.lr0 = Number(item.value)
        else if (item.key === 'optimizer' || item.key === 'device' || item.key === 'pretrained' || item.key === 'cache' || item.key === 'amp') (next as Record<string, unknown>)[item.key] = item.value
        else next.advanced[item.key] = item.value
      }
      return next
    })
    setAdvanced(true); setImportOpen(false)
  }
  const jsonParameters = { task: form.task_type, mode: 'train', name: form.name, model: form.model_path, data: form.data_path, epochs: form.epochs, batch: form.batch, imgsz: form.imgsz, device: form.device, workers: form.workers, optimizer: form.optimizer, patience: form.patience, lr0: form.lr0, pretrained: form.pretrained, cache: form.cache, amp: form.amp, close_mosaic: form.close_mosaic, ...form.advanced }
  const isRemote = Boolean(form.execution_node_id)
  const executionNode = nodes.find((node) => node.id === form.execution_node_id)
  const jsonText = JSON.stringify(jsonParameters, null, 2)
  async function copyJson() { try { await navigator.clipboard.writeText(jsonText); setJsonCopied(true); window.setTimeout(() => setJsonCopied(false), 1600) } catch { setError('复制失败，请手动复制 JSON') } }

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      const task = await api.create(form)
      setForm(initial)
      setUploaded(null); setImportOpen(false); setJsonOpen(false)
      onCreated(task)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '创建失败')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <form className="modal" onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">NEW TRAINING RUN</span><h2>创建训练任务</h2></div><button type="button" className="icon-button" onClick={onClose}><X size={20} /></button></header>
        <div className="modal-body">
          {error && <div className="alert error">{error}</div>}
          <div className="form-grid">
            <label className="span-2">任务名称<input required value={form.name} onChange={(e) => update('name', e.target.value)} placeholder="例如：反光背心 YOLO26x 试验 02" /></label>
            <label className="span-2">执行服务器<select value={form.execution_node_id || ''} onChange={(e) => { const id = e.target.value || null; setForm((current) => ({ ...current, execution_node_id: id, model_path: id ? '' : (resources?.models[0]?.path || ''), data_path: id ? '' : (resources?.datasets[0]?.path || ''), device: id ? '0' : String(resources?.gpus[0]?.index ?? 'cpu'), remote_output_root: '' })) }}><option value="">本机（中台服务器）</option>{nodes.filter((node) => node.kind === 'agent' && node.enabled).map((node) => <option key={node.id} value={node.id}>{node.name} · {node.address || node.hostname || node.status}</option>)}</select><small>{isRemote ? `远程 Agent：${executionNode?.training ? (executionNode.training.ready ? '训练已就绪' : executionNode.training.reason || '训练环境未就绪') : '训练能力未知，请升级 Agent'}；${executionNode?.status === 'online' ? '在线' : `当前${executionNode?.status || '未知'}，任务会等待 Agent 拉取`}` : '使用中台服务器本机资源与路径'}</small></label>
            <label>任务类型<select value={form.task_type} onChange={(e) => update('task_type', e.target.value)}><option value="detect">目标检测</option><option value="segment">实例分割</option><option value="classify">图像分类</option><option value="pose">姿态估计</option><option value="obb">旋转框</option></select></label>
            <label>GPU 设备{isRemote ? <input value={form.device} onChange={(e) => update('device', e.target.value)} placeholder="例如 0 或 0,1；也可填 cpu" /> : <select value={form.device} onChange={(e) => update('device', e.target.value)}>{resources?.gpus.map((gpu) => <option key={gpu.index} value={gpu.index}>{gpu.index} · {gpu.name}</option>)}{(resources?.gpus.length || 0) > 1 && <option value={resources!.gpus.map((gpu) => gpu.index).join(',')}>全部 GPU（DDP）</option>}<option value="cpu">CPU</option></select>}</label>
            {isRemote ? <><label className="span-2">远程基础模型绝对路径<input required value={form.model_path} onChange={(e) => update('model_path', e.target.value)} placeholder="例如 /data/yolo/models/yolo26x.pt" /></label><label className="span-2">远程数据集 YAML 绝对路径<input required value={form.data_path} onChange={(e) => update('data_path', e.target.value)} placeholder="例如 /data/yolo/datasets/mydata/data.yaml" /></label><label className="span-2">远程训练输出根目录<input required value={form.remote_output_root} onChange={(e) => update('remote_output_root', e.target.value)} placeholder="例如 /data/yolo/runs" /></label><div className="alert warning span-2" role="note" style={{ gridColumn: 'span 2', marginBottom: 0 }}><strong>路径必须存在于远程服务器“{executionNode?.name || '当前所选服务器'}”上。</strong><br />请填写该服务器上的模型、YAML 和输出目录路径；不能直接使用中心平台资源列表中的本机路径。数据不会自动复制。</div></> : <><label className="span-2">基础模型<select required value={form.model_path} onChange={(e) => update('model_path', e.target.value)}><option value="">请选择模型</option>{resources?.models.map((file) => <option key={file.path} value={file.path}>{file.name} · {file.path}</option>)}</select></label><div className="dataset-field"><div className="dataset-field-heading"><label htmlFor="task-dataset">数据集配置</label><button type="button" className="button ghost compact" disabled={form.task_type !== 'detect'} onClick={() => setUploading(true)} title={form.task_type === 'detect' ? '上传本地数据集文件夹' : '文件夹上传目前支持目标检测，请选择已有数据集配置'}><Upload size={14} />上传数据集</button></div><label><select id="task-dataset" required value={form.data_path} onChange={(e) => { update('data_path', e.target.value); setUploaded(null) }}><option value="">请选择数据集 YAML</option>{resources?.datasets.map((file) => <option key={file.path} value={file.path}>{file.name} · {file.path}</option>)}</select></label>{form.task_type !== 'detect' && <small className="upload-help">文件夹上传目前支持目标检测，其他任务请使用已有数据集配置。</small>}</div></>}
            {uploaded && <div className="alert dataset-upload-success span-2" role="status" style={{ gridColumn: 'span 2', marginBottom: 0 }}><strong>已上传并选中：{uploaded.name}</strong><p>训练 {uploaded.stats.train_images} 张 · 验证 {uploaded.stats.val_images} 张 · {uploaded.stats.classes} 个类别</p><p>path: {uploaded.directory}</p>{uploaded.warnings.map((warning) => <p key={warning}>{warning}</p>)}</div>}
            <label>训练轮数<input type="number" min="1" value={form.epochs} onChange={(e) => update('epochs', Number(e.target.value))} /></label>
            <label>批大小<input type="number" min="1" value={form.batch} onChange={(e) => update('batch', Number(e.target.value))} /></label>
            <label>输入尺寸<input type="number" min="32" step="32" value={form.imgsz} onChange={(e) => update('imgsz', Number(e.target.value))} /></label>
            <label>数据线程<input type="number" min="0" value={form.workers} onChange={(e) => update('workers', Number(e.target.value))} /></label>
          </div>
          <div className="parameter-toolbar create-parameter-toolbar"><button type="button" className="advanced-toggle parameter-toggle" onClick={() => setAdvanced((value) => !value)}><ChevronDown size={16} className={advanced ? 'rotated' : ''} />高级训练参数</button><button type="button" className="json-preview-button" onClick={() => { setJsonCopied(false); setJsonOpen(true) }}><Code2 size={14} />查看 JSON</button><button type="button" className="batch-import-button" onClick={openImport}><Braces size={14} />批量导入参数</button></div>
          {advanced && <div className="form-grid advanced-fields">
            <label>优化器<select value={form.optimizer} onChange={(e) => update('optimizer', e.target.value)}><option>auto</option><option>SGD</option><option>Adam</option><option>AdamW</option><option>MuSGD</option></select></label>
            <label>初始学习率<input type="number" step="0.00001" placeholder="自动" value={form.lr0 ?? ''} onChange={(e) => update('lr0', e.target.value ? Number(e.target.value) : null)} /></label>
            <label>早停耐心值<input type="number" min="0" value={form.patience} onChange={(e) => update('patience', Number(e.target.value))} /></label>
            <label>关闭 Mosaic 轮数<input type="number" min="0" value={form.close_mosaic} onChange={(e) => update('close_mosaic', Number(e.target.value))} /></label>
            <label className="check"><input type="checkbox" checked={form.amp} onChange={(e) => update('amp', e.target.checked)} />自动混合精度 AMP</label>
            <label className="check"><input type="checkbox" checked={form.pretrained} onChange={(e) => update('pretrained', e.target.checked)} />使用预训练权重</label>
            <label className="check"><input type="checkbox" checked={Boolean(form.cache)} onChange={(e) => update('cache', e.target.checked)} />缓存数据</label>
            {Object.entries(form.advanced).map(([key, value]) => <label key={key} className="imported-advanced-field"><span>{key}</span>{typeof value === 'boolean' ? <select value={String(value)} onChange={(e) => setForm((current) => ({ ...current, advanced: { ...current.advanced, [key]: e.target.value === 'true' } }))}><option value="true">开启</option><option value="false">关闭</option></select> : <input type={typeof value === 'number' ? 'number' : 'text'} value={String(value)} onChange={(e) => setForm((current) => ({ ...current, advanced: { ...current.advanced, [key]: typeof value === 'number' ? Number(e.target.value) : e.target.value } }))} />}</label>)}
          </div>}
        </div>
        {importOpen && <div className="parameter-import-backdrop" onMouseDown={() => setImportOpen(false)}><div className="parameter-import-modal" onMouseDown={(event) => event.stopPropagation()}><header><div><span className="eyebrow">PASTE CONFIGURATION</span><h3>批量识别训练参数</h3><p>可粘贴 Python、YOLO CLI、YAML 或 JSON 样式的参数。全部字段都会列出诊断结果，不会执行粘贴的代码。</p></div><button type="button" className="icon-button" onClick={() => setImportOpen(false)}><X size={18} /></button></header><div className="parameter-import-body"><textarea value={importText} onChange={(event) => changeImportText(event.target.value)} placeholder={'例如：\nmodel.train(epochs=100, batch=16, imgsz=640, lr0=0.01, freeze=10)\n\n或：\nepochs: 100\namp: true\nmosaic: 0.8'} /><ParameterImportResults diagnostics={importDiagnostics} parameters={detectedParameters} selectedKeys={selectedImportKeys} onSelectionChange={setSelectedImportKeys} /></div><footer><span>将导入 {selectedImportKeys.length} 项{importDiagnostics.length > detectedParameters.length ? `，${importDiagnostics.length - detectedParameters.length} 项不会导入` : ''}</span><button type="button" className="button ghost" onClick={() => setImportOpen(false)}>取消</button><button type="button" className="button primary" disabled={!selectedImportKeys.length} onClick={applyImportedParameters}>导入 {selectedImportKeys.length} 个参数</button></footer></div></div>}
        {jsonOpen && <div className="parameter-import-backdrop" onMouseDown={() => setJsonOpen(false)}><div className="parameter-json-modal" onMouseDown={(event) => event.stopPropagation()}><header><div><span className="eyebrow">EFFECTIVE CONFIGURATION</span><h3>完整 JSON 参数</h3><p>{Object.keys(jsonParameters).length} 个当前参数</p></div><button type="button" className="icon-button" onClick={() => setJsonOpen(false)}><X size={18} /></button></header><div className="parameter-json-body"><pre>{jsonText}</pre></div><footer><span>当前新建任务配置</span><button type="button" className="button primary" onClick={copyJson}>{jsonCopied ? <Check size={15} /> : <Clipboard size={15} />}{jsonCopied ? '已复制' : '复制 JSON'}</button></footer></div></div>}
        <footer><button type="button" className="button ghost" onClick={onClose}>取消</button><button className="button primary" disabled={submitting || Boolean(isRemote && executionNode?.training && !executionNode.training.ready)}>{submitting && <LoaderCircle size={16} className="spin" />}创建任务</button></footer>
      </form>
    </div>
  )
}
