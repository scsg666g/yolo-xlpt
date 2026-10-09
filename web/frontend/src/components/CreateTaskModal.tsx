import { useEffect, useState } from 'react'
import { ChevronDown, LoaderCircle, X } from 'lucide-react'
import { api } from '../api'
import type { Resources, TaskCreatePayload, TrainingTask } from '../types'

interface Props {
  open: boolean
  onClose: () => void
  onCreated: (task: TrainingTask) => void
}

const initial: TaskCreatePayload = {
  name: '', task_type: 'detect', model_path: '', data_path: '', epochs: 100, batch: 16,
  imgsz: 640, device: '0', workers: 8, optimizer: 'auto', patience: 50, lr0: null,
  pretrained: true, cache: false, amp: true, close_mosaic: 10, advanced: {},
}

export function CreateTaskModal({ open, onClose, onCreated }: Props) {
  const [resources, setResources] = useState<Resources | null>(null)
  const [form, setForm] = useState<TaskCreatePayload>(initial)
  const [advanced, setAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

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
  }, [open])

  if (!open) return null

  const update = <K extends keyof TaskCreatePayload>(key: K, value: TaskCreatePayload[K]) =>
    setForm((current) => ({ ...current, [key]: value }))

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    setError('')
    try {
      const task = await api.create(form)
      setForm(initial)
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
            <label>任务类型<select value={form.task_type} onChange={(e) => update('task_type', e.target.value)}><option value="detect">目标检测</option><option value="segment">实例分割</option><option value="classify">图像分类</option><option value="pose">姿态估计</option><option value="obb">旋转框</option></select></label>
            <label>GPU 设备<select value={form.device} onChange={(e) => update('device', e.target.value)}>{resources?.gpus.map((gpu) => <option key={gpu.index} value={gpu.index}>{gpu.index} · {gpu.name}</option>)}{(resources?.gpus.length || 0) > 1 && <option value={resources!.gpus.map((gpu) => gpu.index).join(',')}>全部 GPU（DDP）</option>}<option value="cpu">CPU</option></select></label>
            <label className="span-2">基础模型<select required value={form.model_path} onChange={(e) => update('model_path', e.target.value)}><option value="">请选择模型</option>{resources?.models.map((file) => <option key={file.path} value={file.path}>{file.name} · {file.path}</option>)}</select></label>
            <label className="span-2">数据集配置<select required value={form.data_path} onChange={(e) => update('data_path', e.target.value)}><option value="">请选择数据集 YAML</option>{resources?.datasets.map((file) => <option key={file.path} value={file.path}>{file.name} · {file.path}</option>)}</select></label>
            <label>训练轮数<input type="number" min="1" value={form.epochs} onChange={(e) => update('epochs', Number(e.target.value))} /></label>
            <label>批大小<input type="number" min="1" value={form.batch} onChange={(e) => update('batch', Number(e.target.value))} /></label>
            <label>输入尺寸<input type="number" min="32" step="32" value={form.imgsz} onChange={(e) => update('imgsz', Number(e.target.value))} /></label>
            <label>数据线程<input type="number" min="0" value={form.workers} onChange={(e) => update('workers', Number(e.target.value))} /></label>
          </div>
          <button type="button" className="advanced-toggle" onClick={() => setAdvanced((value) => !value)}><ChevronDown size={16} className={advanced ? 'rotated' : ''} />高级训练参数</button>
          {advanced && <div className="form-grid advanced-fields">
            <label>优化器<select value={form.optimizer} onChange={(e) => update('optimizer', e.target.value)}><option>auto</option><option>SGD</option><option>Adam</option><option>AdamW</option><option>MuSGD</option></select></label>
            <label>初始学习率<input type="number" step="0.00001" placeholder="自动" value={form.lr0 ?? ''} onChange={(e) => update('lr0', e.target.value ? Number(e.target.value) : null)} /></label>
            <label>早停耐心值<input type="number" min="0" value={form.patience} onChange={(e) => update('patience', Number(e.target.value))} /></label>
            <label>关闭 Mosaic 轮数<input type="number" min="0" value={form.close_mosaic} onChange={(e) => update('close_mosaic', Number(e.target.value))} /></label>
            <label className="check"><input type="checkbox" checked={form.amp} onChange={(e) => update('amp', e.target.checked)} />自动混合精度 AMP</label>
            <label className="check"><input type="checkbox" checked={form.pretrained} onChange={(e) => update('pretrained', e.target.checked)} />使用预训练权重</label>
          </div>}
        </div>
        <footer><button type="button" className="button ghost" onClick={onClose}>取消</button><button className="button primary" disabled={submitting}>{submitting && <LoaderCircle size={16} className="spin" />}创建任务</button></footer>
      </form>
    </div>
  )
}

