import { useEffect, useMemo, useState } from 'react'
import { Braces, ChevronDown, LoaderCircle, Plus, SlidersHorizontal, Trash2, X } from 'lucide-react'
import { api } from '../api'
import type { TrainingTask } from '../types'

interface Props {
  task: TrainingTask | null
  open: boolean
  onClose: () => void
  onSaved: (task: TrainingTask) => void
}

type ExtraValue = string | number | boolean

const extraDefinitions: Record<string, { label: string; type: 'number' | 'boolean' | 'text'; default: ExtraValue; step?: number }> = {
  weight_decay: { label: '权重衰减', type: 'number', default: 0.0005, step: 0.0001 },
  momentum: { label: '动量', type: 'number', default: 0.937, step: 0.001 },
  warmup_epochs: { label: '预热轮数', type: 'number', default: 3, step: 0.1 },
  cos_lr: { label: '余弦学习率', type: 'boolean', default: false },
  rect: { label: '矩形训练', type: 'boolean', default: false },
  deterministic: { label: '确定性训练', type: 'boolean', default: true },
  seed: { label: '随机种子', type: 'number', default: 0 },
  fraction: { label: '数据集比例', type: 'number', default: 1, step: 0.1 },
  freeze: { label: '冻结层', type: 'text', default: '' },
  multi_scale: { label: '多尺度训练', type: 'number', default: 0, step: 0.1 },
  save_period: { label: '权重保存周期', type: 'number', default: -1 },
  hsv_h: { label: 'HSV 色相增强', type: 'number', default: 0.015, step: 0.001 },
  hsv_s: { label: 'HSV 饱和度增强', type: 'number', default: 0.7, step: 0.05 },
  hsv_v: { label: 'HSV 明度增强', type: 'number', default: 0.4, step: 0.05 },
  degrees: { label: '旋转角度', type: 'number', default: 0, step: 0.5 },
  translate: { label: '平移增强', type: 'number', default: 0.1, step: 0.05 },
  scale: { label: '缩放增强', type: 'number', default: 0.5, step: 0.05 },
  shear: { label: '剪切增强', type: 'number', default: 0, step: 0.5 },
  perspective: { label: '透视增强', type: 'number', default: 0, step: 0.0001 },
  flipud: { label: '上下翻转概率', type: 'number', default: 0, step: 0.05 },
  fliplr: { label: '左右翻转概率', type: 'number', default: 0.5, step: 0.05 },
  mosaic: { label: 'Mosaic 概率', type: 'number', default: 1, step: 0.05 },
  mixup: { label: 'MixUp 概率', type: 'number', default: 0, step: 0.05 },
  copy_paste: { label: 'Copy-paste 概率', type: 'number', default: 0, step: 0.05 },
  box: { label: 'Box loss 权重', type: 'number', default: 7.5, step: 0.1 },
  cls: { label: 'Cls loss 权重', type: 'number', default: 0.5, step: 0.1 },
  dfl: { label: 'DFL loss 权重', type: 'number', default: 1.5, step: 0.1 },
}

export function EditParametersModal({ task, open, onClose, onSaved }: Props) {
  const [form, setForm] = useState({ epochs: 100, batch: 16, imgsz: 640, device: '0', workers: 8, optimizer: 'auto', patience: 50, lr0: '', close_mosaic: 10, pretrained: true, cache: false, amp: true })
  const [extras, setExtras] = useState<Record<string, ExtraValue>>({})
  const [customArgs, setCustomArgs] = useState<Record<string, ExtraValue>>({})
  const [newKey, setNewKey] = useState('')
  const [customKey, setCustomKey] = useState('')
  const [customType, setCustomType] = useState<'number' | 'boolean' | 'text'>('number')
  const [customValue, setCustomValue] = useState('')
  const [expanded, setExpanded] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!task || !open) return
    setForm({
      epochs: task.epochs, batch: task.batch, imgsz: task.imgsz, device: task.device, workers: task.workers,
      optimizer: task.optimizer, patience: Number(task.config.patience ?? 50),
      lr0: task.config.lr0 == null ? '' : String(task.config.lr0), close_mosaic: Number(task.config.close_mosaic ?? 10),
      pretrained: task.config.pretrained !== false, cache: Boolean(task.config.cache), amp: task.config.amp !== false,
    })
    setExtras(Object.fromEntries(Object.keys(extraDefinitions).filter((key) => task.config[key] != null).map((key) => [key, task.config[key] as ExtraValue])))
    setCustomArgs((task.config.custom_args as Record<string, ExtraValue> | undefined) || {})
    setError('')
  }, [task, open])

  const availableKeys = useMemo(() => Object.keys(extraDefinitions).filter((key) => !(key in extras)), [extras])
  if (!open || !task) return null
  const locked = ['queued', 'starting', 'running', 'stopping'].includes(task.status)

  function update<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  function addExtra() {
    if (!newKey) return
    setExtras((current) => ({ ...current, [newKey]: extraDefinitions[newKey].default }))
    setNewKey('')
  }

  function updateExtra(key: string, raw: string | boolean) {
    const definition = extraDefinitions[key]
    const value = definition.type === 'number' ? Number(raw) : raw
    setExtras((current) => ({ ...current, [key]: value }))
  }

  function addCustom() {
    const key = customKey.trim()
    if (!key) return
    const value: ExtraValue = customType === 'boolean' ? customValue !== 'false' : customType === 'number' ? Number(customValue || 0) : customValue
    setCustomArgs((current) => ({ ...current, [key]: value }))
    setCustomKey(''); setCustomValue(''); setCustomType('number')
  }

  function updateCustom(key: string, raw: string | boolean) {
    const current = customArgs[key]
    const value = typeof current === 'number' ? Number(raw) : raw
    setCustomArgs((items) => ({ ...items, [key]: value }))
  }

  async function save(event: React.FormEvent) {
    event.preventDefault()
    if (!task) return
    setSaving(true); setError('')
    try {
      const config: Record<string, unknown> = {
        patience: form.patience, pretrained: form.pretrained, cache: form.cache, amp: form.amp,
        close_mosaic: form.close_mosaic, ...extras,
      }
      if (form.lr0 !== '') config.lr0 = Number(form.lr0)
      const updated = await api.updateConfig(task.id, {
        epochs: form.epochs, batch: form.batch, imgsz: form.imgsz, device: form.device,
        workers: form.workers, optimizer: form.optimizer, config, custom_config: customArgs,
      })
      onSaved(updated)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保存失败')
    } finally { setSaving(false) }
  }

  return <div className="modal-backdrop parameter-backdrop" onMouseDown={onClose}>
    <form className="modal parameter-modal" onSubmit={save} onMouseDown={(event) => event.stopPropagation()}>
      <header><div className="modal-heading"><div className="heading-icon"><SlidersHorizontal size={18} /></div><div><span className="eyebrow">TRAINING CONFIGURATION</span><h2>编辑训练参数</h2><p>{task.name}</p></div></div><button type="button" className="icon-button" onClick={onClose}><X size={20} /></button></header>
      <div className="modal-body">
        {locked && <div className="alert warning">任务正在运行或排队，请停止任务后再修改参数。</div>}
        {error && <div className="alert error">{error}</div>}
        <div className="parameter-section"><h3>基础训练参数</h3><p>这些参数将直接控制下一次训练或恢复任务。</p></div>
        <div className="form-grid parameter-grid">
          <label>训练轮数 Epochs<input type="number" min={Math.max(1, task.completed_epochs)} value={form.epochs} onChange={(e) => update('epochs', Number(e.target.value))} /></label>
          <label>批大小 Batch<input type="number" min="1" value={form.batch} onChange={(e) => update('batch', Number(e.target.value))} /></label>
          <label>输入尺寸 Image size<input type="number" min="32" step="32" value={form.imgsz} onChange={(e) => update('imgsz', Number(e.target.value))} /></label>
          <label>GPU 设备 Device<input value={form.device} onChange={(e) => update('device', e.target.value)} placeholder="例如 0,1 或 cpu" /></label>
          <label>数据线程 Workers<input type="number" min="0" value={form.workers} onChange={(e) => update('workers', Number(e.target.value))} /></label>
          <label>优化器 Optimizer<select value={form.optimizer} onChange={(e) => update('optimizer', e.target.value)}><option>auto</option><option>SGD</option><option>Adam</option><option>AdamW</option><option>MuSGD</option><option>RMSProp</option></select></label>
          <label>早停耐心值 Patience<input type="number" min="0" value={form.patience} onChange={(e) => update('patience', Number(e.target.value))} /></label>
          <label>初始学习率 LR0<input type="number" min="0" max="1" step="0.00001" value={form.lr0} placeholder="自动" onChange={(e) => update('lr0', e.target.value)} /></label>
          <label>关闭 Mosaic 轮数<input type="number" min="0" value={form.close_mosaic} onChange={(e) => update('close_mosaic', Number(e.target.value))} /></label>
          <div className="toggle-cluster"><label><input type="checkbox" checked={form.pretrained} onChange={(e) => update('pretrained', e.target.checked)} /><span>预训练权重</span></label><label><input type="checkbox" checked={form.amp} onChange={(e) => update('amp', e.target.checked)} /><span>AMP</span></label><label><input type="checkbox" checked={form.cache} onChange={(e) => update('cache', e.target.checked)} /><span>缓存数据</span></label></div>
        </div>
        <button type="button" className="advanced-toggle parameter-toggle" onClick={() => setExpanded((value) => !value)}><ChevronDown size={16} className={expanded ? 'rotated' : ''} />高级参数 <em>{Object.keys(extras).length} 项</em></button>
        {expanded && <div className="extra-parameters">
          {Object.entries(extras).map(([key, value]) => { const definition = extraDefinitions[key]; return <div className="extra-row" key={key}><div><span>{definition.label}</span><code>{key}</code></div>{definition.type === 'boolean' ? <select value={String(value)} onChange={(e) => updateExtra(key, e.target.value === 'true')}><option value="true">开启</option><option value="false">关闭</option></select> : <input type={definition.type} step={definition.step} value={String(value)} onChange={(e) => updateExtra(key, e.target.value)} />}<button type="button" title="删除参数" onClick={() => setExtras((current) => { const next = { ...current }; delete next[key]; return next })}><Trash2 size={15} /></button></div> })}
          <div className="add-parameter"><select value={newKey} onChange={(e) => setNewKey(e.target.value)}><option value="">选择要添加的训练参数</option>{availableKeys.map((key) => <option key={key} value={key}>{extraDefinitions[key].label} · {key}</option>)}</select><button type="button" className="button ghost" disabled={!newKey} onClick={addExtra}><Plus size={15} />添加参数</button></div>
          <div className="custom-parameter-section">
            <div className="custom-title"><Braces size={15} /><div><strong>自定义 YOLO 参数</strong><span>用于添加预设列表中没有的合法参数</span></div></div>
            {Object.entries(customArgs).map(([key, value]) => <div className="extra-row custom-row" key={key}><div><span>自定义参数</span><code>{key}</code></div>{typeof value === 'boolean' ? <select value={String(value)} onChange={(e) => updateCustom(key, e.target.value === 'true')}><option value="true">开启</option><option value="false">关闭</option></select> : <input type={typeof value === 'number' ? 'number' : 'text'} value={String(value)} onChange={(e) => updateCustom(key, e.target.value)} />}<button type="button" title="删除参数" onClick={() => setCustomArgs((current) => { const next = { ...current }; delete next[key]; return next })}><Trash2 size={15} /></button></div>)}
            <div className="custom-add-row"><input value={customKey} onChange={(e) => setCustomKey(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))} placeholder="参数名，例如 save_json" /><select value={customType} onChange={(e) => { const type = e.target.value as typeof customType; setCustomType(type); setCustomValue(type === 'boolean' ? 'true' : '') }}><option value="number">数字</option><option value="boolean">布尔</option><option value="text">文本</option></select>{customType === 'boolean' ? <select value={customValue || 'true'} onChange={(e) => setCustomValue(e.target.value)}><option value="true">开启</option><option value="false">关闭</option></select> : <input type={customType === 'number' ? 'number' : 'text'} value={customValue} onChange={(e) => setCustomValue(e.target.value)} placeholder="参数值" />}<button type="button" className="button ghost" disabled={!customKey || customKey in customArgs} onClick={addCustom}><Plus size={15} />自定义添加</button></div>
            <p className="custom-help">参数名只能包含小写字母、数字和下划线；模型、数据集、设备等平台关键参数不能覆盖。</p>
          </div>
        </div>}
      </div>
      <footer><span className="parameter-note">双击配置节点可随时重新打开</span><button type="button" className="button ghost" onClick={onClose}>取消</button><button className="button primary" disabled={saving || locked}>{saving && <LoaderCircle size={16} className="spin" />}保存参数</button></footer>
    </form>
  </div>
}
