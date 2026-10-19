import { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, AlertCircle, Boxes, BrainCircuit, ChevronRight, Image as ImageIcon, Layers3, RefreshCw, Sigma, Sparkles, X, Zap } from 'lucide-react'
import type { TrainingRun, TrainingTask, TrainingTrace, TrainingTraceSample } from '../types'
import { api } from '../api'
import { useModalFocus } from '../useModalFocus'

type Props = { task: TrainingTask; run: TrainingRun | null; open: boolean; onClose: () => void }
type Step = 'input' | 'network' | 'loss' | 'update'

const steps: Array<{ id: Step; title: string; subtitle: string; icon: typeof ImageIcon }> = [
  { id: 'input', title: '数据输入', subtitle: '图片、标签与张量', icon: ImageIcon },
  { id: 'network', title: '网络前向', subtitle: '层输出统计', icon: Layers3 },
  { id: 'loss', title: '损失计算', subtitle: '模型错在哪里', icon: Sigma },
  { id: 'update', title: '参数更新', subtitle: '梯度如何改变权重', icon: Zap },
]

function number(value: number | undefined, digits = 5) { return typeof value === 'number' ? value.toFixed(digits) : '—' }
function shape(value: { shape?: number[] } | null | undefined) { return value?.shape?.length ? `[${value.shape.join(' × ')}]` : '—' }

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return <div className="trace-stat"><span>{label}</span><strong>{value}</strong></div>
}

function InputStep({ sample }: { sample: TrainingTraceSample }) {
  return <div className="trace-grid trace-input-grid">
    <section className="trace-panel trace-image-panel">
      <div className="trace-panel-title"><ImageIcon size={15} /><strong>真实训练输入</strong><span>采样批次 {sample.batch}</span></div>
      {sample.image_url ? <img src={sample.image_url} alt="训练批次输入" /> : <div className="trace-empty"><ImageIcon size={24} /><span>当前只保存了张量摘要</span><small>没有把完整图片写入过程记录</small></div>}
    </section>
    <section className="trace-panel">
      <div className="trace-panel-title"><Boxes size={15} /><strong>输入张量</strong></div>
      <div className="trace-stat-grid"><Stat label="形状" value={shape(sample.images)} /><Stat label="元素数量" value={sample.images?.numel?.toLocaleString() || '—'} /><Stat label="均值" value={number(sample.images?.mean)} /><Stat label="标准差" value={number(sample.images?.std)} /><Stat label="最小值" value={number(sample.images?.min)} /><Stat label="最大值" value={number(sample.images?.max)} /></div>
      <p className="trace-explain">图片会先转换为模型可计算的浮点张量，再进入网络。这里显示的是本次真实采样的形状和数值范围，不是示例数据。</p>
    </section>
  </div>
}

function NetworkStep({ sample }: { sample: TrainingTraceSample }) {
  const layers = Object.entries(sample.layers)
  return <section className="trace-panel trace-wide-panel">
    <div className="trace-panel-title"><Layers3 size={15} /><strong>网络层输出摘要</strong><span>{layers.length} 个采样层</span></div>
    {layers.length ? <div className="trace-layer-list">{layers.map(([name, stats]) => <article key={name}><div className="trace-layer-icon"><BrainCircuit size={14} /></div><div><strong>{name}</strong><span>{shape(stats)} · 均值 {number(stats.mean)} · 标准差 {number(stats.std)}</span></div><div className="trace-bar"><i style={{ width: `${Math.min(100, Math.max(4, Math.abs(stats.mean || 0) * 100))}%` }} /></div><code>{number(stats.min)} ~ {number(stats.max)}</code></article>)}</div> : <div className="trace-empty"><Layers3 size={24} /><span>尚未记录网络层输出</span><small>下一次采样后刷新查看</small></div>}
    <p className="trace-explain">网络前向会依次经过骨干、特征融合和检测头。这里保存每个采样层的张量形状和统计量，帮助观察数值是否异常；它不是完整特征图。</p>
  </section>
}

function LossStep({ sample }: { sample: TrainingTraceSample }) {
  const losses = Object.entries(sample.loss)
  const max = Math.max(...losses.map(([, value]) => Math.abs(value)), 1e-9)
  return <section className="trace-panel trace-wide-panel">
    <div className="trace-panel-title"><Sigma size={15} /><strong>本批次损失</strong><span>数值越高表示当前误差贡献越大</span></div>
    {losses.length ? <div className="trace-loss-list">{losses.map(([name, value]) => <div key={name}><span>{name}</span><div><i style={{ width: `${Math.max(3, Math.abs(value) / max * 100)}%` }} /></div><strong>{number(value, 7)}</strong></div>)}</div> : <div className="trace-empty"><Sigma size={24} /><span>尚未记录损失</span></div>}
    <p className="trace-explain">检测任务通常把定位、分类和分布焦点等损失组合后反向传播。实际损失项以当前模型和训练配置返回的结果为准。</p>
  </section>
}

function UpdateStep({ sample }: { sample: TrainingTraceSample }) {
  const updates = Object.entries(sample.updates)
  return <section className="trace-panel trace-wide-panel">
    <div className="trace-panel-title"><Zap size={15} /><strong>梯度与权重更新</strong><span>前 12 个优化器参数组的摘要</span></div>
    {updates.length ? <div className="trace-update-table"><div className="trace-update-head"><span>参数索引</span><span>梯度均值</span><span>梯度峰值</span><span>权重变化</span></div>{updates.map(([name, update]) => <div className="trace-update-row" key={name}><code>#{name}</code><span>{number(update.gradient_mean, 8)}</span><span>{number(update.gradient_max, 8)}</span><strong>{number(update.weight_delta_mean, 8)}</strong></div>)}</div> : <div className="trace-empty"><Zap size={24} /><span>尚未记录优化器更新</span><small>采样应落在一次优化器 step 之后</small></div>}
    <p className="trace-explain">梯度来自当前批次损失。优化器根据梯度、学习率和自身状态修改权重；这里只保存摘要，不会把整套模型参数复制到数据库。</p>
  </section>
}

export function TrainingTraceModal({ task, run, open, onClose }: Props) {
  const [trace, setTrace] = useState<TrainingTrace | null>(null)
  const [sampleIndex, setSampleIndex] = useState(0)
  const [step, setStep] = useState<Step>('input')
  const [loading, setLoading] = useState(false)
  const runId = run?.id
  const dialogRef = useModalFocus<HTMLElement>(open, onClose, {
    initialFocusSelector: '[data-modal-initial-focus]',
  })

  const load = useCallback(async () => {
    if (!runId) return
    setLoading(true)
    try { setTrace(await api.trainingTrace(task.id, runId)) } finally { setLoading(false) }
  }, [runId, task.id])

  useEffect(() => { if (open) { setSampleIndex(0); setStep('input'); load() } }, [open, load])
  const samples = trace?.samples || []
  const sample = samples[sampleIndex] || null
  const currentStep = useMemo(() => steps.find((item) => item.id === step) || steps[0], [step])
  if (!open) return null

  return <div className="modal-backdrop trace-backdrop" onMouseDown={onClose}>
    <section ref={dialogRef} className="trace-modal" role="dialog" aria-modal="true" aria-labelledby="training-trace-title" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
      <header className="trace-header"><div className="modal-heading"><div className="heading-icon trace-heading-icon"><Activity size={19} /></div><div><span className="eyebrow">TRAINING TRACE · REAL BATCH</span><h2 id="training-trace-title">YOLO 训练过程解析</h2><p>{task.name} · {run ? `Attempt ${String(run.attempt_number).padStart(3, '0')}` : '未选择运行'} · 真实采样回放</p></div></div><div className="trace-header-actions"><button className="button ghost compact" onClick={load} disabled={loading}><RefreshCw size={14} className={loading ? 'spin' : ''} />刷新采样</button><button className="icon-button" data-modal-initial-focus title="关闭" aria-label="关闭训练解析" onClick={onClose}><X size={20} /></button></div></header>
      {!trace?.available ? <div className="trace-unavailable"><AlertCircle size={25} /><strong>{loading ? '正在读取训练过程…' : '当前运行没有可用的过程采样'}</strong><p>{trace?.reason || '请启动新的本机目标检测训练，并在训练结束或运行中重新打开此卡片。'}</p></div> : <>
        <div className="trace-toolbar"><label><span>采样批次</span><select value={sampleIndex} onChange={(event) => setSampleIndex(Number(event.target.value))}>{samples.map((item, index) => <option key={`${item.epoch}-${item.batch}`} value={index}>Epoch {item.epoch} · Batch {item.batch}</option>)}</select></label><div className="trace-summary"><Stat label="已采样" value={`${trace.sample_count || samples.length} / ${trace.maximum || '—'}`} /><Stat label="间隔" value={`每 ${trace.interval || '—'} 批`} /><Stat label="当前步骤" value={currentStep.title} /></div></div>
        <div className="trace-main"><nav className="trace-steps">{steps.map(({ id, title, subtitle, icon: Icon }, index) => <button key={id} className={step === id ? 'active' : ''} onClick={() => setStep(id)}><span className="trace-step-index">{index + 1}</span><Icon size={16} /><span><strong>{title}</strong><small>{subtitle}</small></span><ChevronRight size={14} /></button>)}</nav><div className="trace-content">{sample && step === 'input' && <InputStep sample={sample} />}{sample && step === 'network' && <NetworkStep sample={sample} />}{sample && step === 'loss' && <LossStep sample={sample} />}{sample && step === 'update' && <UpdateStep sample={sample} />}</div></div>
      </>}
      <footer className="trace-footer"><span><Sparkles size={13} />仅展示训练进程实际采集的数据；没有采集的张量不会用模拟数值填充</span><button className="button ghost" onClick={onClose}>关闭</button></footer>
    </section>
  </div>
}
