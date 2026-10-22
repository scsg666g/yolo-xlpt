/* eslint-disable react-refresh/only-export-components -- Shared UI helpers are consumed by both model-test views. */
import {
  AlertCircle, Box, CheckCircle2, Clock3, Cpu, FileArchive, Gauge, Images, Info,
  LoaderCircle, Maximize2, ScanEye, Target, X,
} from 'lucide-react'
import { formatBytes } from '../api'
import type { Artifact, ModelTest, ModelTestSummary } from '../types'
import '../inference.css'

export interface SelectedImage {
  id: number
  file: File
  previewUrl: string
}

export const modelTestStatusLabels: Record<ModelTest['status'], string> = {
  uploading: '准备图片', preparing: '准备模型', queued: '等待测试', running: '正在检测',
  cancel_requested: '正在取消', completed: '检测完成', failed: '检测失败', cancelled: '已取消',
}

export const activeModelTestStatuses = new Set<ModelTest['status']>([
  'uploading', 'preparing', 'queued', 'running', 'cancel_requested',
])

export function modelTestDetectionLabel(count: number | null) {
  return count === null ? '结果未就绪' : count > 0 ? `检测到 ${count} 个目标` : '未检测到目标'
}

function milliseconds(value: number | undefined) {
  if (value === undefined || !Number.isFinite(value)) return '—'
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(1)} ms`
}

export function modelTestDate(value?: string | null) {
  if (!value) return '时间未记录'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '时间未记录' : date.toLocaleString('zh-CN', { hour12: false })
}

export function modelTestAttemptText(test: ModelTestSummary) {
  const context = test.attempt_number
  const source = test.source_attempt_number ?? test.model.attempt_number
  if (context && source && context !== source) {
    return `Attempt ${String(context).padStart(3, '0')} 发起 · 权重来自 ${String(source).padStart(3, '0')}`
  }
  const attempt = context ?? source
  return attempt ? `Attempt ${String(attempt).padStart(3, '0')}` : '未记录 Attempt'
}

export function modelTestWeightName(test: ModelTestSummary) {
  return test.artifact_name || test.model_name || test.model.weight_name || 'best.pt'
}

export function hasExactModelVersion(test: Pick<ModelTest, 'artifact_id' | 'model_sha256' | 'model_size'>) {
  return typeof test.artifact_id === 'number'
    && typeof test.model_sha256 === 'string'
    && /^[0-9a-f]{64}$/.test(test.model_sha256)
    && typeof test.model_size === 'number'
    && Number.isSafeInteger(test.model_size)
    && test.model_size >= 0
}

export function ModelStrip({ test, artifact, fallbackAttempt, bestMetric, history = false }: {
  test?: ModelTestSummary | null
  artifact?: Artifact | null
  fallbackAttempt?: number | null
  bestMetric?: { epoch: number; value: number } | null
  history?: boolean
}) {
  const attempt = test?.source_attempt_number ?? test?.model.attempt_number ?? fallbackAttempt
  const metric = test?.model.best_epoch && test.model.metric_value !== null
    ? { epoch: test.model.best_epoch, value: test.model.metric_value }
    : bestMetric
  const size = test?.model_size ?? test?.model.weight_size ?? artifact?.size
  const name = test ? modelTestWeightName(test) : artifact?.name.split('/').pop() || '当前权重不可用'
  return <section className={`model-test-model-strip${history ? ' history' : ''}`} aria-label="当前测试模型">
    <div><Box size={17} /><span>测试权重<strong>{name}{typeof size === 'number' ? ` · ${formatBytes(size)}` : ''}</strong></span></div>
    <div><FileArchive size={17} /><span>权重来源<strong>{attempt ? `Attempt ${String(attempt).padStart(3, '0')}` : '当前选中运行'}</strong></span></div>
    <div><Gauge size={17} /><span>最高 mAP50-95<strong>{metric ? `Epoch ${metric.epoch} · ${metric.value.toFixed(4)}` : '暂无指标记录'}</strong></span></div>
    <div><Cpu size={17} /><span>执行环境<strong>{test?.model.device || 'CPU'} · {test?.model.imgsz || 640} px</strong></span></div>
  </section>
}

export function TestWorkspace({ test, images, selected, onSelect, onRemove, busy, failed, openLightbox }: {
  test: ModelTest | null
  images?: SelectedImage[]
  selected: number
  onSelect: (index: number) => void
  onRemove?: (index: number) => void
  busy: boolean
  failed: boolean
  openLightbox: (url: string, title: string, opener: HTMLElement) => void
}) {
  const localRows: ModelTest['files'] = (images || []).map(({ file }) => ({
    name: file.name, size: file.size, uploaded: false, detections: null, objects: [], timings: {},
  }))
  const rows = test?.files.length ? test.files : localRows
  if (!rows.length) return null
  const current = Math.max(0, Math.min(selected, rows.length - 1))
  const completed = test?.status === 'completed'
  const selectedResult = rows[current]
  const objects = completed ? selectedResult.objects : []
  const allObjects = completed ? rows.flatMap((file) => file.objects) : []
  const detectedImages = completed ? rows.filter((file) => file.detections !== null && file.detections > 0).length : 0
  const emptyImages = completed ? rows.filter((file) => file.detections === 0).length : 0
  const classCounts = allObjects.reduce<Record<string, { count: number; confidence: number }>>((counts, item) => {
    const previous = counts[item.class_name] || { count: 0, confidence: 0 }
    counts[item.class_name] = { count: previous.count + 1, confidence: previous.confidence + item.confidence }
    return counts
  }, {})
  const classRows = Object.entries(classCounts).sort((left, right) => right[1].count - left[1].count)
  const averageConfidence = allObjects.length ? allObjects.reduce((sum, item) => sum + item.confidence, 0) / allObjects.length : null
  const imageUrl = (index: number, kind: 'original' | 'result') => {
    if (test && (kind === 'result' ? completed : test.files[index]?.uploaded)) return `/api/model-tests/${test.id}/images/${index}/${kind}`
    return kind === 'original' ? images?.[index]?.previewUrl || '' : ''
  }
  const original = imageUrl(current, 'original')
  const result = imageUrl(current, 'result')

  return <div className="model-test-workspace">
    <aside className="model-test-thumbnails">
      {rows.map((file, index) => {
        const thumbnail = imageUrl(index, 'original')
        return <div className="model-test-thumbnail" key={`${file.name}-${index}`}>
          <button className={`model-test-thumbnail-select ${current === index ? 'selected' : ''}`} onClick={() => onSelect(index)}>
            {thumbnail ? <img src={thumbnail} alt={file.name} /> : <span className="model-test-thumbnail-placeholder"><Images size={18} /></span>}
            <span title={file.name}>{file.name}</span>
            <small className={completed && (file.detections || 0) > 0 ? 'has-detections' : ''}>{completed ? modelTestDetectionLabel(file.detections) : formatBytes(file.size)}</small>
          </button>
          {onRemove && <button className="model-test-remove" disabled={busy} aria-label={`移除 ${file.name}`} title="移除图片" onClick={() => onRemove(index)}><X size={14} /></button>}
        </div>
      })}
    </aside>
    <div className="model-test-main">
      {completed && <div className="model-test-summary">共 {rows.length} 张：{detectedImages} 张检测到目标，{emptyImages} 张未检测到。</div>}
      <div className="model-test-comparison">
        {[{ url: original, title: '原图' }, { url: result, title: '检测结果' }].map(({ url, title }) => <section className="model-test-panel" key={title}>
          <div><strong>{title}</strong>{url && <button className="icon-button" title={`放大${title}`} aria-label={`放大${title}`} onClick={(event) => openLightbox(url, title, event.currentTarget)}><Maximize2 size={15} /></button>}</div>
          {title === '检测结果' && completed && <div className={`model-test-outcome ${(selectedResult.detections || 0) > 0 ? 'detected' : 'not-detected'}`} role="status">
            {(selectedResult.detections || 0) > 0 ? <CheckCircle2 size={19} /> : <Info size={19} />}
            <div><strong>{(selectedResult.detections || 0) > 0 ? modelTestDetectionLabel(selectedResult.detections) : `当前置信度阈值 ${test.confidence.toFixed(2)} 下未检测到目标`}</strong>{selectedResult.detections === 0 && <p>可适当降低阈值后重新检测。这不代表图片中一定没有目标。</p>}</div>
          </div>}
          {url ? <button className="model-test-image" onDoubleClick={(event) => openLightbox(url, title, event.currentTarget)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openLightbox(url, title, event.currentTarget) } }} aria-label={`${title}，双击放大`}><img src={url} alt={`${selectedResult.name} ${title}`} /></button>
            : <div className="model-test-placeholder">{busy ? <LoaderCircle className="spin" size={28} /> : failed ? <AlertCircle size={32} /> : <ScanEye size={32} />}<span>{busy ? '检测结果生成中…' : failed ? '检测失败，请查看错误原因' : '运行检测后，查看目标框与置信度'}</span></div>}
        </section>)}
      </div>
      {completed && <section className="model-test-insights" aria-label="检测统计">
        <div className="model-test-stat-grid">
          <div><Target size={17} /><span>全部目标<strong>{allObjects.length}</strong></span></div>
          <div><Images size={17} /><span>命中图片<strong>{detectedImages}/{rows.length}</strong></span></div>
          <div><Gauge size={17} /><span>平均置信度<strong>{averageConfidence === null ? '—' : averageConfidence.toFixed(3)}</strong></span></div>
          <div><Clock3 size={17} /><span>平均推理<strong>{milliseconds(test.timings ? test.timings.inference / rows.length : undefined)}</strong></span></div>
        </div>
        <div className="model-test-detail-grid">
          <div><h3>当前图片目标 <span>{objects.length}</span></h3>{objects.length ? <div className="model-test-object-list">{objects.map((item, index) => <div key={`${item.class_id}-${index}`}><span>#{index + 1}</span><strong>{item.class_name}</strong><em>{item.confidence.toFixed(3)}</em></div>)}</div> : <p>当前图片没有达到阈值的目标。</p>}</div>
          <div><h3>整批类别统计 <span>{classRows.length}</span></h3>{classRows.length ? <div className="model-test-class-list">{classRows.map(([name, value]) => <div key={name}><strong>{name}</strong><span><i style={{ width: `${Math.max(8, value.count / allObjects.length * 100)}%` }} /></span><em>{value.count} · 平均 {(value.confidence / value.count).toFixed(3)}</em></div>)}</div> : <p>本次没有检测到目标。</p>}</div>
          <div><h3>性能数据</h3><dl><div><dt>模型加载</dt><dd>{milliseconds(test.timings?.model_load)}</dd></div><div><dt>预处理</dt><dd>{milliseconds(selectedResult.timings.preprocess)}</dd></div><div><dt>当前图推理</dt><dd>{milliseconds(selectedResult.timings.inference)}</dd></div><div><dt>后处理</dt><dd>{milliseconds(selectedResult.timings.postprocess)}</dd></div><div><dt>整批总耗时</dt><dd>{milliseconds(test.timings?.total)}</dd></div></dl></div>
        </div>
      </section>}
    </div>
  </div>
}
