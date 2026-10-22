/* eslint-disable react-refresh/only-export-components -- Shared UI helpers are consumed by both model-test views. */
import {
  AlertCircle, Box, CheckCircle2, Clock3, Cpu, FileArchive, Gauge, Images, Info,
  LoaderCircle, Maximize2, ScanEye, Target, X,
} from 'lucide-react'
import { formatBytes } from '../api'
import type { Artifact, ModelTest, ModelTestSummary, YoloTaskType } from '../types'
import '../inference.css'

export interface SelectedImage {
  id: number
  file: File
  previewUrl: string
}

export const modelTestStatusLabels: Record<ModelTest['status'], string> = {
  uploading: '准备图片', preparing: '准备模型', queued: '等待测试', running: '正在测试',
  cancel_requested: '正在取消', completed: '测试完成', failed: '测试失败', cancelled: '已取消',
}

export const activeModelTestStatuses = new Set<ModelTest['status']>([
  'uploading', 'preparing', 'queued', 'running', 'cancel_requested',
])

export function modelTestDetectionLabel(count: number | null) {
  return count === null ? '结果未就绪' : count > 0 ? `检测到 ${count} 个目标` : '未检测到目标'
}

export const modelTestMetrics = {
  detect: { key: 'metrics/mAP50-95(B)', label: 'mAP50-95' },
  segment: { key: 'metrics/mAP50-95(M)', label: 'Mask mAP50-95' },
  classify: { key: 'metrics/accuracy_top1', label: 'Top-1 accuracy' },
  pose: { key: 'metrics/mAP50-95(P)', label: 'Pose mAP50-95' },
  obb: { key: 'metrics/mAP50-95(B)', label: 'OBB mAP50-95' },
} as const

export function isSupportedModelTestTask(task: string | null | undefined): task is YoloTaskType {
  return !!task && Object.prototype.hasOwnProperty.call(modelTestMetrics, task)
}

export function modelTestTaskType(test?: Pick<ModelTestSummary, 'task_type' | 'model'> | null): YoloTaskType | null {
  const task = test?.task_type || test?.model.task_type
  if (isSupportedModelTestTask(task)) return task
  // Older snapshots predate task identification and only supported detection.
  return test && !Object.prototype.hasOwnProperty.call(test.model, 'task_type') ? 'detect' : null
}

export function modelTestTaskLabel(task: string | null | undefined) {
  return isSupportedModelTestTask(task)
    ? ({ detect: '目标检测', segment: '实例分割', classify: '图像分类', pose: '姿态估计', obb: '旋转框检测' } as const)[task]
    : '任务类型待识别'
}

export function modelTestResultLabel(task: string | null | undefined, count: number | null) {
  if (count === null) return '结果未就绪'
  const itemLabel = isSupportedModelTestTask(task)
    ? ({ detect: '目标', segment: '分割实例', classify: '候选类别', pose: '个体', obb: '旋转目标' } as const)[task]
    : '预测结果'
  return count > 0 ? `结果 ${count} 个${itemLabel}` : '没有预测结果'
}

export function modelTestParameterText(test: Pick<ModelTestSummary, 'task_type' | 'model' | 'confidence' | 'iou'>) {
  return `Conf ${test.confidence.toFixed(2)} · ${modelTestTaskType(test) === 'classify' ? 'IoU 不适用' : `IoU ${test.iou.toFixed(2)}`}`
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

export function ModelStrip({ test, artifact, fallbackAttempt, fallbackTask, bestMetric, history = false }: {
  test?: ModelTestSummary | null
  artifact?: Artifact | null
  fallbackAttempt?: number | null
  fallbackTask?: string | null
  bestMetric?: { epoch: number; value: number } | null
  history?: boolean
}) {
  const attempt = test?.source_attempt_number ?? test?.model.attempt_number ?? fallbackAttempt
  const metric = typeof test?.model.best_epoch === 'number' && typeof test.model.metric_value === 'number' && Number.isFinite(test.model.metric_value)
    ? { epoch: test.model.best_epoch, value: test.model.metric_value }
    : test ? null : bestMetric
  const taskType = test ? modelTestTaskType(test) : fallbackTask
  const metricName = test?.model.metric_name || (isSupportedModelTestTask(taskType) ? modelTestMetrics[taskType].label : '训练指标')
  const size = test?.model_size ?? test?.model.weight_size ?? artifact?.size
  const name = test ? modelTestWeightName(test) : artifact?.name.split('/').pop() || '当前权重不可用'
  return <section className={`model-test-model-strip${history ? ' history' : ''}`} aria-label="当前测试模型">
    <div><Box size={17} /><span>测试权重<strong>{name}{typeof size === 'number' ? ` · ${formatBytes(size)}` : ''}</strong></span></div>
    <div><FileArchive size={17} /><span>权重来源<strong>{attempt ? `Attempt ${String(attempt).padStart(3, '0')}` : '当前选中运行'}</strong></span></div>
    <div><Gauge size={17} /><span>{metricName}<strong>{metric ? `Epoch ${metric.epoch} · ${metric.value.toFixed(4)}` : '暂无指标记录'}</strong></span></div>
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
  const taskType = modelTestTaskType(test)
  const taskLabel = modelTestTaskLabel(taskType)
  const selectedResult = rows[current]
  const objects = completed ? selectedResult.objects : []
  const allObjects = completed ? rows.flatMap((file) => file.objects) : []
  const resultImages = completed ? rows.filter((file) => (file.prediction_count ?? file.detections) !== null && (file.prediction_count ?? file.detections)! > 0).length : 0
  const emptyImages = completed ? rows.filter((file) => (file.prediction_count ?? file.detections) === 0).length : 0
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
            <small className={completed && (file.prediction_count ?? file.detections ?? 0) > 0 ? 'has-detections' : ''}>{completed ? modelTestResultLabel(taskType, file.prediction_count ?? file.detections) : formatBytes(file.size)}</small>
          </button>
          {onRemove && <button className="model-test-remove" disabled={busy} aria-label={`移除 ${file.name}`} title="移除图片" onClick={() => onRemove(index)}><X size={14} /></button>}
        </div>
      })}
    </aside>
    <div className="model-test-main">
      {completed && <div className="model-test-summary">共 {rows.length} 张 {taskLabel}图片：{resultImages} 张有结果，{emptyImages} 张没有结果。</div>}
      <div className="model-test-comparison">
        {[{ url: original, title: '原图', kind: 'original' }, { url: result, title: taskType ? `${taskLabel}结果` : '预测结果', kind: 'result' }].map(({ url, title, kind }) => <section className="model-test-panel" key={title}>
          <div><strong>{title}</strong>{url && <button className="icon-button" title={`放大${title}`} aria-label={`放大${title}`} onClick={(event) => openLightbox(url, title, event.currentTarget)}><Maximize2 size={15} /></button>}</div>
          {kind === 'result' && completed && <div className={`model-test-outcome ${(selectedResult.prediction_count ?? selectedResult.detections ?? 0) > 0 ? 'detected' : 'not-detected'}`} role="status">
            {(selectedResult.prediction_count ?? selectedResult.detections ?? 0) > 0 ? <CheckCircle2 size={19} /> : <Info size={19} />}
            <div><strong>{(selectedResult.prediction_count ?? selectedResult.detections ?? 0) > 0 ? modelTestResultLabel(taskType, selectedResult.prediction_count ?? selectedResult.detections) : `当前阈值下未获得${taskLabel}预测结果`}</strong>{(selectedResult.prediction_count ?? selectedResult.detections) === 0 && <p>可适当调整阈值后重新测试；这不能单独作为图片中不存在该类内容的依据。</p>}</div>
          </div>}
          {url ? <button className="model-test-image" onDoubleClick={(event) => openLightbox(url, title, event.currentTarget)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openLightbox(url, title, event.currentTarget) } }} aria-label={`${title}，双击放大`}><img src={url} alt={`${selectedResult.name} ${title}`} /></button>
            : <div className="model-test-placeholder">{busy ? <LoaderCircle className="spin" size={28} /> : failed ? <AlertCircle size={32} /> : <ScanEye size={32} />}<span>{busy ? '模型结果生成中…' : failed ? '模型测试失败，请查看错误原因' : '运行测试后，查看模型预测结果'}</span></div>}
        </section>)}
      </div>
      {completed && <section className="model-test-insights" aria-label="测试统计">
        <div className="model-test-stat-grid">
          <div><Target size={17} /><span>{taskType === 'classify' ? '全部候选类别' : '全部预测结果'}<strong>{allObjects.length}</strong></span></div>
          <div><Images size={17} /><span>有结果图片<strong>{resultImages}/{rows.length}</strong></span></div>
          <div><Gauge size={17} /><span>平均置信度<strong>{averageConfidence === null ? '—' : averageConfidence.toFixed(3)}</strong></span></div>
          <div><Clock3 size={17} /><span>平均推理<strong>{milliseconds(test.timings ? test.timings.inference / rows.length : undefined)}</strong></span></div>
        </div>
        <div className="model-test-detail-grid">
          <div><h3>当前图片预测 <span>{objects.length}</span></h3>{objects.length ? <div className="model-test-object-list">{objects.map((item, index) => <div key={`${item.class_id ?? item.class_name}-${index}`}><span>#{index + 1}</span><strong>{item.class_name}{taskType === 'segment' && item.mask_available ? ' · 含掩码' : ''}{taskType === 'pose' && item.keypoints ? ` · ${item.keypoints.length} 个关键点` : ''}{taskType === 'obb' && item.obb_points ? ' · 旋转框' : ''}</strong><em>{item.confidence.toFixed(3)}</em></div>)}</div> : <p>当前图片没有可显示的预测结果。</p>}</div>
          <div><h3>整批类别统计 <span>{classRows.length}</span></h3>{classRows.length ? <div className="model-test-class-list">{classRows.map(([name, value]) => <div key={name}><strong>{name}</strong><span><i style={{ width: `${Math.max(8, value.count / allObjects.length * 100)}%` }} /></span><em>{value.count} · 平均 {(value.confidence / value.count).toFixed(3)}</em></div>)}</div> : <p>本次没有预测结果。</p>}</div>
          <div><h3>性能数据</h3><dl><div><dt>模型加载</dt><dd>{milliseconds(test.timings?.model_load)}</dd></div><div><dt>预处理</dt><dd>{milliseconds(selectedResult.timings.preprocess)}</dd></div><div><dt>当前图推理</dt><dd>{milliseconds(selectedResult.timings.inference)}</dd></div><div><dt>后处理</dt><dd>{milliseconds(selectedResult.timings.postprocess)}</dd></div><div><dt>整批总耗时</dt><dd>{milliseconds(test.timings?.total)}</dd></div></dl></div>
        </div>
      </section>}
    </div>
  </div>
}
