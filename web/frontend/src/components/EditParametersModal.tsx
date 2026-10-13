/* eslint-disable react-refresh/only-export-components -- parameter helpers are shared with import dialogs */
import { useEffect, useMemo, useRef, useState } from 'react'
import { Braces, Check, ChevronDown, Clipboard, Code2, LoaderCircle, Plus, Search, SlidersHorizontal, Trash2, X } from 'lucide-react'
import { api } from '../api'
import type { TrainingParameterDefinition, TrainingTask } from '../types'

interface Props {
  task: TrainingTask | null
  open: boolean
  onClose: () => void
  onSaved: (task: TrainingTask) => void
}

type ExtraValue = string | number | boolean
type ImportValue = ExtraValue
type ParameterDefinition = { label: string; type: 'number' | 'boolean' | 'text'; default?: ExtraValue; step?: number }

export interface DetectedParameter {
  key: string
  label: string
  value: ImportValue
  type: 'number' | 'boolean' | 'text'
}

export type ImportDiagnosticStatus = 'supported' | 'context' | 'deprecated' | 'unknown' | 'invalid'

export interface ImportDiagnostic {
  key: string
  label: string
  rawValue: string
  line: number
  occurrences: number
  status: ImportDiagnosticStatus
  message: string
}

export interface ImportAnalysis {
  parameters: DetectedParameter[]
  diagnostics: ImportDiagnostic[]
}

const extraDefinitions: Record<string, ParameterDefinition & { default: ExtraValue }> = {
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
  save: { label: '保存训练检查点', type: 'boolean', default: true },
  cls_remap: { label: '按类别名重映射分类头', type: 'boolean', default: true },
  verbose: { label: '输出详细训练日志', type: 'boolean', default: true },
  single_cls: { label: '所有类别合并为单类', type: 'boolean', default: false },
  classes: { label: '仅训练指定类别 ID', type: 'text', default: '0' },
  profile: { label: '分析 ONNX/TensorRT 性能', type: 'boolean', default: false },
  compile: { label: '启用 torch.compile', type: 'boolean', default: false },
  overlap_mask: { label: '合并实例掩码（分割）', type: 'boolean', default: true },
  mask_ratio: { label: '掩码下采样比率（分割）', type: 'number', default: 4 },
  dropout: { label: 'Dropout 比率（分类）', type: 'number', default: 0, step: 0.05 },
  val: { label: '训练期间运行验证', type: 'boolean', default: true },
  split: { label: '验证数据集划分', type: 'text', default: 'val' },
  save_json: { label: '保存 COCO JSON 结果', type: 'boolean', default: false },
  conf: { label: '验证置信度阈值', type: 'number', default: 0.001, step: 0.001 },
  iou: { label: '验证 NMS IoU 阈值', type: 'number', default: 0.7, step: 0.05 },
  max_det: { label: '单张图最大检测数', type: 'number', default: 300 },
  plots: { label: '保存训练与验证图表', type: 'boolean', default: true },
  lrf: { label: '最终学习率系数', type: 'number', default: 0.01, step: 0.001 },
  warmup_momentum: { label: '预热初始动量', type: 'number', default: 0.8, step: 0.05 },
  warmup_bias_lr: { label: '预热偏置学习率', type: 'number', default: 0.1, step: 0.01 },
  cls_pw: { label: '类别不平衡权重指数', type: 'number', default: 0, step: 0.1 },
  pose: { label: 'Pose loss 权重（姿态）', type: 'number', default: 12, step: 0.5 },
  kobj: { label: '关键点置信度权重（姿态）', type: 'number', default: 1, step: 0.1 },
  rle: { label: 'RLE loss 权重（姿态）', type: 'number', default: 1, step: 0.1 },
  angle: { label: '角度 loss 权重（旋转框）', type: 'number', default: 1, step: 0.1 },
  dlog: { label: '深度 SILog loss 权重', type: 'number', default: 1, step: 0.1 },
  dgrad: { label: '深度梯度 loss 权重', type: 'number', default: 0.5, step: 0.1 },
  dlam: { label: '深度尺度不变系数', type: 'number', default: 1, step: 0.1 },
  nbs: { label: '损失归一化标称 Batch', type: 'number', default: 64 },
  bgr: { label: 'RGB/BGR 通道交换概率', type: 'number', default: 0, step: 0.05 },
  cutmix: { label: 'CutMix 增强概率', type: 'number', default: 0, step: 0.05 },
  copy_paste_mode: { label: 'Copy-paste 策略（分割/旋转框）', type: 'text', default: 'flip' },
  auto_augment: { label: '自动增强策略（分类）', type: 'text', default: 'randaugment' },
  erasing: { label: '随机擦除概率（分类）', type: 'number', default: 0.4, step: 0.05 },
}

const baseDefinitions: Record<string, ParameterDefinition> = {
  epochs: { label: '训练轮数', type: 'number' }, batch: { label: 'Batch 大小', type: 'number' }, imgsz: { label: '输入尺寸', type: 'number' },
  device: { label: 'GPU 设备', type: 'text' }, workers: { label: '数据线程', type: 'number' }, optimizer: { label: '优化器', type: 'text' },
  patience: { label: '早停耐心值', type: 'number' }, lr0: { label: '初始学习率', type: 'number' }, close_mosaic: { label: '关闭 Mosaic 轮数', type: 'number' },
  pretrained: { label: '预训练权重', type: 'boolean' }, cache: { label: '缓存数据', type: 'boolean' }, amp: { label: 'AMP', type: 'boolean' },
}

const baseParameterKeys = new Set(Object.keys(baseDefinitions))

export const importDefinitions: Record<string, ParameterDefinition> = { ...baseDefinitions, ...extraDefinitions }

export function definitionsFromCatalog(parameters: TrainingParameterDefinition[]): Record<string, ParameterDefinition> {
  return {
    ...baseDefinitions,
    ...Object.fromEntries(parameters.map((item) => [item.key, {
      label: item.label,
      type: item.type,
      default: item.default == null ? item.type === 'boolean' ? false : item.type === 'number' ? 0 : '' : item.default,
    }])),
  }
}

const contextDefinitions: Record<string, { label: string; message: string }> = {
  task: { label: '任务类型', message: '任务上下文参数，请在任务类型中设置' },
  mode: { label: '运行模式', message: '平台训练任务固定使用 train，无需导入' },
  model: { label: '基础模型', message: '模型资源不会在参数导入中覆盖，请在模型节点设置' },
  data: { label: '数据集', message: '数据集资源不会在参数导入中覆盖，请在数据集节点设置' },
  name: { label: '运行名称', message: '运行名称由平台管理，不会作为训练参数导入' },
  project: { label: '输出目录', message: '训练输出目录由平台管理，不会作为训练参数导入' },
  resume: { label: '断点恢复', message: '断点恢复由平台操作管理，不会作为训练参数导入' },
  save_dir: { label: '保存目录', message: '保存目录由平台管理，不会作为训练参数导入' },
  source: { label: '输入源', message: '输入源不是训练参数，不会导入' },
}

const deprecatedDefinitions: Record<string, string> = {
  label_smoothing: '当前 Ultralytics 版本已移除此参数，没有直接替代项',
  save_hybrid: '当前 Ultralytics 版本已移除此参数',
  crop_fraction: '当前 Ultralytics 版本已移除此参数',
}

function parseImportValue(raw: string, type: DetectedParameter['type']): ImportValue | null {
  let value = raw.trim().replace(/[;,]+$/, '').trim()
  if (!value) return null
  value = value.replace(/\s+#.*$/, '').trim()
  if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) value = value.slice(1, -1)
  if (type === 'boolean') {
    if (/^(true|yes|on)$/i.test(value)) return true
    if (/^(false|no|off)$/i.test(value)) return false
    return null
  }
  if (type === 'number') {
    const number = Number(value)
    return Number.isFinite(number) ? number : null
  }
  return value
}

function stripImportComments(text: string): string {
  let result = ''
  let quote = ''
  let escaped = false
  for (let index = 0; index < text.length; index++) {
    const character = text[index]
    if (quote) {
      result += character
      if (escaped) escaped = false
      else if (character === '\\') escaped = true
      else if (character === quote) quote = ''
      continue
    }
    if (character === '"' || character === "'") { quote = character; result += character; continue }
    if (character === '#' || character === '/' && text[index + 1] === '/') {
      while (index < text.length && text[index] !== '\n') index++
      if (index < text.length) result += '\n'
      continue
    }
    result += character
  }
  return result
}

function editDistance(left: string, right: string): number {
  const row = Array.from({ length: right.length + 1 }, (_, index) => index)
  for (let leftIndex = 1; leftIndex <= left.length; leftIndex++) {
    let previous = row[0]
    row[0] = leftIndex
    for (let rightIndex = 1; rightIndex <= right.length; rightIndex++) {
      const current = row[rightIndex]
      row[rightIndex] = Math.min(row[rightIndex] + 1, row[rightIndex - 1] + 1, previous + (left[leftIndex - 1] === right[rightIndex - 1] ? 0 : 1))
      previous = current
    }
  }
  return row[right.length]
}

function closestSupportedKey(key: string, definitions: Record<string, ParameterDefinition>): string | undefined {
  const matches = Object.keys(definitions).map((candidate) => ({ candidate, distance: editDistance(key, candidate) })).sort((left, right) => left.distance - right.distance)
  const closest = matches[0]
  return closest && closest.distance <= Math.max(2, Math.floor(key.length * .25)) ? closest.candidate : undefined
}

export function analyzeImportedParameters(text: string, definitions: Record<string, ParameterDefinition> = importDefinitions): ImportAnalysis {
  const cleaned = stripImportComments(text)
  const pattern = /(?:^|[\s,{(])['"]?([A-Za-z_][A-Za-z0-9_]*)['"]?\s*(?:=|:)\s*("[^"]*"|'[^']*'|\[[^\]]*\]|[^,\s})]+)/gim
  const candidates = new Map<string, Array<{ rawValue: string; line: number }>>()
  let match: RegExpExecArray | null
  while ((match = pattern.exec(cleaned))) {
    const key = match[1].toLowerCase()
    const keyOffset = match[0].lastIndexOf(match[1])
    const line = cleaned.slice(0, match.index + Math.max(0, keyOffset)).split('\n').length
    const entries = candidates.get(key) || []
    entries.push({ rawValue: match[2], line })
    candidates.set(key, entries)
  }

  const parameters: DetectedParameter[] = []
  const diagnostics: ImportDiagnostic[] = []
  for (const [key, entries] of candidates) {
    const effective = entries.at(-1)!
    const duplicateMessage = entries.length > 1 ? `；出现 ${entries.length} 次，采用最后一次（第 ${effective.line} 行）` : ''
    const definition = definitions[key]
    if (definition) {
      const value = parseImportValue(effective.rawValue, definition.type)
      if (value == null) {
        diagnostics.push({ key, label: definition.label, rawValue: effective.rawValue, line: effective.line, occurrences: entries.length, status: 'invalid', message: `值格式无效，需要${definition.type === 'number' ? '数字' : definition.type === 'boolean' ? 'true 或 false' : '文本'}${duplicateMessage}` })
      } else {
        parameters.push({ key, label: definition.label, value, type: definition.type })
        diagnostics.push({ key, label: definition.label, rawValue: String(value), line: effective.line, occurrences: entries.length, status: 'supported', message: `可以导入${duplicateMessage}` })
      }
      continue
    }
    if (contextDefinitions[key]) {
      const context = contextDefinitions[key]
      diagnostics.push({ key, label: context.label, rawValue: effective.rawValue, line: effective.line, occurrences: entries.length, status: 'context', message: `${context.message}${duplicateMessage}` })
      continue
    }
    if (deprecatedDefinitions[key]) {
      diagnostics.push({ key, label: '已废弃参数', rawValue: effective.rawValue, line: effective.line, occurrences: entries.length, status: 'deprecated', message: `${deprecatedDefinitions[key]}${duplicateMessage}` })
      continue
    }
    const suggestion = closestSupportedKey(key, definitions)
    diagnostics.push({ key, label: '未知参数', rawValue: effective.rawValue, line: effective.line, occurrences: entries.length, status: 'unknown', message: `${suggestion ? `平台不支持该参数，是否想输入 ${suggestion}？` : '平台当前不支持该参数，请确认名称和当前 YOLO 版本'}${duplicateMessage}` })
  }
  return { parameters, diagnostics }
}

export function parseImportedParameters(text: string): DetectedParameter[] {
  return analyzeImportedParameters(text).parameters
}

export function ParameterImportResults({ diagnostics, parameters, selectedKeys, onSelectionChange }: {
  diagnostics: ImportDiagnostic[]
  parameters: DetectedParameter[]
  selectedKeys: string[]
  onSelectionChange: (keys: string[]) => void
}) {
  const supportedKeys = parameters.map((item) => item.key)
  const excludedCount = diagnostics.length - parameters.length
  const allSelected = supportedKeys.length > 0 && supportedKeys.every((key) => selectedKeys.includes(key))
  const parameterValues = new Map(parameters.map((item) => [item.key, item.value]))
  const labels: Record<ImportDiagnosticStatus, string> = { supported: '可导入', context: '单独设置', deprecated: '已废弃', unknown: '未识别', invalid: '值无效' }
  return <>
    <div className="import-summary" role="status">
      <span><strong>{diagnostics.length}</strong>检测字段</span><span className="supported"><strong>{parameters.length}</strong>可导入</span><span className={excludedCount ? 'excluded' : ''}><strong>{excludedCount}</strong>不导入</span>
    </div>
    <div className="detected-toolbar"><strong>{diagnostics.length ? '逐项检查结果' : '等待粘贴参数'}</strong>{!!supportedKeys.length && <button type="button" onClick={() => onSelectionChange(allSelected ? [] : supportedKeys)}>{allSelected ? '取消全选' : '选择全部可用项'}</button>}</div>
    <div className="detected-list import-diagnostic-list">
      {diagnostics.map((item) => {
        const supported = item.status === 'supported'
        const checked = supported && selectedKeys.includes(item.key)
        return <div key={item.key} className={`import-diagnostic ${item.status} ${checked ? 'selected' : ''}`}>
          {supported ? <input aria-label={`选择 ${item.key}`} type="checkbox" checked={checked} onChange={() => onSelectionChange(checked ? selectedKeys.filter((key) => key !== item.key) : [...selectedKeys, item.key])} /> : <i aria-hidden="true">!</i>}
          <span><strong>{item.label}</strong><code>{item.key} · 第 {item.line} 行</code><small>{item.message}</small></span>
          <div><em>{String(parameterValues.get(item.key) ?? item.rawValue)}</em><b>{labels[item.status]}</b></div>
        </div>
      })}
      {!diagnostics.length && <div className="detected-empty">粘贴参数后会列出全部检测结果，未识别项也不会被隐藏</div>}
    </div>
  </>
}

export function EditParametersModal({ task, open, onClose, onSaved }: Props) {
  const [form, setForm] = useState({ epochs: 100, batch: 16, imgsz: 640, device: '0', workers: 8, optimizer: 'auto', patience: 50, lr0: '', close_mosaic: 10, pretrained: true, cache: false, amp: true })
  const [extras, setExtras] = useState<Record<string, ExtraValue>>({})
  const [customArgs, setCustomArgs] = useState<Record<string, ExtraValue>>({})
  const [newKey, setNewKey] = useState('')
  const [parameterSearch, setParameterSearch] = useState('')
  const [parameterPickerOpen, setParameterPickerOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [jsonOpen, setJsonOpen] = useState(false)
  const [jsonCopied, setJsonCopied] = useState(false)
  const [importText, setImportText] = useState('')
  const [detectedParameters, setDetectedParameters] = useState<DetectedParameter[]>([])
  const [importDiagnostics, setImportDiagnostics] = useState<ImportDiagnostic[]>([])
  const [selectedImportKeys, setSelectedImportKeys] = useState<string[]>([])
  const [customKey, setCustomKey] = useState('')
  const [customType, setCustomType] = useState<'number' | 'boolean' | 'text'>('number')
  const [customValue, setCustomValue] = useState('')
  const [expanded, setExpanded] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [catalogDefinitions, setCatalogDefinitions] = useState<Record<string, ParameterDefinition> | null>(null)
  const latestTask = useRef(task)
  latestTask.current = task

  useEffect(() => {
    const currentTask = latestTask.current
    if (!currentTask || !open) return
    setForm({
      epochs: currentTask.epochs, batch: currentTask.batch, imgsz: currentTask.imgsz, device: currentTask.device, workers: currentTask.workers,
      optimizer: currentTask.optimizer, patience: Number(currentTask.config.patience ?? 50),
      lr0: currentTask.config.lr0 == null ? '' : String(currentTask.config.lr0), close_mosaic: Number(currentTask.config.close_mosaic ?? 10),
      pretrained: currentTask.config.pretrained !== false, cache: Boolean(currentTask.config.cache), amp: currentTask.config.amp !== false,
    })
    setExtras(Object.fromEntries(Object.keys(extraDefinitions).filter((key) => currentTask.config[key] != null).map((key) => [key, currentTask.config[key] as ExtraValue])))
    setCustomArgs((currentTask.config.custom_args as Record<string, ExtraValue> | undefined) || {})
    setNewKey(''); setParameterSearch(''); setParameterPickerOpen(false)
    setImportOpen(false); setJsonOpen(false); setJsonCopied(false); setImportText(''); setDetectedParameters([]); setImportDiagnostics([]); setSelectedImportKeys([])
    setError('')
  }, [task?.id, open])

  useEffect(() => {
    if (!open) return
    let active = true
    api.trainingParameters().then((result) => {
      if (!active) return
      if (!result.parameters.length) { setCatalogDefinitions(null); return }
      const definitions = definitionsFromCatalog(result.parameters)
      setCatalogDefinitions(definitions)
      const currentTask = latestTask.current
      if (currentTask) {
        setExtras((current) => ({
          ...Object.fromEntries(Object.keys(definitions).filter((key) => !baseParameterKeys.has(key) && currentTask.config[key] != null).map((key) => [key, currentTask.config[key] as ExtraValue])),
          ...current,
        }))
      }
    }).catch(() => {
      if (active) setCatalogDefinitions(null)
    })
    return () => { active = false }
  }, [open])

  const activeDefinitions = catalogDefinitions || importDefinitions
  const activeExtraDefinitions = useMemo(() => Object.fromEntries(Object.entries(activeDefinitions).filter(([key]) => !baseParameterKeys.has(key))), [activeDefinitions])

  const availableKeys = useMemo(() => Object.keys(activeExtraDefinitions).filter((key) => !(key in extras)), [activeExtraDefinitions, extras])
  const filteredAvailableKeys = useMemo(() => {
    const keyword = parameterSearch.trim().toLowerCase()
    if (!keyword) return availableKeys
    return availableKeys.filter((key) => `${activeExtraDefinitions[key].label} ${key}`.toLowerCase().includes(keyword))
  }, [activeExtraDefinitions, availableKeys, parameterSearch])
  const jsonParameters = useMemo(() => {
    const parameters: Record<string, ExtraValue> = {
      task: task?.task_type || 'detect', mode: 'train', model: task?.model_path || '', data: task?.data_path || '',
      epochs: form.epochs, batch: form.batch, imgsz: form.imgsz, device: form.device, workers: form.workers,
      optimizer: form.optimizer, patience: form.patience, pretrained: form.pretrained, cache: form.cache,
      amp: form.amp, close_mosaic: form.close_mosaic, ...extras, ...customArgs,
    }
    if (form.lr0 !== '') parameters.lr0 = Number(form.lr0)
    return parameters
  }, [customArgs, extras, form, task?.data_path, task?.model_path, task?.task_type])
  const jsonText = useMemo(() => JSON.stringify(jsonParameters, null, 2), [jsonParameters])
  if (!open || !task) return null
  const locked = ['queued', 'starting', 'running', 'stopping'].includes(task.status)

  function update<K extends keyof typeof form>(key: K, value: (typeof form)[K]) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  function addExtra() {
    if (!newKey) return
    const definition = activeExtraDefinitions[newKey]
    if (!definition) return
    setExtras((current) => ({ ...current, [newKey]: definition.default ?? (definition.type === 'boolean' ? false : definition.type === 'number' ? 0 : '') }))
    setNewKey(''); setParameterSearch(''); setParameterPickerOpen(false)
  }

  function updateExtra(key: string, raw: string | boolean) {
    const definition = activeExtraDefinitions[key]
    if (!definition) return
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

  function openImport() {
    setImportText(''); setDetectedParameters([]); setImportDiagnostics([]); setSelectedImportKeys([]); setImportOpen(true)
  }

  async function copyJson() {
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(jsonText)
      else {
        const textarea = document.createElement('textarea')
        textarea.value = jsonText
        textarea.style.position = 'fixed'
        textarea.style.opacity = '0'
        document.body.appendChild(textarea)
        textarea.select()
        document.execCommand('copy')
        textarea.remove()
      }
      setJsonCopied(true)
      window.setTimeout(() => setJsonCopied(false), 1800)
    } catch {
      setError('复制失败，请在 JSON 窗口中手动选择文本')
    }
  }

  function changeImportText(text: string) {
    const analysis = analyzeImportedParameters(text, activeDefinitions)
    setImportText(text); setDetectedParameters(analysis.parameters); setImportDiagnostics(analysis.diagnostics); setSelectedImportKeys(analysis.parameters.map((item) => item.key))
  }

  function applyImportedParameters() {
    const selected = detectedParameters.filter((item) => selectedImportKeys.includes(item.key))
    const importedBase = selected.filter((item) => baseParameterKeys.has(item.key))
    const importedExtras = selected.filter((item) => !baseParameterKeys.has(item.key))
    if (importedBase.length) setForm((current) => ({ ...current, ...Object.fromEntries(importedBase.map((item) => [item.key, item.value])) }))
    if (importedExtras.length) setExtras((current) => ({ ...current, ...Object.fromEntries(importedExtras.map((item) => [item.key, item.value])) }))
    setImportOpen(false)
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
        <div className="parameter-toolbar"><button type="button" className="advanced-toggle parameter-toggle" onClick={() => setExpanded((value) => !value)}><ChevronDown size={16} className={expanded ? 'rotated' : ''} />高级参数 <em>{Object.keys(extras).length} 项</em></button><button type="button" className="json-preview-button" onClick={() => { setJsonCopied(false); setJsonOpen(true) }}><Code2 size={14} />查看 JSON</button><button type="button" className="batch-import-button" onClick={openImport}><Braces size={14} />批量导入参数</button></div>
        {expanded && <div className="extra-parameters">
          {Object.entries(extras).map(([key, value]) => { const definition = activeExtraDefinitions[key] || { label: key, type: typeof value === 'boolean' ? 'boolean' as const : typeof value === 'number' ? 'number' as const : 'text' as const }; return <div className="extra-row" key={key}><div><span>{definition.label}</span><code>{key}</code></div>{definition.type === 'boolean' ? <select value={String(value)} onChange={(e) => updateExtra(key, e.target.value === 'true')}><option value="true">开启</option><option value="false">关闭</option></select> : <input type={definition.type} step={definition.step} value={String(value)} onChange={(e) => updateExtra(key, e.target.value)} />}<button type="button" title="删除参数" onClick={() => setExtras((current) => { const next = { ...current }; delete next[key]; return next })}><Trash2 size={15} /></button></div> })}
          <div className="add-parameter">
            <div className="parameter-picker">
              <button type="button" className={parameterPickerOpen ? 'open' : ''} onClick={() => setParameterPickerOpen((value) => !value)}><span>{newKey && activeExtraDefinitions[newKey] ? <><strong>{activeExtraDefinitions[newKey].label}</strong><code>{newKey}</code></> : '选择要添加的训练参数'}</span><ChevronDown size={15} /></button>
              {parameterPickerOpen && <div className="parameter-picker-menu">
                <label><Search size={14} /><input autoFocus value={parameterSearch} onChange={(event) => setParameterSearch(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && filteredAvailableKeys[0]) { event.preventDefault(); setNewKey(filteredAvailableKeys[0]); setParameterPickerOpen(false); setParameterSearch('') } }} placeholder="搜索中文名称或参数名，例如 freeze" /></label>
                <div>{filteredAvailableKeys.map((key) => <button type="button" key={key} onClick={() => { setNewKey(key); setParameterPickerOpen(false); setParameterSearch('') }}><span><strong>{activeExtraDefinitions[key].label}</strong><code>{key}</code></span>{newKey === key && <Check size={14} />}</button>)}{!filteredAvailableKeys.length && <p>没有找到匹配的参数</p>}</div>
              </div>}
            </div>
            <button type="button" className="button ghost" disabled={!newKey} onClick={addExtra}><Plus size={15} />添加参数</button>
          </div>
          <div className="custom-parameter-section">
            <div className="custom-title"><Braces size={15} /><div><strong>自定义 YOLO 参数</strong><span>用于添加预设列表中没有的合法参数</span></div></div>
            {Object.entries(customArgs).map(([key, value]) => <div className="extra-row custom-row" key={key}><div><span>自定义参数</span><code>{key}</code></div>{typeof value === 'boolean' ? <select value={String(value)} onChange={(e) => updateCustom(key, e.target.value === 'true')}><option value="true">开启</option><option value="false">关闭</option></select> : <input type={typeof value === 'number' ? 'number' : 'text'} value={String(value)} onChange={(e) => updateCustom(key, e.target.value)} />}<button type="button" title="删除参数" onClick={() => setCustomArgs((current) => { const next = { ...current }; delete next[key]; return next })}><Trash2 size={15} /></button></div>)}
            <div className="custom-add-row"><input value={customKey} onChange={(e) => setCustomKey(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))} placeholder="参数名，例如 save_json" /><select value={customType} onChange={(e) => { const type = e.target.value as typeof customType; setCustomType(type); setCustomValue(type === 'boolean' ? 'true' : '') }}><option value="number">数字</option><option value="boolean">布尔</option><option value="text">文本</option></select>{customType === 'boolean' ? <select value={customValue || 'true'} onChange={(e) => setCustomValue(e.target.value)}><option value="true">开启</option><option value="false">关闭</option></select> : <input type={customType === 'number' ? 'number' : 'text'} value={customValue} onChange={(e) => setCustomValue(e.target.value)} placeholder="参数值" />}<button type="button" className="button ghost" disabled={!customKey || customKey in customArgs} onClick={addCustom}><Plus size={15} />自定义添加</button></div>
            <p className="custom-help">参数名只能包含小写字母、数字和下划线；模型、数据集、设备等平台关键参数不能覆盖。</p>
          </div>
        </div>}
      </div>
      {importOpen && <div className="parameter-import-backdrop" onMouseDown={() => setImportOpen(false)}><div className="parameter-import-modal" onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">PASTE CONFIGURATION</span><h3>批量识别训练参数</h3><p>可粘贴 Python、YOLO CLI、YAML 或 JSON 样式的参数。全部字段都会列出诊断结果，不会执行粘贴的代码。</p></div><button type="button" className="icon-button" onClick={() => setImportOpen(false)}><X size={18} /></button></header>
        <div className="parameter-import-body"><textarea value={importText} onChange={(event) => changeImportText(event.target.value)} placeholder={'例如：\nmodel.train(epochs=100, batch=16, imgsz=640, lr0=0.01, freeze=10)\n\n或：\nepochs: 100\namp: true\nmosaic: 0.8'} />
          <ParameterImportResults diagnostics={importDiagnostics} parameters={detectedParameters} selectedKeys={selectedImportKeys} onSelectionChange={setSelectedImportKeys} />
        </div>
        <footer><span>将导入 {selectedImportKeys.length} 项{importDiagnostics.length > detectedParameters.length ? `，${importDiagnostics.length - detectedParameters.length} 项不会导入` : ''}</span><button type="button" className="button ghost" onClick={() => setImportOpen(false)}>取消</button><button type="button" className="button primary" disabled={!selectedImportKeys.length} onClick={applyImportedParameters}>导入 {selectedImportKeys.length} 个参数</button></footer>
      </div></div>}
      {jsonOpen && <div className="parameter-import-backdrop" onMouseDown={() => setJsonOpen(false)}><div className="parameter-json-modal" onMouseDown={(event) => event.stopPropagation()}>
        <header><div><span className="eyebrow">EFFECTIVE CONFIGURATION</span><h3>完整 JSON 参数</h3><p>{Object.keys(jsonParameters).length} 个当前参数</p></div><button type="button" className="icon-button" onClick={() => setJsonOpen(false)}><X size={18} /></button></header>
        <div className="parameter-json-body"><pre>{jsonText}</pre></div>
        <footer><span>当前表单配置</span><button type="button" className="button primary" onClick={copyJson}>{jsonCopied ? <Check size={15} /> : <Clipboard size={15} />}{jsonCopied ? '已复制' : '复制 JSON'}</button></footer>
      </div></div>}
      <footer><span className="parameter-note">双击配置节点可随时重新打开</span><button type="button" className="button ghost" onClick={onClose}>取消</button><button className="button primary" disabled={saving || locked}>{saving && <LoaderCircle size={16} className="spin" />}保存参数</button></footer>
    </form>
  </div>
}
