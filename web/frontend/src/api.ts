import type { Artifact, ComputeResources, DatasetUploadResult, MetricPoint, Resources, TaskCreatePayload, TaskEvent, TrainingParameterDefinition, TrainingRun, TrainingTask } from './types'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    const detail = typeof body.detail === 'string' ? body.detail
      : Array.isArray(body.detail) ? body.detail.map((item: { loc?: Array<string | number>; msg?: string }) => `${item.loc?.slice(1).join('.') || '参数'}：${item.msg || '格式错误'}`).join('；')
      : '请求失败'
    const error = new Error(detail) as Error & { status?: number }
    error.status = response.status
    throw error
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  createDatasetUpload: (name: string, files: Array<{ path: string; size: number }>, signal: AbortSignal) =>
    request<{ upload_id: string; destination: string }>('/api/datasets/uploads', { method: 'POST', body: JSON.stringify({ name, files }), signal }),
  datasetUploadStatus: (id: string, signal: AbortSignal) =>
    request<{ uploaded_indices: number[]; result?: DatasetUploadResult }>(`/api/datasets/uploads/${id}`, { signal }),
  completeDatasetUpload: (id: string, names: string[], signal: AbortSignal) =>
    request<DatasetUploadResult>(`/api/datasets/uploads/${id}/complete`, { method: 'POST', body: JSON.stringify({ names }), signal }),
  cancelDatasetUpload: (id: string) => request<void>(`/api/datasets/uploads/${id}`, { method: 'DELETE' }),
  tasks: () => request<TrainingTask[]>('/api/tasks'),
  reorderTasks: (task_ids: string[]) => request<void>('/api/tasks/order', { method: 'PATCH', body: JSON.stringify({ task_ids }) }),
  deletedTasks: () => request<TrainingTask[]>('/api/tasks?include_deleted=true').then((tasks) => tasks.filter((task) => task.deleted_at)),
  task: (id: string) => request<TrainingTask>(`/api/tasks/${id}`),
  create: (payload: TaskCreatePayload) => request<TrainingTask>('/api/tasks', { method: 'POST', body: JSON.stringify(payload) }),
  start: (id: string) => request<TrainingTask>(`/api/tasks/${id}/start`, { method: 'POST' }),
  resume: (id: string) => request<TrainingTask>(`/api/tasks/${id}/resume`, { method: 'POST' }),
  retry: (id: string) => request<TrainingTask>(`/api/tasks/${id}/retry`, { method: 'POST' }),
  stop: (id: string) => request<TrainingTask>(`/api/tasks/${id}/stop`, { method: 'POST' }),
  clone: (id: string) => request<TrainingTask>(`/api/tasks/${id}/clone`, { method: 'POST' }),
  remove: (id: string) => request<void>(`/api/tasks/${id}/trash`, { method: 'POST' }),
  restore: (id: string) => request<TrainingTask>(`/api/tasks/${id}/restore`, { method: 'POST' }),
  permanentRemove: (id: string) => request<void>(`/api/tasks/${id}/permanent`, { method: 'DELETE' }),
  saveLayout: (id: string, layout: TrainingTask['canvas_layout']) =>
    request<TrainingTask>(`/api/tasks/${id}/layout`, { method: 'PATCH', body: JSON.stringify({ layout }) }),
  updateConfig: (id: string, payload: {
    epochs: number; batch: number; imgsz: number; device: string; workers: number; optimizer: string; config: Record<string, unknown>; custom_config: Record<string, unknown>
  }) => request<TrainingTask>(`/api/tasks/${id}/config`, { method: 'PATCH', body: JSON.stringify(payload) }),
  updateResources: (id: string, payload: { model_path?: string; data_path?: string }) =>
    request<TrainingTask>(`/api/tasks/${id}/resources`, { method: 'PATCH', body: JSON.stringify(payload) }),
  metrics: (id: string) => request<MetricPoint[]>(`/api/tasks/${id}/metrics`),
  artifacts: (id: string) => request<Artifact[]>(`/api/tasks/${id}/artifacts`),
  events: (id: string) => request<TaskEvent[]>(`/api/tasks/${id}/events`),
  logs: (id: string) => request<{ lines: string[]; path: string | null }>(`/api/tasks/${id}/logs?tail=240`),
  runs: (id: string) => request<TrainingRun[]>(`/api/tasks/${id}/runs`),
  runMetrics: (id: string, runId: string) => request<MetricPoint[]>(`/api/tasks/${id}/runs/${runId}/metrics`),
  runArtifacts: (id: string, runId: string) => request<Artifact[]>(`/api/tasks/${id}/runs/${runId}/artifacts`),
  runEvents: (id: string, runId: string) => request<TaskEvent[]>(`/api/tasks/${id}/runs/${runId}/events`),
  runLogs: (id: string, runId: string) => request<{ lines: string[]; path: string | null }>(`/api/tasks/${id}/runs/${runId}/logs?tail=240`),
  resources: () => request<Resources>('/api/resources'),
  compute: () => request<ComputeResources>('/api/compute'),
  trainingParameters: () => request<{ source: string; parameters: TrainingParameterDefinition[] }>('/api/training-parameters'),
  importRuns: () => request<{ imported: number }>('/api/import', { method: 'POST' }),
}

export function uploadDatasetFile(id: string, index: number, file: File, signal: AbortSignal, onProgress: (bytes: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    const abort = () => xhr.abort()
    const cleanup = () => signal.removeEventListener('abort', abort)
    if (signal.aborted) { reject(new DOMException('上传已取消', 'AbortError')); return }
    xhr.open('PUT', `/api/datasets/uploads/${id}/files/${index}`)
    xhr.setRequestHeader('Content-Type', 'application/octet-stream')
    xhr.timeout = 10 * 60 * 1000
    xhr.upload.onprogress = (event) => onProgress(event.loaded)
    xhr.onload = () => {
      cleanup()
      if (xhr.status >= 200 && xhr.status < 300) { onProgress(file.size); resolve(); return }
      let detail = `文件上传失败（${xhr.status}）`
      try { detail = JSON.parse(xhr.responseText).detail || detail } catch { /* Non-JSON gateway error. */ }
      reject(new Error(detail))
    }
    xhr.onerror = () => { cleanup(); reject(new Error('网络连接中断，请点击重试继续上传')) }
    xhr.ontimeout = () => { cleanup(); reject(new Error('文件上传超时，请重试')) }
    xhr.onabort = () => { cleanup(); reject(new DOMException('上传已取消', 'AbortError')) }
    signal.addEventListener('abort', abort, { once: true })
    xhr.send(file)
  })
}

export function statusLabel(status: string): string {
  return {
    draft: '草稿', queued: '排队中', starting: '启动中', running: '训练中', stopping: '停止中',
    stopped: '已停止', completed: '已完成', failed: '失败', interrupted: '已中断',
  }[status] || status
}

export function metricValue(task: TrainingTask, keys: string[]): string {
  for (const key of keys) {
    const value = task.latest_metrics[key]
    if (typeof value === 'number') return value.toFixed(4)
  }
  return '—'
}

export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`
}
