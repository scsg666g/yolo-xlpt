import type { Artifact, ComputeNode, ComputeNodeCredential, ComputeNodeSettings, ComputeResources, DatasetUploadResult, EffectiveBestWeight, ManagedDataset, MetricPoint, ModelTest, ModelTestPage, PlatformHealth, Resources, TaskCreatePayload, TaskEvent, TrainingParameterDefinition, TrainingRun, TrainingTask, TrainingTrace } from './types'

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

async function requestWithTimeout<T>(path: string, init?: RequestInit, timeoutMs = 15_000,
  timeoutMessage = '请求超时，请检查网络后重试'): Promise<T> {
  const controller = new AbortController()
  let timedOut = false
  const abort = () => controller.abort()
  if (init?.signal?.aborted) controller.abort()
  init?.signal?.addEventListener('abort', abort, { once: true })
  const timer = window.setTimeout(() => { timedOut = true; controller.abort() }, timeoutMs)
  try {
    // Include response body decoding in the timeout, not just response headers.
    return await request<T>(path, { ...init, signal: controller.signal })
  } catch (reason) {
    if (timedOut) throw new Error(timeoutMessage)
    throw reason
  } finally {
    window.clearTimeout(timer)
    init?.signal?.removeEventListener('abort', abort)
  }
}

const deletionTimeout = '删除请求超时，后台可能已经受理。请关闭弹窗，查看任务列表或回收站确认状态后再操作。'

export const api = {
  health: (signal?: AbortSignal) => requestWithTimeout<PlatformHealth>('/api/health', { signal }, 8_000, '健康检查超时'),
  createModelTest: (
    artifact_id: number,
    confidence: number,
    iou: number,
    files: File[],
    signal: AbortSignal,
    run_id?: string | null,
    rerun_of_test_id?: string | null,
  ) => requestWithTimeout<ModelTest>('/api/model-tests', {
    method: 'POST',
    body: JSON.stringify({
      artifact_id,
      confidence,
      iou,
      run_id: run_id || undefined,
      rerun_of_test_id: rerun_of_test_id || undefined,
      files: files.map(({ name, size }) => ({ name, size })),
    }),
    signal,
  }),
  modelTests: (taskId: string, signal?: AbortSignal, cursor?: string, limit = 50, runId?: string | null) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (cursor) params.set('cursor', cursor)
    if (runId) params.set('run_id', runId)
    return requestWithTimeout<ModelTestPage>(`/api/tasks/${encodeURIComponent(taskId)}/model-tests?${params}`, { signal })
  },
  taskModelTest: (taskId: string, testId: string, signal?: AbortSignal) =>
    requestWithTimeout<ModelTest>(`/api/tasks/${encodeURIComponent(taskId)}/model-tests/${encodeURIComponent(testId)}`, { signal }),
  uploadModelTestImage: (id: string, index: number, file: File, signal: AbortSignal) =>
    requestWithTimeout<void>(`/api/model-tests/${id}/images/${index}`, { method: 'PUT', body: file, headers: { 'Content-Type': 'application/octet-stream' }, signal }, 120_000),
  startModelTest: (id: string, signal: AbortSignal) => requestWithTimeout<ModelTest>(`/api/model-tests/${id}/start`, { method: 'POST', signal }, 120_000),
  abandonModelTest: (id: string, keepalive = false) =>
    request<ModelTest>(`/api/model-tests/${id}/abandon`, { method: 'POST', keepalive }),
  modelTest: (id: string, signal: AbortSignal) => requestWithTimeout<ModelTest>(`/api/model-tests/${id}`, { signal }),
  deleteModelTest: (id: string) => requestWithTimeout<void>(`/api/model-tests/${id}`, { method: 'DELETE' }),
  createDatasetUpload: (name: string, files: Array<{ path: string; size: number }>, signal: AbortSignal) =>
    request<{ upload_id: string; destination: string }>('/api/datasets/uploads', { method: 'POST', body: JSON.stringify({ name, files }), signal }),
  datasetUploadStatus: (id: string, signal: AbortSignal) =>
    request<{ uploaded_indices: number[]; result?: DatasetUploadResult }>(`/api/datasets/uploads/${id}`, { signal }),
  completeDatasetUpload: (id: string, names: string[], signal: AbortSignal) =>
    request<DatasetUploadResult>(`/api/datasets/uploads/${id}/complete`, { method: 'POST', body: JSON.stringify({ names }), signal }),
  cancelDatasetUpload: (id: string) => request<void>(`/api/datasets/uploads/${id}`, { method: 'DELETE' }),
  managedDatasets: (signal?: AbortSignal) => requestWithTimeout<ManagedDataset[]>('/api/datasets/managed', { signal }),
  deleteManagedDataset: (name: string) => requestWithTimeout<void>(`/api/datasets/managed/${encodeURIComponent(name)}`, { method: 'DELETE' }, 30_000,
    '删除请求超时，请刷新数据集列表确认状态'),
  tasks: (signal?: AbortSignal) => request<TrainingTask[]>('/api/tasks', { signal }),
  reorderTasks: (task_ids: string[]) => request<void>('/api/tasks/order', { method: 'PATCH', body: JSON.stringify({ task_ids }) }),
  deletedTasks: (signal?: AbortSignal) => requestWithTimeout<TrainingTask[]>('/api/tasks?deleted_only=true', { signal }).then((tasks) => tasks.filter((task) => task.deleted_at)),
  task: (id: string, signal?: AbortSignal) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}`, { signal }),
  create: (payload: TaskCreatePayload) => request<TrainingTask>('/api/tasks', { method: 'POST', body: JSON.stringify(payload) }),
  start: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/start`, { method: 'POST' }),
  resume: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/resume`, { method: 'POST' }),
  retry: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/retry`, { method: 'POST' }),
  stop: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/stop`, { method: 'POST' }),
  forceInterrupt: (id: string, runId: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/force-interrupt`, {
    method: 'POST', body: JSON.stringify({ run_id: runId, confirm_process_stopped: true }),
  }),
  clone: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/clone`, { method: 'POST' }),
  rename: (id: string, name: string) => request<TrainingTask>(`/api/tasks/${id}/name`, { method: 'PATCH', body: JSON.stringify({ name }) }),
  remove: (id: string) => requestWithTimeout<void>(`/api/tasks/${id}/trash`, { method: 'POST' }, 30_000, deletionTimeout),
  restore: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/restore`, { method: 'POST' }, 30_000,
    '恢复请求超时，后台可能已经受理。请刷新任务列表和回收站确认状态。'),
  permanentRemove: (id: string) => requestWithTimeout<{ status: string; message: string } | undefined>(`/api/tasks/${id}/permanent`, { method: 'DELETE' }, 30_000, deletionTimeout),
  prepareArtifact: (id: number, retry = false, signal?: AbortSignal) => requestWithTimeout<{ status: string; message?: string }>(`/api/artifacts/${id}/prepare?retry=${retry}`, { method: 'POST', signal }),
  saveLayout: (id: string, layout: TrainingTask['canvas_layout'], signal?: AbortSignal) =>
    requestWithTimeout<TrainingTask>(`/api/tasks/${id}/layout`, { method: 'PATCH', body: JSON.stringify({ layout }), signal }, 15_000,
      '保存布局超时，请检查网络后重试'),
  updateConfig: (id: string, payload: {
    epochs: number; batch: number; imgsz: number; device: string; workers: number; optimizer: string; config: Record<string, unknown>; custom_config: Record<string, unknown>
  }) => request<TrainingTask>(`/api/tasks/${id}/config`, { method: 'PATCH', body: JSON.stringify(payload) }),
  updateResources: (id: string, payload: { model_path?: string; data_path?: string }) =>
    request<TrainingTask>(`/api/tasks/${id}/resources`, { method: 'PATCH', body: JSON.stringify(payload) }),
  metrics: (id: string) => request<MetricPoint[]>(`/api/tasks/${id}/metrics`),
  artifacts: (id: string) => request<Artifact[]>(`/api/tasks/${id}/artifacts`),
  events: (id: string) => request<TaskEvent[]>(`/api/tasks/${id}/events`),
  logs: (id: string) => request<{ lines: string[]; path: string | null }>(`/api/tasks/${id}/logs?tail=240`),
  runs: (id: string, signal?: AbortSignal) => requestWithTimeout<TrainingRun[]>(`/api/tasks/${id}/runs`, { signal }),
  runMetrics: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<MetricPoint[]>(`/api/tasks/${id}/runs/${runId}/metrics`, { signal }),
  runArtifacts: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<Artifact[]>(`/api/tasks/${id}/runs/${runId}/artifacts`, { signal }),
  runEffectiveBestWeight: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<EffectiveBestWeight>(`/api/tasks/${id}/runs/${runId}/effective-best-weight`, { signal }),
  runEvents: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<TaskEvent[]>(`/api/tasks/${id}/runs/${runId}/events`, { signal }),
  runLogs: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<{ lines: string[]; path: string | null }>(`/api/tasks/${id}/runs/${runId}/logs?tail=240`, { signal }),
  trainingTrace: (id: string, runId: string) => request<TrainingTrace>(`/api/tasks/${id}/runs/${runId}/trace`),
  resources: () => request<Resources>('/api/resources'),
  compute: () => request<ComputeResources>('/api/compute'),
  computeNodes: (signal?: AbortSignal) => request<ComputeNode[]>('/api/compute/nodes', { signal }),
  computeNode: (id: string, signal?: AbortSignal) => request<ComputeNode>(`/api/compute/nodes/${id}`, { signal }),
  createComputeNode: (payload: ComputeNodeSettings) => request<ComputeNodeCredential>('/api/compute/nodes', { method: 'POST', body: JSON.stringify(payload) }),
  updateComputeNode: (id: string, payload: ComputeNodeSettings) => request<ComputeNode>(`/api/compute/nodes/${id}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  rotateComputeToken: (id: string) => request<ComputeNodeCredential>(`/api/compute/nodes/${id}/token`, { method: 'POST' }),
  deleteComputeNode: (id: string) => request<void>(`/api/compute/nodes/${id}`, { method: 'DELETE' }),
  trainingParameters: () => request<{ source: string; parameters: TrainingParameterDefinition[] }>('/api/training-parameters'),
  importRuns: () => requestWithTimeout<{ imported: number }>('/api/import', { method: 'POST' }, 30_000, '同步历史任务超时，请稍后刷新查看'),
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
