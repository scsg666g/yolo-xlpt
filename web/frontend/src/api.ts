import type { DatasetScanRequest, DatasetScan, DatasetCandidatePage, DatasetBatchRequest, DatasetRegistrationBatch, DatasetHistoryRemoval, DatasetHistoryRemovalPreview, DatasetOperationSummary, DatasetOperationPage, DatasetLocation, DatasetOperation, Artifact, ComputeNode, ComputeNodeCredential, ComputeNodeRefreshResponse, ComputeNodeSettings, ComputeResources, DatasetUploadResult, DeletionJob, EffectiveBestWeight, ManagedDataset, MetricPoint, ModelAsset, ModelAssetPage, ModelAssetSourceType, ModelTest, ModelTestPage, PlatformHealth, PlatformStorageHealth, Resources, TaskCreatePayload, TaskDeletionPreview, TaskEvent, TaskGroup, TrainingParameterDefinition, TrainingRun, TrainingTask, TrainingTrace } from './types'

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

const pendingWrites = new Map<string, Promise<unknown>>()
const uncertainSubmission = '请求结果尚未确认，后台可能已经受理。请先刷新相关列表或重新打开详情确认状态，再决定是否重试。'

function requestWithTimeout<T>(path: string, init?: RequestInit, timeoutMs = 15_000,
  timeoutMessage?: string): Promise<T> {
  const write = !!init?.method && !['GET', 'HEAD'].includes(init.method.toUpperCase())
  // Share concurrent clicks only when no caller owns cancellation. Never retry a write automatically.
  const key = write && !init.signal && !init.keepalive ? JSON.stringify([path, init.method, init.body]) : null
  if (key && pendingWrites.has(key)) return pendingWrites.get(key) as Promise<T>
  const run = async () => {
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
      if (timedOut) throw new Error(timeoutMessage || (write ? `提交超时。${uncertainSubmission}` : '请求超时，请检查网络后重试'))
      if (write && reason instanceof TypeError) throw new Error(`网络连接中断。${uncertainSubmission}`)
      throw reason
    } finally {
      window.clearTimeout(timer)
      init?.signal?.removeEventListener('abort', abort)
    }
  }
  const promise = run()
  if (key) {
    pendingWrites.set(key, promise)
    void promise.finally(() => { if (pendingWrites.get(key) === promise) pendingWrites.delete(key) }).catch(() => undefined)
  }
  return promise
}

const deletionTimeout = '删除请求超时，后台可能已经受理。请关闭弹窗，查看任务列表或回收站确认状态后再操作。'
const datasetSubmissionTimeout = '数据集操作提交超时，后台可能已经受理。请刷新数据集和操作记录确认状态，再决定是否重试。'

export const api = {
  health: (signal?: AbortSignal) => requestWithTimeout<PlatformHealth>('/api/health', { signal }, 8_000, '健康检查超时'),
  storageHealth: (signal?: AbortSignal) => requestWithTimeout<PlatformStorageHealth>('/api/health/storage', { signal }, 8_000, '存储状态检查超时'),
  createModelTest: (
    artifact_id: number,
    confidence: number,
    iou: number,
    files: File[],
    request_id: string,
    signal: AbortSignal,
    run_id?: string | null,
    rerun_of_test_id?: string | null,
  ) => requestWithTimeout<ModelTest>('/api/model-tests', {
    method: 'POST',
    body: JSON.stringify({
      artifact_id,
      request_id,
      confidence,
      iou,
      run_id: run_id || undefined,
      rerun_of_test_id: rerun_of_test_id || undefined,
      files: files.map(({ name, size }) => ({ name, size })),
    }),
    signal,
  }),
  createStandaloneModelTest: (
    model_asset_id: string,
    confidence: number,
    iou: number,
    files: File[],
    request_id: string,
    signal: AbortSignal,
    rerun_of_test_id?: string | null,
  ) => requestWithTimeout<ModelTest>('/api/model-tests', {
    method: 'POST',
    body: JSON.stringify({
      model_asset_id,
      request_id,
      confidence,
      iou,
      rerun_of_test_id: rerun_of_test_id || undefined,
      files: files.map(({ name, size }) => ({ name, size })),
    }),
    signal,
  }),
  modelAssets: (signal?: AbortSignal) => requestWithTimeout<ModelAssetPage>('/api/model-assets', { signal }),
  removeModelAsset: (id: string, signal?: AbortSignal) => requestWithTimeout<void>(
    `/api/model-assets/${encodeURIComponent(id)}`, { method: 'DELETE', signal }, 30_000,
    '移除请求超时，后台可能已完成。请关闭弹窗并刷新模型库确认状态。',
  ),
  createLocalModelAsset: (path: string, name: string | undefined, signal?: AbortSignal) => requestWithTimeout<ModelAsset>('/api/model-assets/local', {
    method: 'POST', body: JSON.stringify({ path, name: name || undefined }), signal,
  }, 120_000, '校验模型超时，后台可能已经受理。请先刷新模型库确认状态，再决定是否重试。'),
  createRemoteModelAsset: (execution_node_id: string, path: string, name: string | undefined, signal?: AbortSignal) => requestWithTimeout<ModelAsset>('/api/model-assets/remote', {
    method: 'POST', body: JSON.stringify({ execution_node_id, path, name: name || undefined }), signal,
  }, 120_000, '远程模型登记超时，可刷新已登记模型确认状态'),
  prepareModelAsset: (id: string, retry = false, signal?: AbortSignal) => requestWithTimeout<ModelAsset>(
    `/api/model-assets/${encodeURIComponent(id)}/prepare?retry=${retry}`, { method: 'POST', signal }, 120_000,
    '模型准备超时，请稍后刷新确认状态',
  ),
  globalModelTests: (signal?: AbortSignal, cursor?: string, limit = 50, sourceType?: ModelAssetSourceType | 'training_artifact' | '', status?: ModelTest['status'] | '') => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (cursor) params.set('cursor', cursor)
    if (sourceType) params.set('source_type', sourceType)
    if (status) params.set('status', status)
    return requestWithTimeout<ModelTestPage>(`/api/model-tests?${params}`, { signal })
  },
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
    requestWithTimeout<ModelTest>(`/api/model-tests/${id}/abandon`, { method: 'POST', keepalive }),
  modelTest: (id: string, signal: AbortSignal) => requestWithTimeout<ModelTest>(`/api/model-tests/${id}`, { signal }),
  deleteModelTest: (id: string) => requestWithTimeout<void>(`/api/model-tests/${id}`, { method: 'DELETE' }),
  createDatasetUpload: (name: string, files: Array<{ path: string; size: number; sha256?: string }>, signal: AbortSignal, location_id?: string) =>
    requestWithTimeout<{ upload_id: string; destination: string }>('/api/datasets/uploads', { method: 'POST', body: JSON.stringify({ name, files, location_id }), signal }),
  datasetUploadStatus: (id: string, signal: AbortSignal) =>
    requestWithTimeout<{ uploaded_indices: number[]; result?: DatasetUploadResult; operation?: DatasetOperation }>(`/api/datasets/uploads/${id}`, { signal }),
  completeDatasetUpload: (id: string, names: string[], signal: AbortSignal) =>
    requestWithTimeout<DatasetUploadResult | { operation: DatasetOperation }>(`/api/datasets/uploads/${id}/complete`, { method: 'POST', body: JSON.stringify({ names }), signal }, 120_000, '数据集校验提交超时，后台可能已完成。请查看上传状态或数据集操作记录后继续。'),
  cancelDatasetUpload: (id: string) => requestWithTimeout<void>(`/api/datasets/uploads/${id}`, { method: 'DELETE' }),
  datasetScans: (signal?: AbortSignal) => requestWithTimeout<DatasetScan[]>('/api/dataset-storage/discovery/scans', { signal }),
  createDatasetScan: (payload: DatasetScanRequest) => requestWithTimeout<DatasetScan>('/api/dataset-storage/discovery/scans', { method: 'POST', body: JSON.stringify(payload) }),
  cancelDatasetScan: (id: string) => requestWithTimeout<DatasetScan>(`/api/dataset-storage/discovery/scans/${id}/cancel`, { method: 'POST' }),
  datasetScan: (id: string, signal?: AbortSignal) => requestWithTimeout<DatasetScan>(`/api/dataset-storage/discovery/scans/${id}`, { signal }),
  datasetCandidates: (id: string, params: URLSearchParams, signal?: AbortSignal) => requestWithTimeout<DatasetCandidatePage>(`/api/dataset-storage/discovery/scans/${id}/candidates?${params}`, { signal }),
  datasetRegistrationBatches: (id: string, signal?: AbortSignal) => requestWithTimeout<DatasetRegistrationBatch[]>(`/api/dataset-storage/discovery/scans/${id}/batches`, { signal }),
  registerDatasetBatch: (id: string, payload: DatasetBatchRequest) => requestWithTimeout<DatasetRegistrationBatch>(`/api/dataset-storage/discovery/scans/${id}/batches`, { method: 'POST', body: JSON.stringify(payload) }),
  cancelDatasetBatch: (id: string) => requestWithTimeout<DatasetRegistrationBatch>(`/api/dataset-storage/discovery/batches/${id}/cancel`, { method: 'POST' }),
  datasetLocations: (signal?: AbortSignal) => requestWithTimeout<DatasetLocation[]>('/api/dataset-storage/locations', { signal }),
  createDatasetLocation: (payload: { node_id: string; name: string; path: string; create: boolean; is_default: boolean }) => requestWithTimeout<{ id: string; operation_id: string }>('/api/dataset-storage/locations', { method: 'POST', body: JSON.stringify(payload) }, 15_000, datasetSubmissionTimeout),
  updateDatasetLocation: (id: string, enabled: boolean, is_default: boolean) => requestWithTimeout(`/api/dataset-storage/locations/${id}`, { method: 'PATCH', body: JSON.stringify({ enabled, is_default }) }, 15_000, datasetSubmissionTimeout),
  checkDatasetLocation: (id: string, create = false) => requestWithTimeout<DatasetOperation>(`/api/dataset-storage/locations/${id}/check`, { method: 'POST', body: JSON.stringify({ create }) }, 15_000, datasetSubmissionTimeout),
  forgetDatasetLocation: (id: string) => requestWithTimeout<void>(`/api/dataset-storage/locations/${id}`, { method: 'DELETE' }, 15_000, datasetSubmissionTimeout),
  datasetEntries: (signal?: AbortSignal) => requestWithTimeout<ManagedDataset[]>('/api/dataset-storage/datasets', { signal }, 15_000, '数据集列表读取超时，请检查网络后刷新'),
  refreshDatasetEntry: (id: string, request_id?: string) => requestWithTimeout<DatasetOperation>(`/api/dataset-storage/datasets/${encodeURIComponent(id)}/refresh`, { method: 'POST', body: JSON.stringify({ request_id }) }, 15_000, datasetSubmissionTimeout),
  registerDataset: (location_id: string, name: string, ownership: string, acknowledge: boolean) => requestWithTimeout<DatasetOperation>('/api/dataset-storage/datasets/register', { method: 'POST', body: JSON.stringify({ location_id, name, ownership, acknowledge }) }, 15_000, datasetSubmissionTimeout),
  migrateDataset: (id: string, location_id: string, name: string, remove_source: boolean) => requestWithTimeout<DatasetOperation>(`/api/dataset-storage/datasets/${id}/migrate`, { method: 'POST', body: JSON.stringify({ location_id, name, remove_source }) }, 15_000, datasetSubmissionTimeout),
  adoptDataset: (id: string, confirm_path: string) => requestWithTimeout(`/api/dataset-storage/datasets/${id}/adopt`, { method: 'POST', body: JSON.stringify({ confirm_path }) }, 15_000, datasetSubmissionTimeout),
  datasetDeletePreview: (id: string, signal?: AbortSignal) => requestWithTimeout<ManagedDataset & { can_delete: boolean }>(`/api/dataset-storage/datasets/${id}/delete-preview`, { signal }),
  deleteDatasetEntry: (id: string) => requestWithTimeout<DatasetOperation>(`/api/dataset-storage/datasets/${id}`, { method: 'DELETE' }, 30_000, datasetSubmissionTimeout),
  unregisterDataset: (id: string) => requestWithTimeout<void>(`/api/dataset-storage/datasets/${encodeURIComponent(id)}/registration`, { method: 'DELETE' }, 15_000, '删除登记请求超时，请刷新列表确认；原始文件会保留。'),
  previewDatasetHistoryRemoval: (payload: DatasetHistoryRemoval, signal?: AbortSignal) => requestWithTimeout<DatasetHistoryRemovalPreview>('/api/dataset-storage/operations/history-removal-preview', { method: 'POST', body: JSON.stringify(payload), signal }),
  removeDatasetHistory: (payload: DatasetHistoryRemoval) => requestWithTimeout<{ removed: number }>('/api/dataset-storage/operations/history-removal', { method: 'POST', body: JSON.stringify(payload) }),
  datasetOperationSummary: (signal?: AbortSignal) => requestWithTimeout<DatasetOperationSummary>('/api/dataset-storage/operations/summary', { signal }),
  datasetOperationPage: (params: URLSearchParams, signal?: AbortSignal) => requestWithTimeout<DatasetOperationPage>(`/api/dataset-storage/operations/page?${params}`, { signal }),
  datasetOperations: () => requestWithTimeout<DatasetOperation[]>('/api/dataset-storage/operations'),
  datasetOperation: (id: string, signal?: AbortSignal) => requestWithTimeout<DatasetOperation>(`/api/dataset-storage/operations/${id}`, { signal }),
  controlDatasetOperation: (id: string, action: 'retry' | 'cancel') => requestWithTimeout<DatasetOperation>(`/api/dataset-storage/operations/${id}/${action}`, { method: 'POST' }, 15_000, datasetSubmissionTimeout),
  managedDatasets: (signal?: AbortSignal) => requestWithTimeout<ManagedDataset[]>('/api/datasets/managed', { signal }),
  deleteManagedDataset: (name: string) => requestWithTimeout<void>(`/api/datasets/managed/${encodeURIComponent(name)}`, { method: 'DELETE' }, 30_000,
    '删除请求超时，请刷新数据集列表确认状态'),
  tasks: (signal?: AbortSignal) => requestWithTimeout<TrainingTask[]>('/api/tasks', { signal }),
  taskGroups: (signal?: AbortSignal) => requestWithTimeout<TaskGroup[]>('/api/task-groups', { signal }),
  createTaskGroup: (name: string) => requestWithTimeout<TaskGroup>('/api/task-groups', { method: 'POST', body: JSON.stringify({ name }) }),
  createTaskGroupFromTasks: (name: string, task_ids: string[]) => requestWithTimeout<TaskGroup>('/api/task-groups/from-tasks', { method: 'POST', body: JSON.stringify({ name, task_ids }) }),
  renameTaskGroup: (id: string, name: string) => requestWithTimeout<TaskGroup>(`/api/task-groups/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify({ name }) }),
  deleteTaskGroup: (id: string) => requestWithTimeout<void>(`/api/task-groups/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  assignTaskGroup: (id: string, group_id: string | null) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/group`, { method: 'PATCH', body: JSON.stringify({ group_id }) }),
  reorderTasks: (task_ids: string[]) => requestWithTimeout<void>('/api/tasks/order', { method: 'PATCH', body: JSON.stringify({ task_ids }) }),
  deletedTasks: (signal?: AbortSignal) => requestWithTimeout<TrainingTask[]>('/api/tasks?deleted_only=true', { signal }).then((tasks) => tasks.filter((task) => task.deleted_at)),
  taskDeletionJobs: (signal?: AbortSignal) => requestWithTimeout<DeletionJob[]>('/api/task-deletions?limit=50', { signal }),
  attemptDeletionJobs: (signal?: AbortSignal, taskId?: string) => requestWithTimeout<import('./types').AttemptDeletionJob[]>(`/api/attempt-deletions?limit=50${taskId ? `&task_id=${encodeURIComponent(taskId)}` : ''}`, { signal }),
  taskDeletionPreview: (id: string, signal?: AbortSignal) => requestWithTimeout<TaskDeletionPreview>(
    `/api/tasks/${encodeURIComponent(id)}/deletion-preview`, { signal }, 15_000, '删除范围读取超时，请重试',
  ),
  task: (id: string, signal?: AbortSignal) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}`, { signal }),
  create: (payload: TaskCreatePayload) => requestWithTimeout<TrainingTask>('/api/tasks', { method: 'POST', body: JSON.stringify(payload) }),
  start: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/start`, { method: 'POST' }),
  resume: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/resume`, { method: 'POST' }),
  retry: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/retry`, { method: 'POST' }),
  stop: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/stop`, { method: 'POST' }),
  forceInterrupt: (id: string, runId: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/force-interrupt`, {
    method: 'POST', body: JSON.stringify({ run_id: runId, confirm_process_stopped: true }),
  }),
  clone: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/clone`, { method: 'POST' }),
  rename: (id: string, name: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/name`, { method: 'PATCH', body: JSON.stringify({ name }) }),
  remove: (id: string) => requestWithTimeout<void>(`/api/tasks/${id}/trash`, { method: 'POST' }, 30_000, deletionTimeout),
  restore: (id: string) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/restore`, { method: 'POST' }, 30_000,
    '恢复请求超时，后台可能已经受理。请刷新任务列表和回收站确认状态。'),
  permanentRemove: (id: string) => requestWithTimeout<{ status: string; operation_id: string; message: string }>(`/api/tasks/${id}/permanent`, { method: 'DELETE' }, 30_000, deletionTimeout),
  prepareArtifact: (id: number, retry = false, signal?: AbortSignal) => requestWithTimeout<{ status: string; message?: string }>(`/api/artifacts/${id}/prepare?retry=${retry}`, { method: 'POST', signal }),
  saveLayout: (id: string, layout: TrainingTask['canvas_layout'], signal?: AbortSignal) =>
    requestWithTimeout<TrainingTask>(`/api/tasks/${id}/layout`, { method: 'PATCH', body: JSON.stringify({ layout }), signal }, 15_000,
      '保存布局超时，后台可能已保存，请刷新确认状态。'),
  updateConfig: (id: string, payload: {
    epochs: number; batch: number; imgsz: number; device: string; workers: number; optimizer: string; config: Record<string, unknown>; custom_config: Record<string, unknown>
  }) => requestWithTimeout<TrainingTask>(`/api/tasks/${id}/config`, { method: 'PATCH', body: JSON.stringify(payload) }),
  updateResources: (id: string, payload: { model_path?: string; data_path?: string }) =>
    requestWithTimeout<TrainingTask>(`/api/tasks/${id}/resources`, { method: 'PATCH', body: JSON.stringify(payload) }, 15_000,
      '保存资源路径超时，后台可能已经受理。请关闭窗口后重新打开确认。'),
  metrics: (id: string) => requestWithTimeout<MetricPoint[]>(`/api/tasks/${id}/metrics`),
  artifacts: (id: string) => requestWithTimeout<Artifact[]>(`/api/tasks/${id}/artifacts`),
  events: (id: string) => requestWithTimeout<TaskEvent[]>(`/api/tasks/${id}/events`),
  logs: (id: string) => requestWithTimeout<{ lines: string[]; path: string | null }>(`/api/tasks/${id}/logs?tail=240`),
  runs: (id: string, signal?: AbortSignal) => requestWithTimeout<TrainingRun[]>(`/api/tasks/${id}/runs`, { signal }),
  experiments: (offset: number, signal?: AbortSignal, runIds?: string[]) => requestWithTimeout<{
    items: Array<{ task: TrainingTask; run: TrainingRun }>; total: number; offset: number; limit: number
  }>(`/api/experiments?offset=${offset}&limit=200${runIds ? `&run_ids=${encodeURIComponent(runIds.join(','))}` : ''}`, { signal }),
  previewRunDeletion: (taskId: string, runId: string) => requestWithTimeout<import('./types').RunDeletionPreview>(`/api/tasks/${encodeURIComponent(taskId)}/runs/${encodeURIComponent(runId)}/deletion-preview`),
  deleteRun: (taskId: string, runId: string) => requestWithTimeout<{ status: 'pending' | 'running' | 'failed' | 'succeeded'; message: string; operation_id: string }>(
    `/api/tasks/${encodeURIComponent(taskId)}/runs/${encodeURIComponent(runId)}`,
    { method: 'DELETE' }, 30_000, '删除 Attempt 请求超时，后台可能已经受理，请稍后刷新确认状态',
  ),
  runMetrics: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<MetricPoint[]>(`/api/tasks/${id}/runs/${runId}/metrics`, { signal }),
  runArtifacts: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<Artifact[]>(`/api/tasks/${id}/runs/${runId}/artifacts`, { signal }),
  runEffectiveBestWeight: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<EffectiveBestWeight>(`/api/tasks/${id}/runs/${runId}/effective-best-weight`, { signal }),
  runEvents: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<TaskEvent[]>(`/api/tasks/${id}/runs/${runId}/events`, { signal }),
  runLogs: (id: string, runId: string, signal?: AbortSignal) => requestWithTimeout<{ lines: string[]; path: string | null }>(`/api/tasks/${id}/runs/${runId}/logs?tail=240`, { signal }),
  prepareRunLog: (id: string, runId: string, refresh: boolean, signal?: AbortSignal) => requestWithTimeout<{
    status: 'ready' | 'pending' | 'running' | 'finalizing' | 'failed'; message?: string; snapshot_at?: string
  }>(`/api/tasks/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}/logs/prepare?refresh=${refresh}`, { method: 'POST', signal }),
  trainingTrace: (id: string, runId: string) => requestWithTimeout<TrainingTrace>(`/api/tasks/${id}/runs/${runId}/trace`),
  resources: () => requestWithTimeout<Resources>('/api/resources'),
  compute: () => requestWithTimeout<ComputeResources>('/api/compute'),
  computeNodes: (signal?: AbortSignal) => requestWithTimeout<ComputeNode[]>('/api/compute/nodes', { signal }, 8_000, '算力节点状态读取超时'),
  computeNode: (id: string, signal?: AbortSignal) => requestWithTimeout<ComputeNode>(`/api/compute/nodes/${id}`, { signal }),
  refreshComputeNodes: (signal?: AbortSignal) => requestWithTimeout<ComputeNodeRefreshResponse>(
    '/api/compute/nodes/refresh', { method: 'POST', signal }, 45_000, '等待新采样超时，请稍后查看各节点的采样时间',
  ),
  createComputeNode: (payload: ComputeNodeSettings) => requestWithTimeout<ComputeNodeCredential>('/api/compute/nodes', { method: 'POST', body: JSON.stringify(payload) }),
  updateComputeNode: (id: string, payload: ComputeNodeSettings) => requestWithTimeout<ComputeNode>(`/api/compute/nodes/${id}`, { method: 'PATCH', body: JSON.stringify(payload) }),
  rotateComputeToken: (id: string) => requestWithTimeout<ComputeNodeCredential>(`/api/compute/nodes/${id}/token`, { method: 'POST' }),
  deleteComputeNode: (id: string) => requestWithTimeout<void>(`/api/compute/nodes/${id}`, { method: 'DELETE' }),
  trainingParameters: () => requestWithTimeout<{ source: string; parameters: TrainingParameterDefinition[] }>('/api/training-parameters'),
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
    if (typeof value === 'number' && Number.isFinite(value)) return value.toFixed(4)
  }
  return '—'
}

export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`
}
