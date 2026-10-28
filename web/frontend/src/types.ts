export type TaskStatus =
  | 'draft'
  | 'queued'
  | 'starting'
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'completed'
  | 'failed'
  | 'interrupted'

export interface TrainingCompletionInfo {
  reason?: 'early_stopping' | 'epochs_completed' | 'training_completed' | 'exception'
  source?: string
  best_epoch?: number
  patience?: number
  expected_epochs?: number
  final_metrics?: Record<string, number | string | null>
}

export interface PlatformHealth {
  status: 'ok' | 'degraded'
  instance: string
  version: string
  scheduler: 'running' | 'stopped' | 'error'
  scheduler_error?: string | null
  scheduler_last_error?: string | null
  scheduler_last_error_at?: string | null
  local_deletion_worker: 'running' | 'stopped'
  inference_worker: 'running' | 'stopped'
  database: 'ok'
  database_latency_ms: number
  runs_root: string
}

export interface PlatformStorageHealth {
  path: string
  status: 'writable' | 'read_only' | 'missing' | 'not_directory' | 'unavailable'
  free_bytes: number | null
  total_bytes: number | null
}

export interface TrainingTask {
  id: string
  name: string
  task_type: string
  status: TaskStatus
  source: string
  group_id: string | null
  execution_node_id: string | null
  model_path: string
  data_path: string
  run_dir: string | null
  log_path: string | null
  epochs: number
  completed_epochs: number
  batch: number
  imgsz: number
  device: string
  workers: number
  optimizer: string
  progress: number
  pid: number | null
  exit_code: number | null
  queue_reason?: string | null
  error_message: string | null
  completion_info?: TrainingCompletionInfo | null
  config: Record<string, unknown>
  latest_metrics: Record<string, number | string | null>
  canvas_layout: { nodes?: Array<{ id: string; position: { x: number; y: number } }>; viewport?: { x: number; y: number; zoom: number } }
  created_at: string
  queued_at: string | null
  started_at: string | null
  finished_at: string | null
  updated_at: string
  display_order: number
  current_run_id: string | null
  deleted_at: string | null
  deleted_reason: string | null
}

export type DeletionJobStatus = 'pending' | 'running' | 'finalizing' | 'failed' | 'cancelled' | 'succeeded'

export type DeletionJobPhase =
  | 'stopping_tests'
  | 'validating'
  | 'queued'
  | 'waiting_agent'
  | 'agent_upgrade_required'
  | 'deleting_files'
  | 'verifying'
  | 'center_cleanup'
  | 'completed'
  | 'failed'

export interface DeletionJob {
  id: string
  task_id: string
  task_name: string
  executor: 'local' | 'remote'
  node_id: string | null
  node_name: string | null
  node_status: 'online' | 'offline' | 'pending' | 'disabled' | null
  last_seen_at: string | null
  agent_version: string | null
  required_agent_version: string | null
  status: DeletionJobStatus
  phase: DeletionJobPhase
  message: string
  error: string | null
  retryable: boolean
  updated_at: string
  created_at: string
  receipt: Record<string, unknown> | null
}

export interface AttemptDeletionJob extends DeletionJob {
  scope: 'attempt'
  run_id: string
  attempt_number: number
}

export interface TaskDeletionPreview {
  task_id: string
  task_name: string
  executor: 'local' | 'remote'
  node_name: string | null
  attempt_count: number
  test_count: number
  can_delete: boolean
  blocking_reason: string | null
  delete_items: string[]
  keep_items: string[]
}

export interface TaskGroup {
  id: string
  name: string
  display_order: number
  created_at: string
  updated_at: string
}

export interface TrainingRun {
  monitoring_status?: 'local' | 'online' | 'reconnecting' | 'unavailable' | null
  monitoring_last_seen_at?: string | null
  id: string
  task_id: string
  attempt_number: number
  launch_mode: 'fresh' | 'resume' | 'retry'
  status: TaskStatus
  execution_node_id: string | null
  pid: number | null
  process_started_at: string | null
  command: string[]
  parameter_snapshot: Record<string, unknown>
  model_path: string
  data_path: string
  checkpoint_source?: { run_id?: string; attempt_number?: number; checkpoint_path?: string; source_deleted?: boolean }
  checkpoint_path: string | null
  run_dir: string | null
  log_path: string | null
  completed_epochs: number
  progress: number
  latest_metrics: Record<string, number | string | null>
  exit_code: number | null
  queue_reason?: string | null
  error_message: string | null
  completion_info?: TrainingCompletionInfo | null
  queued_at: string | null
  started_at: string | null
  finished_at: string | null
  created_at: string
  updated_at: string
  /** Attempt 文件异步清理状态；旧 API 可能暂时不返回这些字段。 */
  deletion_status?: 'pending' | 'running' | 'failed' | 'succeeded' | null
  deletion_error?: string | null
  deletion_requested_at?: string | null
}

export interface MetricPoint {
  epoch: number
  elapsed: number
  values: Record<string, number | string | null>
  created_at: string
}

export interface Artifact {
  id: number
  kind: string
  name: string
  path: string
  size: number
  modified_at: string | null
}

export interface EffectiveBestWeight {
  artifact: Artifact | null
  source_run_id: string | null
  source_attempt_number: number | null
  inherited: boolean
}

export interface TaskEvent {
  id: number
  level: string
  event_type: string
  message: string
  payload: Record<string, unknown>
  created_at: string
}

export interface TrainingTraceSample {
  epoch: number
  batch: number
  images: { shape?: number[]; numel?: number; mean?: number; std?: number; min?: number; max?: number } | null
  labels: number | null
  loss: Record<string, number>
  learning_rate: number[]
  layers: Record<string, { shape?: number[]; numel?: number; mean?: number; std?: number; min?: number; max?: number }>
  updates: Record<string, { gradient_mean: number; gradient_max: number; weight_mean_before: number; weight_mean_after?: number; weight_delta_mean?: number }>
  image?: string
  image_url?: string
}

export interface TrainingTrace {
  available: boolean
  reason?: string
  source?: string
  updated_at?: number
  sample_count?: number
  maximum?: number
  interval?: number
  samples: TrainingTraceSample[]
}

export interface ResourceFile {
  name: string
  path: string
  size: number
  root: string
}

export interface GpuInfo {
  index: number
  uuid?: string
  name: string
  driver_version?: string
  memory_total: number | null
  memory_used: number | null
  memory_free?: number | null
  utilization: number | null
  memory_utilization?: number | null
  temperature: number | null
  power_draw?: number | null
  power_limit?: number | null
  graphics_clock?: number | null
  memory_clock?: number | null
  processes?: GpuProcess[]
}

export interface GpuProcess {
  pid: number
  name: string
  memory_used: number | null
  association?: {
    state: 'managed' | 'other_node' | 'other_platform' | 'unmanaged' | 'unknown' | 'ambiguous' | 'orphaned' | 'stale'
    last_state?: string
    task_id?: string
    task_name?: string
    run_id?: string
    attempt_number?: number
    root_pid?: number | null
    match_method?: 'runtime_marker' | 'ancestor' | 'launcher'
    reason?: string
    sampled_at?: string
  }
}

export interface ComputeResources {
  available: boolean
  provider: string | null
  message: string | null
  sampled_at: string
  host: {
    hostname: string
    platform: string
    cpu_count: number | null
    load_average: number[]
    memory_total: number | null
    memory_used: number | null
  }
  gpus: GpuInfo[]
}

export interface ComputeNodeSettings {
  name: string
  address: string
  notes: string
  enabled: boolean
  report_interval_seconds: number
}

export interface ComputeNode extends ComputeNodeSettings {
  id: string
  kind: 'local' | 'agent'
  status: 'online' | 'offline' | 'pending' | 'disabled'
  offline_after_seconds: number
  last_seen_at: string | null
  sampled_at: string | null
  agent_version: string | null
  required_agent_version: string
  training_version_supported?: boolean
  process_attribution_supported: boolean
  required_process_attribution_agent_version: string
  training_unavailable_reason: string | null
  training: {
    enabled: boolean
    ready: boolean
    reason: string | null
    supported_parameters: string[]
    yolo_version: string | null
    python_version: string | null
    output_writable: boolean
    output_roots?: string[]
    dataset_discovery?: boolean
    model_test_roots?: string[]
    lock_occupied: boolean
    gpu_parallelism?: boolean
  } | null
  hostname: string | null
  summary: {
    gpu_count: number
    utilization: number | null
    memory_used: number
    memory_total: number
    process_count: number
  } | null
  compute?: ComputeResources | null
}

export interface ComputeNodeCredential {
  node: ComputeNode
  token: string
}

export interface ComputeNodeRefreshResult {
  id: string
  result: 'fresh' | 'stale' | 'timeout' | 'unsupported' | 'error'
  reason?: string | null
  previous_sampled_at?: string | null
  sampled_at?: string | null
  node: ComputeNode
}

export interface ComputeNodeRefreshResponse {
  nodes: ComputeNodeRefreshResult[]
}

export interface Resources {
  gpus: GpuInfo[]
  models: ResourceFile[]
  datasets: ResourceFile[]
  classification_datasets: ResourceFile[]
}

export interface DatasetUploadResult {
  node_id?: string
  name: string
  directory: string
  names: string[]
  resource: ResourceFile
  stats: { train_images: number; val_images: number; labels: number; classes: number; background_images: number }
  warnings: string[]
}

export interface ManagedDatasetReference {
  id: string
  name: string
  status: string
  deleted: boolean
}

export interface ManagedDataset {
  checked_at?: string | null
  refresh_status?: string | null
  refresh_error?: string | null
  blocking_operation_kind?: string | null
  can_delete?: boolean
  blocking_operation_id?: string | null
  node_name?: string
  id: string
  node_id: string
  location_id: string
  ownership: string
  status: string
  history?: Array<{ node_id: string; path: string; operation_id: string }>
  name: string
  directory: string
  config_path: string
  created_at: string | null
  total_bytes: number
  stats: {
    train_images?: number
    val_images?: number
    labels?: number
    classes?: number
    background_images?: number
  }
  classes: string[]
  warnings: string[]
  references: ManagedDatasetReference[]
}

export interface TaskCreatePayload {
  name: string
  task_type: string
  group_id?: string | null
  model_path: string
  data_path: string
  execution_node_id?: string | null
  remote_output_root?: string
  epochs: number
  batch: number
  imgsz: number
  device: string
  workers: number
  optimizer: string
  patience: number
  lr0: number | null
  pretrained: boolean
  cache: boolean | string
  amp: boolean
  close_mosaic: number
  advanced: Record<string, unknown>
}

export interface TrainingParameterDefinition {
  key: string
  label: string
  type: 'number' | 'boolean' | 'text'
  default: string | number | boolean | null
  section: string
}

export type ModelAssetSourceType = 'local_path' | 'remote_path'
export type ModelAssetStatus = 'preparing' | 'ready' | 'failed'
export type YoloTaskType = 'detect' | 'segment' | 'classify' | 'pose' | 'obb'

export interface ModelPrediction {
  class_id?: number
  class_name: string
  confidence: number
  bbox_xyxy?: number[]
  mask_available?: boolean
  mask_polygon_xy?: number[][]
  keypoints?: Array<{ index: number; x: number; y: number; confidence: number | null }>
  obb_points?: number[][]
  obb_xywhr?: number[]
}

export interface ModelAsset {
  id: string
  name: string
  source_type: ModelAssetSourceType
  execution_node_id: string | null
  execution_node_name?: string | null
  path: string
  artifact_id?: number | null
  status: ModelAssetStatus
  error: string | null
  size: number | null
  sha256: string | null
  history_count?: number
  removal_blocked_reason?: string | null
  created_at: string
  updated_at: string
}

export interface ModelAssetPage {
  items: ModelAsset[]
  local_root?: string
  local_roots?: string[]
}

export interface ModelTest {
  id: string
  status: 'uploading' | 'preparing' | 'queued' | 'running' | 'cancel_requested' | 'completed' | 'failed' | 'cancelled'
  /** Execution service is distinct from the server that supplied the weights. */
  execution?: { node_id: string; device: string; location: 'center'; timeout_seconds: number }
  queue_ahead?: number | null
  status_message?: string | null
  task_id?: string | null
  run_id?: string | null
  source_run_id?: string | null
  artifact_id?: number | null
  model_asset_id?: string | null
  model_source_type?: 'training_artifact' | ModelAssetSourceType | null
  model_source_name?: string | null
  model_source_path?: string | null
  execution_node_id?: string | null
  execution_node_name?: string | null
  task_name?: string | null
  artifact_name?: string | null
  model_name?: string | null
  model_sha256?: string | null
  model_size?: number | null
  task_type?: YoloTaskType | null
  rerun_of_test_id?: string | null
  attempt_number?: number | null
  source_attempt_number?: number | null
  created_at?: string | null
  started_at?: string | null
  finished_at?: string | null
  storage_bytes?: number
  confidence: number
  iou: number
  error: string | null
  expires_at?: number | null
  model: {
    weight_name: string
    weight_size: number
    modified_at: string | null
    attempt_number: number | null
    best_epoch: number | null
    metric_name: string | null
    metric_value: number | null
    device: string
    imgsz: number
    source_type?: 'training_artifact' | ModelAssetSourceType
    source_name?: string | null
    source_path?: string | null
    execution_node_id?: string | null
    execution_node_name?: string | null
    task_name?: string | null
    task_type?: YoloTaskType | null
    sha256?: string | null
    history_group?: {
      kind: 'task_group' | 'ungrouped' | 'standalone_model'
      id: string | null
      name: string
      source_type?: ModelAssetSourceType | null
      execution_node_id?: string | null
      execution_node_name?: string | null
    }
  }
  timings: { model_load: number; preprocess: number; inference: number; postprocess: number; total: number } | null
  files: Array<{
    name: string
    size: number
    uploaded: boolean
    detections: number | null
    prediction_count?: number | null
    objects: ModelPrediction[]
    timings: { preprocess?: number; inference?: number; postprocess?: number }
  }>
}

export interface ModelTestSummary extends Omit<ModelTest, 'timings' | 'files'> {
  files: Array<Pick<ModelTest['files'][number], 'name' | 'size' | 'uploaded' | 'detections' | 'prediction_count'>>
}

export interface ModelTestPage {
  items: ModelTestSummary[]
  next_cursor: string | null
}

export interface RunDeletionPreview {
  allowed: boolean
  blockers: string[]
  paths: string[]
  indexed_bytes: number
  checkpoint_releases: Array<{ run_id: string; attempt_number: number }>
  note: string
}

export interface DatasetLocation {
  node_name?: string
  id: string; node_id: string; name: string; path: string; resolved_path: string | null
  enabled: boolean; is_default: boolean; status: string; updated_at: string
  details: { free_bytes?: number; total_bytes?: number; message?: string; checked_at?: string }
}
export interface DatasetOperation {
  progress?: { stage: string; phase: string; executor: string; processed: number; total: number | null; unit: 'files' | 'bytes' | 'directories'; current_file: string; sampled_at: string; changed_at: string; started_at: string; rate: number | null; eta_seconds: number | null } | null
  progress_steps?: Array<{ stage: string; phase: string; executor: string }>
  queued_at?: string | null; heartbeat_at?: string | null; started_at?: string | null; queue_ahead?: number; waiting_lock?: boolean
  can_delete_history?: boolean
  node_name?: string; source_node?: string; source_node_name?: string; target_node?: string; target_node_name?: string
  id: string; kind: string; status: string; phase: string; name: string; node_id: string
  transferred: number; total_bytes: number; error: string | null; can_cancel: boolean
  source_path?: string; destination?: string; waiting_offline?: boolean; waiting_training?: boolean
  created_at: string; updated_at: string; result: Partial<DatasetUploadResult> & { dataset_id?: string; visited?: number; found?: number; limited?: boolean; limit_reason?: string }
}

export interface DatasetOperationSummary {
  total: number; active: number; failed: number; revision: string; counts: Record<string, number>
  reference_revision?: string
  managed_task_types?: string[]
  latest_created_at: string | null; snapshot: string; nodes: Array<{ id: string; name: string }>
}
export interface DatasetOperationPage {
  items: DatasetOperation[]; total: number; page: number; page_size: number; snapshot: string
}

export interface DatasetHistoryRemoval {
  ids?: string[]; status?: string; kind?: string; node_id?: string; q?: string
  before: string; after?: string; fingerprint?: string
}
export interface DatasetHistoryRemovalPreview { count: number; fingerprint: string }

export interface DatasetScanRequest {
  request_id: string; location_ids: string[]; depth: number; max_candidates: number; max_directories: number; max_seconds: number
}
export interface DatasetScanProgress {
  visited?: number; found?: number; skipped_links?: number; skipped_depth?: number; limited?: boolean; limit_reason?: string; current_path?: string
}
export interface DatasetScan {
  id: string; specification: Omit<DatasetScanRequest, 'request_id'>; created_at: string; status: string
  operations: Array<DatasetOperation & { progress: DatasetScanProgress }>; counts: Record<string, number>
}
export interface DatasetCandidate {
  id: string; location_id: string; node_id: string; node_name: string; path: string; relative_path: string; name: string
  state: string; reason: string; task: string; classes: string[]; selection_token: string
}
export interface DatasetCandidatePage { items: DatasetCandidate[]; total: number; counts: Record<string, number>; page: number; page_size: number }
export interface DatasetBatchRequest { request_id: string; candidate_ids: string[]; candidate_tokens: Record<string, string>; ownership: 'external' | 'managed'; acknowledge: boolean }
export interface DatasetRegistrationBatch {
  id: string; session_id: string; created_at: string; status: string; counts: Record<string, number>
  items: Array<{ candidate_id: string; name: string; path: string; node_id: string; status: string; error?: string; operation_id?: string; phase?: string; can_cancel?: boolean }>
}
