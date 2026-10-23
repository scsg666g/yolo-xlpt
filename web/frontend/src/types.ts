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
  error_message: string | null
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

export interface TaskGroup {
  id: string
  name: string
  display_order: number
  created_at: string
  updated_at: string
}

export interface TrainingRun {
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
  checkpoint_path: string | null
  run_dir: string | null
  log_path: string | null
  completed_epochs: number
  progress: number
  latest_metrics: Record<string, number | string | null>
  exit_code: number | null
  error_message: string | null
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

export interface Resources {
  gpus: GpuInfo[]
  models: ResourceFile[]
  datasets: ResourceFile[]
}

export interface DatasetUploadResult {
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
