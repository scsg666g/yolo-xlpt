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
  scheduler: 'running' | 'stopped'
  local_deletion_worker: 'running' | 'stopped'
  inference_worker: 'running' | 'stopped'
  database: 'ok'
  database_latency_ms: number
  runs_root: string
}

export interface TrainingTask {
  id: string
  name: string
  task_type: string
  status: TaskStatus
  source: string
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

export interface TrainingRun {
  id: string
  task_id: string
  attempt_number: number
  launch_mode: 'fresh' | 'resume' | 'retry'
  status: TaskStatus
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
  training: {
    enabled: boolean
    ready: boolean
    reason: string | null
    supported_parameters: string[]
    yolo_version: string | null
    python_version: string | null
    output_writable: boolean
    lock_occupied: boolean
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
export interface ModelTest {
  id: string
  status: 'uploading' | 'preparing' | 'queued' | 'running' | 'cancel_requested' | 'completed' | 'failed' | 'cancelled'
  task_id?: string
  run_id?: string | null
  source_run_id?: string | null
  artifact_id?: number | null
  artifact_name?: string | null
  model_name?: string | null
  model_sha256?: string | null
  model_size?: number | null
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
    attempt_number: number
    best_epoch: number | null
    metric_name: string
    metric_value: number | null
    device: string
    imgsz: number
  }
  timings: { model_load: number; preprocess: number; inference: number; postprocess: number; total: number } | null
  files: Array<{
    name: string
    size: number
    uploaded: boolean
    detections: number | null
    objects: Array<{ class_id: number; class_name: string; confidence: number; bbox_xyxy: number[] }>
    timings: { preprocess?: number; inference?: number; postprocess?: number }
  }>
}

export interface ModelTestSummary extends Omit<ModelTest, 'timings' | 'files'> {
  files: Array<Pick<ModelTest['files'][number], 'name' | 'size' | 'uploaded' | 'detections'>>
}

export interface ModelTestPage {
  items: ModelTestSummary[]
  next_cursor: string | null
}
