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

export interface TaskEvent {
  id: number
  level: string
  event_type: string
  message: string
  payload: Record<string, unknown>
  created_at: string
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
