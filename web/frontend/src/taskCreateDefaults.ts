import type { Resources, TaskCreatePayload } from './types'

export type ResourceFieldEdits = Record<'model_path' | 'data_path' | 'device', boolean>

export function defaultLocalDevice(resources: Resources | null): string {
  return String(resources?.gpus[0]?.index ?? 'cpu')
}

export function applyLocalResourceDefaults(
  form: TaskCreatePayload,
  resources: Resources,
  edited: ResourceFieldEdits,
): TaskCreatePayload {
  // A late local response must never overwrite fields belonging to a remote node.
  if (form.execution_node_id) return form
  const datasets = form.task_type === 'classify' ? resources.classification_datasets : resources.datasets
  return {
    ...form,
    model_path: edited.model_path || form.model_path ? form.model_path : resources.models[0]?.path || '',
    data_path: edited.data_path || form.data_path ? form.data_path : datasets[0]?.path || '',
    device: edited.device ? form.device : defaultLocalDevice(resources),
  }
}
