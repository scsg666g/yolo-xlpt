/** Format only evidence recorded for this run; legacy runs remain unknown. */
export function localDeviceBindingLabel(parameters?: Record<string, unknown>): string | null {
  const raw = parameters?.local_device_binding
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null
  const binding = raw as Record<string, unknown>
  if (binding.version !== 1 || typeof binding.requested_device !== 'string'
    || typeof binding.cuda_visible_devices !== 'string' || binding.cuda_visible_devices.length > 4096
    || !Array.isArray(binding.devices) || !binding.devices.length || binding.devices.length > 128) return null
  const devices: Array<{ physical: number; logical: number; uuid: string }> = []
  for (const value of binding.devices) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null
    const item = value as Record<string, unknown>
    if (!Number.isSafeInteger(item.physical_index) || Number(item.physical_index) < 0
      || !Number.isSafeInteger(item.logical_index) || Number(item.logical_index) < 0
      || typeof item.uuid !== 'string' || !/^GPU-[a-z0-9-]+$/i.test(item.uuid)) return null
    devices.push({ physical: Number(item.physical_index), logical: Number(item.logical_index), uuid: item.uuid })
  }
  if (new Set(devices.map((item) => item.physical)).size !== devices.length
    || devices.some((item, index) => item.logical !== index)
    || binding.requested_device.split(',').map((value) => value.trim()).join(',') !== devices.map((item) => item.physical).join(',')
    || binding.cuda_visible_devices !== devices.map((item) => item.uuid).join(',')) return null
  const mapping = devices.map((item) => `物理 GPU ${item.physical} → 训练逻辑 GPU ${item.logical}`).join('；')
  const order = typeof binding.cuda_device_order === 'string' && binding.cuda_device_order.length <= 100 ? `；CUDA_DEVICE_ORDER=${binding.cuda_device_order}` : ''
  return `${mapping}；CUDA_VISIBLE_DEVICES=${binding.cuda_visible_devices}${order}`
}
