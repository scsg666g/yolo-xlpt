export function trainingDataLabel(taskType: string) {
  return taskType === 'classify' ? '分类数据集目录' : '数据集 YAML'
}

export function remoteTrainingDataPathError(taskType: string, value: string) {
  const path = value.trim()
  if (!path) return null
  if (!path.startsWith('/')) return `远程${trainingDataLabel(taskType)}必须是绝对路径`
  const yaml = /\.ya?ml$/i.test(path)
  if (taskType === 'classify' && yaml) return '图像分类任务需要数据集目录，不能使用 YAML 文件'
  if (taskType !== 'classify' && !yaml) return '非分类任务需要 .yaml 或 .yml 数据集配置'
  return null
}
