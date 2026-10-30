type GroupedTask = { id: string; group_id?: string | null }

export function reorderVisibleGroupTasks<T extends GroupedTask>(tasks: T[], visibleTasks: T[], knownGroupIds: string[], sourceId: string, targetId: string, after: boolean): T[] {
  if (sourceId === targetId) return tasks
  const known = new Set(knownGroupIds)
  const groupKey = (task: T) => task.group_id && known.has(task.group_id) ? task.group_id : null
  const byId = new Map(tasks.map((task) => [task.id, task]))
  const source = byId.get(sourceId)
  const target = byId.get(targetId)
  if (!source || !target || groupKey(source) !== groupKey(target)) return tasks
  const visibleIds = new Set(visibleTasks.map((task) => task.id))
  const groupIds = tasks.filter((task) => visibleIds.has(task.id) && groupKey(task) === groupKey(source)).map((task) => task.id)
  if (!groupIds.includes(sourceId) || !groupIds.includes(targetId)) return tasks
  const nextIds = groupIds.filter((id) => id !== sourceId)
  const targetIndex = nextIds.indexOf(targetId)
  nextIds.splice(targetIndex + (after ? 1 : 0), 0, sourceId)
  if (nextIds.every((id, index) => id === groupIds[index])) return tasks
  let cursor = 0
  return tasks.map((task) => visibleIds.has(task.id) && groupKey(task) === groupKey(source) ? byId.get(nextIds[cursor++])! : task)
}
