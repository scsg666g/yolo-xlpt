type GroupedTask = { id: string; group_id?: string | null }

type CreatedTask = GroupedTask & { created_at?: string | null }

function creationTime(value: string | null | undefined): number {
  if (!value) return Number.NEGATIVE_INFINITY
  // Historical database timestamps omit the UTC suffix; explicit offsets must
  // still describe the same instant on clients in any time zone.
  const normalized = value.trim().replace(' ', 'T')
  const timestamp = Date.parse(/(?:Z|[+-]\d{2}:?\d{2})$/i.test(normalized) ? normalized : normalized + 'Z')
  return Number.isFinite(timestamp) ? timestamp : Number.NEGATIVE_INFINITY
}

/** Keep the dashboard's group order, newest tasks first, and ungrouped tasks last. */
export function sortOverviewTasks<T extends CreatedTask>(tasks: readonly T[], groups: readonly { id: string }[]): T[] {
  const groupRanks = new Map(groups.map((group, index) => [group.id, index]))
  const rank = (task: T) => groupRanks.get(task.group_id || '') ?? groups.length
  const timestamps = new Map(tasks.map((task) => [task.id, creationTime(task.created_at)]))
  return [...tasks].sort((left, right) => rank(left) - rank(right)
    || timestamps.get(right.id)! - timestamps.get(left.id)!
    || left.id.localeCompare(right.id))
}

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
