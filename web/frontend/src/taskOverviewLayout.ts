import { sortOverviewTasks } from './taskOrder'

type Task = { id: string; group_id?: string | null; created_at?: string | null }
type Group = { id: string; name: string }

/** One column per nonempty group; derive rows afresh so removed tasks leave no gaps. */
export function overviewGroupLayout<T extends Task>(tasks: readonly T[], groups: readonly Group[]) {
  const known = new Map(groups.map((group) => [group.id, group]))
  const columns = new Map<string | null, T[]>()
  for (const task of sortOverviewTasks(tasks, groups)) {
    const key = task.group_id && known.has(task.group_id) ? task.group_id : null
    const column = columns.get(key) || []
    column.push(task)
    columns.set(key, column)
  }
  return [...columns].flatMap(([groupId, column], columnIndex) => column.map((task, row) => ({
    task,
    position: { x: columnIndex * 390, y: 48 + row * 268 },
    groupName: groupId ? known.get(groupId)!.name : '未分组',
    groupCount: column.length,
    firstInGroup: row === 0,
    previousId: row > 0 ? column[row - 1].id : null,
  })))
}
