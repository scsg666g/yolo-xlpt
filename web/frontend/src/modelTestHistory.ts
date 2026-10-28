import type { ModelTestSummary } from './types'

/** Refresh the visible head without discarding already loaded cursor pages. */
export function mergeHistoryHead<T extends { id: string }>(current: readonly T[], incoming: readonly T[]): T[] {
  const ids = new Set(incoming.map((item) => item.id))
  return [...incoming, ...current.filter((item) => !ids.has(item.id))]
}

export function retainedHistorySelection(current: string | null, items: readonly { id: string }[]): string | null {
  return current && items.some((item) => item.id === current) ? current : items[0]?.id || null
}

export function historyNeedsDetailRefresh(summary: Pick<ModelTestSummary, 'status' | 'finished_at'>, detail: Pick<ModelTestSummary, 'status' | 'finished_at'> | null): boolean {
  return !detail || summary.status !== detail.status || summary.finished_at !== detail.finished_at
}

/** Recheck older active rows omitted from the first cursor page, with bounded concurrency. */
export async function refreshActiveHistoryTail<T extends { id: string; status: string }>(
  current: readonly T[], head: readonly T[], activeStatuses: ReadonlySet<string>,
  fetchItem: (id: string) => Promise<T>, start = 0,
): Promise<{ updates: T[]; removed: string[]; next: number }> {
  const headIds = new Set(head.map((item) => item.id))
  const candidates = current.filter((item) => !headIds.has(item.id) && activeStatuses.has(item.status))
  if (!candidates.length) return { updates: [], removed: [], next: 0 }
  const offset = start % candidates.length
  const batch = Array.from({ length: Math.min(8, candidates.length) }, (_, index) => candidates[(offset + index) % candidates.length])
  const updates: T[] = []
  const removed: string[] = []
  // At most four requests in flight, and at most eight per refresh.
  for (let index = 0; index < batch.length; index += 4) {
    const results = await Promise.allSettled(batch.slice(index, index + 4).map((item) => fetchItem(item.id)))
    results.forEach((result, itemIndex) => {
      if (result.status === 'fulfilled') updates.push(result.value)
      else if ((result.reason as { status?: number })?.status === 404) removed.push(batch[index + itemIndex].id)
    })
  }
  return { updates, removed, next: (offset + batch.length) % candidates.length }
}

export function applyHistoryRefresh<T extends { id: string; status: string }>(current: readonly T[], head: readonly T[],
  tail: { updates: readonly T[]; removed: readonly string[] }, statusFilter = ''): T[] {
  const updated = new Map(tail.updates.map((item) => [item.id, item]))
  const removed = new Set(tail.removed)
  return mergeHistoryHead(current.map((item) => updated.get(item.id) || item), head)
    .filter((item) => !removed.has(item.id) && (!statusFilter || item.status === statusFilter))
}

/** Backfill new records between a refreshed head and the old loaded window. */
export async function bridgeHistoryHead<T extends { id: string }>(current: readonly T[],
  first: { items: T[]; next_cursor: string | null },
  fetchPage: (cursor: string) => Promise<{ items: T[]; next_cursor: string | null }>,
): Promise<T[]> {
  const known = new Set(current.map((item) => item.id))
  if (!known.size) return first.items
  let page = first
  const items = [...page.items]
  const cursors = new Set<string>()
  while (page.next_cursor && !page.items.some((item) => known.has(item.id)) && !cursors.has(page.next_cursor)) {
    cursors.add(page.next_cursor)
    page = await fetchPage(page.next_cursor)
    items.push(...page.items)
  }
  return [...new Map(items.map((item) => [item.id, item])).values()]
}
