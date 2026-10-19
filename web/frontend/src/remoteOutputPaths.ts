import type { ComputeNode } from './types'

export function remoteOutputRoots(node?: ComputeNode): string[] {
  return [...new Set((node?.training?.output_roots || []).filter((path) => path.startsWith('/')))]
}

function normalizedPath(value: string): string | null {
  if (!value.startsWith('/') || value.includes('\0')) return null
  const parts: string[] = []
  for (const part of value.split('/')) {
    if (part === '..') parts.pop()
    else if (part && part !== '.') parts.push(part)
  }
  return `/${parts.join('/')}`
}

// Browser validation is an early hint. The Agent also resolves symlinks and
// checks the actual remote filesystem before accepting a training request.
export function remoteOutputPathError(value: string, roots: string[]): string | null {
  if (!value.trim() || !roots.length) return null
  const path = normalizedPath(value.trim())
  if (!path) return '请填写远程服务器上的绝对路径，或从允许目录中选择。'
  const allowed = roots.some((root) => {
    const normalized = normalizedPath(root)
    return normalized !== null && (normalized === '/' || path === normalized || path.startsWith(`${normalized}/`))
  })
  return allowed ? null : '此输出目录不在当前 Agent 允许范围内。请选择下方允许目录，或填写其中已存在的子目录。'
}
