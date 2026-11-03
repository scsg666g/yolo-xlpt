// The cookie stays HttpOnly. CSRF material is memory-only and never persisted.
let csrfToken = ''
let generation = 0
export const AUTH_EXPIRED_EVENT = 'yolo-auth-expired'

export function setCsrfToken(token: string | null) {
  const next = token || ''
  if (next !== csrfToken) generation += 1
  csrfToken = next
}
export function authGeneration() { return generation }
export function csrfHeaders(): Record<string, string> {
  return csrfToken ? { 'X-CSRF-Token': csrfToken } : {}
}
export function notifyAuthExpired(message = '请重新登录；后台训练会继续运行。', expectedGeneration?: number) {
  // A slow 401 from the previous session must not dismiss a fresh login.
  if (expectedGeneration !== undefined && expectedGeneration !== generation) return
  setCsrfToken(null)
  window.dispatchEvent(new CustomEvent(AUTH_EXPIRED_EVENT, { detail: message }))
}

export interface AccountStatus {
  initialized: boolean
  authenticated: boolean
  username: string | null
  csrf_token: string | null
}
