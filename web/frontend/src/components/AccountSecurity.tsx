import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { LoaderCircle, LogOut, ShieldCheck, UserRound, X } from 'lucide-react'
import { api } from '../api'
import { notifyAuthExpired } from '../authSession'
import { PasswordField } from './AuthGate'
import { useModalFocus } from '../useModalFocus'

export function AccountSecurity({ onDraftChange }: { onDraftChange: (dirty: boolean, saving: boolean) => void }) {
  const [username, setUsername] = useState('')
  const [current, setCurrent] = useState('')
  const [password, setPassword] = useState('')
  const [confirmation, setConfirmation] = useState('')
  const [busy, setBusy] = useState<'password' | 'logout' | null>(null)
  const [error, setError] = useState('')
  const [logoutOpen, setLogoutOpen] = useState(false)
  const [logoutError, setLogoutError] = useState('')
  const logoutPending = useRef(false)
  const logoutTrigger = useRef<HTMLButtonElement>(null)
  const dirty = !!(current || password || confirmation)
  function closeLogout() {
    if (!logoutPending.current) setLogoutOpen(false)
  }
  // The confirmation is portalled above settings; prevent interaction with its parent.
  useEffect(() => {
    if (!logoutOpen) return
    const settings = logoutTrigger.current?.closest<HTMLElement>('[role="dialog"]')
    if (!settings) return
    const wasInert = settings.inert
    settings.inert = true
    return () => { settings.inert = wasInert }
  }, [logoutOpen])
  const logoutDialog = useModalFocus<HTMLDivElement>(logoutOpen, closeLogout, {
    canClose: busy !== 'logout', initialFocusSelector: '[data-logout-cancel]',
  })
  useEffect(() => {
    const controller = new AbortController()
    api.accountStatus(controller.signal).then(account => setUsername(account.username || '')).catch(reason => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '账号读取失败')
    })
    return () => controller.abort()
  }, [])
  useEffect(() => { onDraftChange(dirty, !!busy) }, [dirty, busy, onDraftChange])
  async function change(event: React.FormEvent) {
    event.preventDefault()
    if (busy) return
    if (password !== confirmation) { setError('两次输入的新密码不一致'); return }
    if (current === password) { setError('新密码不能与当前密码相同'); return }
    setBusy('password'); setError('')
    try {
      await api.changePassword(current, password)
      setCurrent(''); setPassword(''); setConfirmation('')
      notifyAuthExpired('密码已修改，请使用新密码重新登录。后台训练继续运行。')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '密码修改失败，请重试')
    } finally { setBusy(null) }
  }
  async function logout() {
    if (busy || logoutPending.current) return
    logoutPending.current = true
    setBusy('logout'); setLogoutError('')
    try {
      await api.logout(); notifyAuthExpired()
    } catch (reason) {
      if ((reason as { status?: number }).status === 401) { notifyAuthExpired(); return }
      setLogoutError(`退出结果暂未确认，请重试。${reason instanceof Error ? reason.message : '退出失败'}`)
    } finally { logoutPending.current = false; setBusy(null) }
  }
  return <div className="account-security">
    <div className="account-summary"><span className="account-avatar"><UserRound size={24} /></span><div><strong>{username || '管理员'}</strong><span>平台管理员 · 当前账号</span></div><span className="account-badge"><ShieldCheck size={14} />已登录</span><button ref={logoutTrigger} type="button" className="button" onClick={() => { setLogoutError(''); setLogoutOpen(true) }} disabled={!!busy} aria-haspopup="dialog"><LogOut size={16} />退出登录</button></div>
    <form id="account-password-form" className="account-password-form" onSubmit={change} aria-label="修改密码">
      <div><h4>修改密码</h4><p>使用 10～128 个字符的新密码。修改后所有浏览器会话需重新登录。</p></div>
      <PasswordField label="当前密码" value={current} onChange={setCurrent} autoComplete="current-password" disabled={!!busy} />
      <PasswordField label="新密码" value={password} onChange={setPassword} autoComplete="new-password" minLength={10} disabled={!!busy} />
      <PasswordField label="确认新密码" value={confirmation} onChange={setConfirmation} autoComplete="new-password" minLength={10} disabled={!!busy} />
      {error && <div className="alert error" role="alert">{error}</div>}
    </form>
    <div className="account-training-note"><ShieldCheck size={18} /><p>修改密码、退出账号或登录过期只影响网页访问，已提交的训练、数据集处理和远程 Agent 会继续运行。</p></div>
    {logoutOpen && createPortal(<div className="account-logout-backdrop" onMouseDown={event => {
      event.stopPropagation()
      if (event.target === event.currentTarget) closeLogout()
    }} onKeyDown={event => event.stopPropagation()}>
      <div ref={logoutDialog} className="account-logout-dialog" role="alertdialog" aria-modal="true" aria-labelledby="account-logout-title" aria-describedby="account-logout-description" aria-busy={busy === 'logout'} tabIndex={-1}>
        <button type="button" className="account-logout-close" aria-label="关闭退出确认" onClick={closeLogout} disabled={busy === 'logout'}><X size={20} /></button>
        <span className="account-logout-icon"><LogOut size={26} /></span>
        <h2 id="account-logout-title">退出当前账号？</h2>
        <p id="account-logout-description">退出后需要重新登录才能访问平台，仅退出当前浏览器会话。</p>
        <div className="account-training-note"><ShieldCheck size={18} /><p>本地与远程训练、数据集处理和远程 Agent 将继续运行。</p></div>
        {dirty && <p className="account-logout-draft">尚未保存的密码修改将被放弃。</p>}
        {logoutError && <div className="alert error" role="alert">{logoutError}</div>}
        <div className="account-logout-actions">
          <button type="button" className="button" data-logout-cancel onClick={closeLogout} disabled={busy === 'logout'}>取消</button>
          <button type="button" className="button danger" onClick={() => void logout()} disabled={busy === 'logout'}>{busy === 'logout' ? <LoaderCircle size={16} className="spin" /> : <LogOut size={16} />}{busy === 'logout' ? '正在退出…' : '确认退出'}</button>
        </div>
      </div>
    </div>, document.body)}
  </div>
}
