import { useEffect, useState } from 'react'
import { LoaderCircle, LogOut, ShieldCheck, UserRound } from 'lucide-react'
import { api } from '../api'
import { notifyAuthExpired } from '../authSession'
import { PasswordField } from './AuthGate'

export function AccountSecurity({ onDraftChange }: { onDraftChange: (dirty: boolean, saving: boolean) => void }) {
  const [username, setUsername] = useState('')
  const [current, setCurrent] = useState('')
  const [password, setPassword] = useState('')
  const [confirmation, setConfirmation] = useState('')
  const [busy, setBusy] = useState<'password' | 'logout' | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const dirty = !!(current || password || confirmation)
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
    setBusy('password'); setError(''); setNotice('')
    try {
      await api.changePassword(current, password)
      setCurrent(''); setPassword(''); setConfirmation('')
      notifyAuthExpired('密码已修改，请使用新密码重新登录。后台训练继续运行。')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '密码修改失败，请重试')
    } finally { setBusy(null) }
  }
  async function logout() {
    if (busy || !window.confirm('退出当前账号？后台训练和远程 Agent 将继续运行。')) return
    setBusy('logout'); setError('')
    try {
      await api.logout(); notifyAuthExpired()
    } catch (reason) {
      if ((reason as { status?: number }).status === 401) { notifyAuthExpired(); return }
      setNotice('退出结果暂未确认，请重试；后台训练不受影响。')
      setError(reason instanceof Error ? reason.message : '退出失败')
    } finally { setBusy(null) }
  }
  return <div className="account-security">
    <div className="account-summary"><span className="account-avatar"><UserRound size={24} /></span><div><strong>{username || '管理员'}</strong><span>平台管理员 · 当前账号</span></div><span className="account-badge"><ShieldCheck size={14} />已登录</span><button type="button" className="button" onClick={() => void logout()} disabled={!!busy}>{busy === 'logout' ? <LoaderCircle size={16} className="spin" /> : <LogOut size={16} />}退出登录</button></div>
    <form id="account-password-form" className="account-password-form" onSubmit={change} aria-label="修改密码">
      <div><h4>修改密码</h4><p>使用 10～128 个字符的新密码。修改后所有浏览器会话需重新登录。</p></div>
      <PasswordField label="当前密码" value={current} onChange={setCurrent} autoComplete="current-password" disabled={!!busy} />
      <PasswordField label="新密码" value={password} onChange={setPassword} autoComplete="new-password" minLength={10} disabled={!!busy} />
      <PasswordField label="确认新密码" value={confirmation} onChange={setConfirmation} autoComplete="new-password" minLength={10} disabled={!!busy} />
      {error && <div className="alert error" role="alert">{error}</div>}
      {notice && <div className="alert" role="status">{notice}</div>}
    </form>
    <div className="account-training-note"><ShieldCheck size={18} /><p>修改密码、退出账号或登录过期只影响网页访问，已提交的训练、数据集处理和远程 Agent 会继续运行。</p></div>
  </div>
}
