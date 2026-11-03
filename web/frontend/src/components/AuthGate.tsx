import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { ArrowRight, Eye, EyeOff, LoaderCircle, LockKeyhole, Pause, Play, RefreshCw, ScanEye, ShieldCheck, UserRound } from 'lucide-react'
import { api } from '../api'
import { AUTH_EXPIRED_EVENT, setCsrfToken, type AccountStatus } from '../authSession'
import { HolographicHalo } from './HolographicHalo'

// One desktop composition scales as a whole; long setup/error states may scroll.
// The compact CSS layout keeps native-sized controls for phones and tablets.
function AuthScene({ children }: { children: ReactNode }) {
  const viewport = useRef<HTMLDivElement>(null)
  const scene = useRef<HTMLDivElement>(null)
  const [layout, setLayout] = useState({ scale: 1, width: 1440, height: 900 })
  useLayoutEffect(() => {
    const frame = viewport.current!
    const content = scene.current!
    const update = () => {
      if (window.matchMedia('(max-width: 1099px)').matches) return
      const scale = Math.min(frame.clientWidth / 1440, Math.max(frame.clientHeight / 900, .78))
      const next = { scale, width: 1440 * scale, height: content.offsetHeight * scale }
      setLayout(previous => previous.scale === next.scale && previous.height === next.height ? previous : next)
    }
    const observer = new ResizeObserver(update)
    observer.observe(frame)
    observer.observe(content)
    update()
    return () => observer.disconnect()
  }, [])
  return <div className="auth-scene-viewport" ref={viewport}>
    <div className="auth-scene-bounds" style={{ width: layout.width, height: layout.height, '--auth-scene-scale': layout.scale } as CSSProperties}>
      <div className="auth-scene" ref={scene}>{children}</div>
    </div>
  </div>
}

export function PasswordField({ label, value, onChange, autoComplete, minLength = 1, disabled = false }: {
  label: string; value: string; onChange: (value: string) => void
  autoComplete: string; minLength?: number; disabled?: boolean
}) {
  const [visible, setVisible] = useState(false)
  return <label className="account-field"><span>{label}</span><span className="account-input-wrap">
    <LockKeyhole size={17} aria-hidden="true" /><input type={visible ? 'text' : 'password'} required value={value} onChange={event => onChange(event.target.value)} autoComplete={autoComplete} minLength={minLength} maxLength={128} placeholder={`请输入${label}`} disabled={disabled} />
    <button type="button" className="account-password-toggle" aria-label={`${visible ? '隐藏' : '显示'}${label}`} aria-pressed={visible} onClick={() => setVisible(!visible)}>{visible ? <EyeOff size={17} /> : <Eye size={17} />}</button>
  </span></label>
}

export function AuthGate({ children }: { children: ReactNode }) {
  const [account, setAccount] = useState<AccountStatus | null>(null)
  const [error, setError] = useState('')
  const [checking, setChecking] = useState(true)
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmation, setConfirmation] = useState('')
  const [code, setCode] = useState('')
  const [paused, setPaused] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)
  const check = useCallback(async (initial = false) => {
    controllerRef.current?.abort()
    const controller = new AbortController(); controllerRef.current = controller
    if (initial) { setChecking(true); setError('') }
    try {
      const result = await api.accountStatus(controller.signal)
      if (controller.signal.aborted) return
      setCsrfToken(result.csrf_token); setAccount(result)
      if (initial) setError('')
    } catch (reason) {
      if (!controller.signal.aborted && initial) setError(reason instanceof Error ? reason.message : '无法连接平台')
    } finally { if (!controller.signal.aborted) setChecking(false) }
  }, [])
  useEffect(() => {
    void check(true)
    const expired = (event: Event) => {
      controllerRef.current?.abort()
      setAccount(current => current ? { ...current, authenticated: false, csrf_token: null } : null)
      setPassword(''); setConfirmation(''); setError(''); setChecking(false)
      setNotice((event as CustomEvent<string>).detail || '请重新登录；后台训练会继续运行。')
    }
    window.addEventListener(AUTH_EXPIRED_EVENT, expired)
    return () => { controllerRef.current?.abort(); window.removeEventListener(AUTH_EXPIRED_EVENT, expired) }
  }, [check])
  useEffect(() => {
    if (!account?.authenticated) return
    const refresh = () => { if (!document.hidden) void check() }
    const timer = window.setInterval(refresh, 60_000)
    window.addEventListener('focus', refresh)
    return () => { window.clearInterval(timer); window.removeEventListener('focus', refresh) }
  }, [account?.authenticated, check])

  if (account?.authenticated) return children
  const setup = account && !account.initialized
  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (busy) return
    if (setup && password !== confirmation) { setError('两次输入的密码不一致'); return }
    setBusy(true); setError(''); setNotice('')
    try {
      const result = setup ? await api.setupAccount(username.trim(), password, code.trim()) : await api.login(username.trim(), password)
      setCsrfToken(result.csrf_token)
      setPassword(''); setConfirmation(''); setCode('')
      setAccount({ initialized: true, authenticated: true, ...result })
      // The current pathname/query/hash never changed, so deep links resume exactly.
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '登录失败，请重试')
      if ((reason as { status?: number }).status === 409) void check()
    } finally { setBusy(false) }
  }
  return <main className="auth-screen">
    <HolographicHalo paused={paused} />
    <AuthScene>
    <header className="auth-header"><div className="auth-brand"><ScanEye aria-hidden="true" /><strong>YOLO</strong><span>VISION LAB</span></div><span className="auth-header-label">模型训练平台</span></header>
    <div className="auth-content"><section className="auth-form-region" aria-labelledby="auth-title">
      <div className="auth-kicker">ENTER THE NEXT DIMENSION</div>
      <h1 id="auth-title">连接你的<br /><span>视觉宇宙。</span></h1>
      <p className="auth-subtitle">{setup ? '首次使用 · 设置你的管理员账号' : '一个入口，探索模型的无限可能'}</p>
      {checking ? <div className="auth-checking" role="status"><LoaderCircle className="spin" size={18} />正在连接你的工作空间…</div> : !account ? <div className="auth-connection-error"><p role="alert">{error || '暂时无法读取登录状态'}</p><button className="auth-retry" type="button" onClick={() => void check(true)}><RefreshCw size={16} />重新连接</button></div> : <form onSubmit={submit} aria-label={setup ? '设置管理员' : '登录平台'}>
        {setup && <div className="auth-setup-help"><ShieldCheck size={17} /><div>在平台服务器的后端目录执行 <code>python -m app.auth setup-code</code>，将返回的一次性校验码填入下方。{<span>账号信息仅保存在当前平台实例。</span>}</div></div>}
        {setup && <label className="account-field"><span>服务器校验码</span><span className="account-input-wrap"><ShieldCheck size={17} aria-hidden="true" /><input required value={code} onChange={e => setCode(e.target.value)} placeholder="一次性校验码" autoComplete="off" maxLength={128} disabled={busy} /></span></label>}
        <label className="account-field"><span>账号</span><span className="account-input-wrap"><UserRound size={17} aria-hidden="true" /><input required value={username} onChange={e => setUsername(e.target.value)} autoComplete="username" placeholder={setup ? '设置管理员账号' : '请输入账号'} maxLength={80} disabled={busy} /></span></label>
        <PasswordField label={setup ? '新密码' : '密码'} value={password} onChange={setPassword} autoComplete={setup ? 'new-password' : 'current-password'} minLength={setup ? 10 : 1} disabled={busy} />
        {setup && <><PasswordField label="确认新密码" value={confirmation} onChange={setConfirmation} autoComplete="new-password" minLength={10} disabled={busy} /><p className="auth-password-hint">密码须为 10～128 个字符。</p></>}
        {error && <p className="auth-error" role="alert">{error}</p>}
        {notice && <p className="auth-notice" role="status">{notice}</p>}
        <button type="submit" className="auth-submit" disabled={busy}><span>{busy ? (setup ? '正在创建…' : '正在进入…') : setup ? '创建账号并进入平台' : '开启探索'}</span>{busy ? <LoaderCircle size={19} className="spin" /> : <ArrowRight size={19} />}</button>
        <p className="auth-training-note"><ShieldCheck size={14} />{setup ? '无默认密码 · 请妥善保管账号' : '登录状态不会影响后台训练'}</p>
      </form>}
    </section><div className="auth-visual-region" aria-hidden="true"><div className="auth-orbit-caption"><strong>VISION</strong><span>BEYOND THE ORDINARY</span></div></div></div>
    <footer className="auth-footer"><span>YOLO · 模型训练平台 <small>v{import.meta.env.VITE_PLATFORM_VERSION}</small></span><button type="button" className="auth-motion-toggle" onClick={() => setPaused(!paused)} aria-pressed={paused}>{paused ? <Play size={14} /> : <Pause size={14} />}{paused ? '开启动效' : '暂停动效'}</button></footer>
    </AuthScene>
  </main>
}
