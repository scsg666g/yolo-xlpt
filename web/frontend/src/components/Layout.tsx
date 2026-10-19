import { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, Boxes, Check, Cpu, Database, LayoutDashboard, Minus, Moon, Plus, RefreshCw, RotateCcw, Settings2, Sun, X } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'
import packageMetadata from '../../package.json'
import { api } from '../api'
import {
  COMPARE_LIMIT_DEFAULT,
  COMPARE_LIMIT_MAX,
  COMPARE_LIMIT_MIN,
  readStoredCompareRuns,
  useCompareLimit,
  writeCompareLimit,
  writeStoredCompareRuns,
} from '../preferences'
import type { PlatformHealth } from '../types'

const HEALTH_POLL_INTERVAL_MS = 30_000

type HealthPhase = 'checking' | 'online' | 'error'

interface HealthView {
  phase: HealthPhase
  details: PlatformHealth | null
  error: string
  checkedAt: Date | null
}

function healthIssues(details: PlatformHealth): string[] {
  const issues: string[] = []
  if (details.status !== 'ok') issues.push('服务状态异常')
  if (details.database !== 'ok') issues.push('数据库异常')
  if (details.scheduler !== 'running') issues.push('训练调度器未运行')
  if (details.local_deletion_worker !== 'running') issues.push('文件清理服务未运行')
  if (details.inference_worker !== 'running') issues.push('模型测试服务未运行')
  return issues
}

function workerLabel(status: 'running' | 'stopped' | undefined) {
  if (!status) return '—'
  return status === 'running' ? '运行中' : '已停止'
}

export function Layout() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() =>
    document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light',
  )
  const [settingsOpen, setSettingsOpen] = useState(false)
  const compareLimit = useCompareLimit()
  const [compareLimitDraft, setCompareLimitDraft] = useState(compareLimit)
  const [storedCompareCount, setStoredCompareCount] = useState(0)
  const [health, setHealth] = useState<HealthView>({ phase: 'checking', details: null, error: '', checkedAt: null })
  const healthController = useRef<AbortController | null>(null)
  const settingsTrigger = useRef<HTMLButtonElement>(null)
  const settingsDialog = useRef<HTMLElement>(null)

  function changeTheme(nextTheme: 'light' | 'dark') {
    setTheme(nextTheme)
    document.documentElement.dataset.theme = nextTheme
    localStorage.setItem('yolo-forge-theme', nextTheme)
  }

  const checkHealth = useCallback(async (showChecking = false) => {
    healthController.current?.abort()
    const controller = new AbortController()
    healthController.current = controller
    if (showChecking) setHealth((current) => ({ ...current, phase: 'checking', error: '' }))
    try {
      const details = await api.health(controller.signal)
      if (controller.signal.aborted) return
      const issues = healthIssues(details)
      setHealth({
        phase: issues.length ? 'error' : 'online',
        details,
        error: issues.join('；'),
        checkedAt: new Date(),
      })
    } catch (reason) {
      if (controller.signal.aborted) return
      setHealth({
        phase: 'error',
        details: null,
        error: reason instanceof Error ? reason.message : '无法连接训练服务',
        checkedAt: new Date(),
      })
    } finally {
      if (healthController.current === controller) healthController.current = null
    }
  }, [])

  useEffect(() => {
    let disposed = false
    let timer: number | undefined

    const schedule = () => {
      if (disposed || document.hidden) return
      window.clearTimeout(timer)
      timer = window.setTimeout(() => void run(), HEALTH_POLL_INTERVAL_MS)
    }
    const run = async () => {
      await checkHealth()
      schedule()
    }
    const visibilityChanged = () => {
      window.clearTimeout(timer)
      if (document.hidden) healthController.current?.abort()
      else void run()
    }

    document.addEventListener('visibilitychange', visibilityChanged)
    if (!document.hidden) void run()
    return () => {
      disposed = true
      window.clearTimeout(timer)
      document.removeEventListener('visibilitychange', visibilityChanged)
      healthController.current?.abort()
    }
  }, [checkHealth])

  useEffect(() => {
    if (!settingsOpen) return
    const trigger = settingsTrigger.current
    setCompareLimitDraft(compareLimit)
    setStoredCompareCount(readStoredCompareRuns().length)
    settingsDialog.current?.querySelector<HTMLElement>('[data-settings-initial-focus]')?.focus()
    return () => {
      if (trigger?.isConnected) trigger.focus()
    }
  }, [compareLimit, settingsOpen])

  function applyCompareLimit() {
    const current = readStoredCompareRuns()
    const retained = current.slice(0, compareLimitDraft)
    if (retained.length !== current.length) writeStoredCompareRuns(retained)
    writeCompareLimit(compareLimitDraft)
    setStoredCompareCount(retained.length)
  }

  function closeSettings() {
    setSettingsOpen(false)
  }

  function handleSettingsKeyDown(event: React.KeyboardEvent<HTMLElement>) {
    if (event.nativeEvent.isComposing) return
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      closeSettings()
      return
    }
    if (event.key !== 'Tab') return

    const controls = event.currentTarget.querySelectorAll<HTMLElement>(
      'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex]:not([tabindex="-1"])',
    )
    const first = controls[0]
    const last = controls[controls.length - 1]
    if (!first || !last) {
      event.preventDefault()
      event.currentTarget.focus()
    } else if (event.shiftKey && (document.activeElement === first || document.activeElement === event.currentTarget)) {
      event.preventDefault()
      last.focus()
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault()
      first.focus()
    }
  }

  const serviceLabel = health.phase === 'online' ? '训练服务在线' : health.phase === 'error' ? '训练服务异常' : '训练服务检查中'
  const serviceDescription = health.phase === 'online' ? '所有核心服务运行正常' : health.phase === 'checking' ? '正在读取平台运行状态…' : health.error || '核心服务状态异常'
  const dotColor = health.phase === 'online' ? 'var(--green)' : health.phase === 'error' ? '#ff747b' : '#d9a552'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Boxes size={19} /></div>
          <div className="brand-copy">
            <div className="brand-title">
              <strong>YOLO Forge</strong>
              <span className="brand-version" title={`平台版本 ${packageMetadata.version}`}>v{packageMetadata.version}</span>
            </div>
            <span>训练控制台</span>
          </div>
        </div>
        <div className="theme-picker" aria-label="界面主题">
          <span>界面主题</span>
          <div>
            <button type="button" className={theme === 'light' ? 'active' : ''} onClick={() => changeTheme('light')} title="明亮主题" aria-pressed={theme === 'light'}><Sun size={14} />明亮</button>
            <button type="button" className={theme === 'dark' ? 'active' : ''} onClick={() => changeTheme('dark')} title="深色主题" aria-pressed={theme === 'dark'}><Moon size={14} />深色</button>
          </div>
        </div>
        <nav>
          <NavLink to="/" end><LayoutDashboard size={18} />任务中心</NavLink>
          <NavLink to="/compare"><Activity size={18} />实验对比</NavLink>
          <NavLink to="/compute"><Cpu size={18} />算力资源</NavLink>
          <NavLink to="/datasets"><Database size={18} />数据集管理</NavLink>
        </nav>
        <div className="sidebar-footer">
          <div className="service-state" title={health.error || serviceLabel} aria-label={serviceLabel}>
            <i style={{ background: dotColor, boxShadow: `0 0 10px ${dotColor}` }} />{serviceLabel}
          </div>
          <button ref={settingsTrigger} type="button" className="icon-text" title="平台设置" aria-label="打开平台设置" onClick={() => setSettingsOpen(true)}><Settings2 size={16} />平台设置</button>
        </div>
      </aside>
      <main className="main"><Outlet /></main>

      {settingsOpen && <div className="modal-backdrop" onMouseDown={closeSettings}>
        <section ref={settingsDialog} className="modal platform-settings-modal" role="dialog" aria-modal="true" aria-labelledby="platform-settings-title" tabIndex={-1} onKeyDown={handleSettingsKeyDown} onMouseDown={(event) => event.stopPropagation()}>
          <header>
            <div><span className="eyebrow">PLATFORM STATUS</span><h2 id="platform-settings-title">平台设置与状态</h2></div>
            <button type="button" className="icon-button" data-settings-initial-focus title="关闭" aria-label="关闭平台设置" onClick={closeSettings}><X size={19} /></button>
          </header>
          <div className="modal-body">
            <div className={`alert${health.phase === 'error' ? ' error' : health.phase === 'checking' ? ' warning' : ''}`} role={health.phase === 'error' ? 'alert' : 'status'}>{serviceDescription}</div>

            <div className="parameter-section">
              <h3>界面主题</h3>
              <p>选择保存在当前浏览器中的显示模式。</p>
            </div>
            <div className="header-actions">
              <button type="button" className={`button ${theme === 'light' ? 'primary' : 'ghost'}`} aria-pressed={theme === 'light'} onClick={() => changeTheme('light')}><Sun size={15} />明亮主题</button>
              <button type="button" className={`button ${theme === 'dark' ? 'primary' : 'ghost'}`} aria-pressed={theme === 'dark'} onClick={() => changeTheme('dark')}><Moon size={15} />深色主题</button>
            </div>

            <div className="parameter-section" style={{ marginTop: 24 }}>
              <h3>实验对比</h3>
              <p>设置当前浏览器最多同时展示的训练任务数量。</p>
            </div>
            <div className="compare-limit-setting">
              <div>
                <strong>最大同时对比任务数</strong>
                <span>允许 {COMPARE_LIMIT_MIN}～{COMPARE_LIMIT_MAX} 个，默认 {COMPARE_LIMIT_DEFAULT} 个</span>
              </div>
              <div className="compare-limit-controls">
                <div className="number-stepper" aria-label="最大同时对比任务数">
                  <button type="button" disabled={compareLimitDraft <= COMPARE_LIMIT_MIN} title="减少对比数量" aria-label="减少对比数量" onClick={() => setCompareLimitDraft((value) => Math.max(COMPARE_LIMIT_MIN, value - 1))}><Minus size={15} /></button>
                  <output aria-live="polite">{compareLimitDraft}</output>
                  <button type="button" disabled={compareLimitDraft >= COMPARE_LIMIT_MAX} title="增加对比数量" aria-label="增加对比数量" onClick={() => setCompareLimitDraft((value) => Math.min(COMPARE_LIMIT_MAX, value + 1))}><Plus size={15} /></button>
                </div>
                <button type="button" className="icon-button" disabled={compareLimitDraft === COMPARE_LIMIT_DEFAULT} title={`恢复默认 ${COMPARE_LIMIT_DEFAULT} 个`} aria-label={`恢复默认 ${COMPARE_LIMIT_DEFAULT} 个`} onClick={() => setCompareLimitDraft(COMPARE_LIMIT_DEFAULT)}><RotateCcw size={15} /></button>
                <button type="button" className="button primary compact" disabled={compareLimitDraft === compareLimit} onClick={applyCompareLimit}><Check size={15} />应用</button>
              </div>
            </div>
            <div className={`compare-limit-message${storedCompareCount > compareLimitDraft ? ' warning' : ''}`} aria-live="polite">
              {storedCompareCount > compareLimitDraft
                ? `当前已选择 ${storedCompareCount} 个任务；应用后保留前 ${compareLimitDraft} 个，并移除其余 ${storedCompareCount - compareLimitDraft} 个。`
                : compareLimitDraft === compareLimit ? `当前上限为 ${compareLimit} 个。` : `新上限 ${compareLimitDraft} 个尚未应用。`}
            </div>

            <div className="parameter-section" style={{ marginTop: 24 }}>
              <h3>核心服务</h3>
              <p>状态每 30 秒更新，页面隐藏时会暂停检查。</p>
            </div>
            <div className="metric-strip service-metric-strip" aria-label="核心服务状态">
              <div><span>数据库</span><strong>{health.details?.database === 'ok' ? '正常' : '—'}</strong></div>
              <div><span>训练调度器</span><strong>{workerLabel(health.details?.scheduler)}</strong></div>
              <div><span>文件清理</span><strong>{workerLabel(health.details?.local_deletion_worker)}</strong></div>
              <div><span>模型测试</span><strong>{workerLabel(health.details?.inference_worker)}</strong></div>
            </div>
            <div className="form-grid" style={{ marginTop: 18 }}>
              <label>平台实例<input readOnly value={health.details?.instance || '—'} /></label>
              <label>数据库延迟<input readOnly value={health.details ? `${health.details.database_latency_ms.toFixed(2)} ms` : '—'} /></label>
              <label className="span-2">训练输出目录<input readOnly title={health.details?.runs_root || ''} value={health.details?.runs_root || '—'} /></label>
            </div>
          </div>
          <footer>
            <span style={{ marginRight: 'auto' }}>{health.checkedAt ? `上次检查：${health.checkedAt.toLocaleTimeString('zh-CN')}` : '尚未完成检查'}</span>
            <button type="button" className="button ghost" disabled={health.phase === 'checking'} onClick={() => void checkHealth(true)}><RefreshCw size={15} className={health.phase === 'checking' ? 'spin' : ''} />重新检查</button>
            <button type="button" className="button" onClick={closeSettings}>关闭</button>
          </footer>
        </section>
      </div>}
    </div>
  )
}
