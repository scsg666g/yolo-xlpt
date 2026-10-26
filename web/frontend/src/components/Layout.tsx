import { DatasetStorageSettings } from './DatasetStorage'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, Boxes, Check, Cpu, Database, Download, FolderOpen, LayoutDashboard, Minus, Monitor, Moon, Plus, RefreshCw, RotateCcw, ScanEye, Server, Settings2, Sun, X } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'
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
import type { ComputeNode, PlatformHealth, PlatformStorageHealth } from '../types'
import { resetAppearance, updateAppearance, useAppearance } from '../appearance'
import { AppearanceSettings } from './AppearanceSettings'

const HEALTH_POLL_INTERVAL_MS = 30_000

const settingsSections = [
  { id: 'overview', label: '概览', description: '平台状态与实例信息。', icon: LayoutDashboard },
  { id: 'appearance', label: '外观', description: '为当前浏览器设置配色与阅读偏好。', icon: Sun },
  { id: 'training', label: '实验与训练', description: '实验对比参数。', icon: Activity },
  { id: 'services', label: '服务状态', description: '数据库与后台服务运行状态。', icon: Cpu },
  { id: 'storage', label: '存储与路径', description: '训练输出目录等平台路径。', icon: FolderOpen },
] as const

type SettingsSection = typeof settingsSections[number]['id']

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
  if (details.scheduler === 'error') issues.push(details.scheduler_error ? `训练调度器：${details.scheduler_error}` : '训练调度器运行异常')
  else if (details.scheduler === 'stopped') issues.push('训练调度器未运行')
  if (details.local_deletion_worker !== 'running') issues.push('文件清理服务未运行')
  if (details.inference_worker !== 'running') issues.push('模型测试服务未运行')
  return issues
}

function workerLabel(status: 'running' | 'stopped' | 'error' | undefined) {
  if (!status) return '—'
  return status === 'running' ? '运行中' : status === 'error' ? '异常' : '已停止'
}

function dateTimeLabel(value: string | null | undefined) {
  if (!value) return ''
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString('zh-CN', { hour12: false })
}

function nodeStatusLabel(status: ComputeNode['status']) {
  return status === 'online' ? '在线' : status === 'offline' ? '离线' : status === 'disabled' ? '已禁用' : '等待接入'
}

function bytesLabel(value: number | null) {
  if (value == null || !Number.isFinite(value)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit += 1 }
  return `${amount.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

function storageStatusLabel(status: PlatformStorageHealth['status']) {
  return status === 'writable' ? '可写' : status === 'read_only' ? '只读' : status === 'missing' ? '目录不存在' : status === 'not_directory' ? '路径不是目录' : '不可检查'
}

export function Layout() {
  const appearance = useAppearance()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsSection, setSettingsSection] = useState<SettingsSection>('overview')
  const compareLimit = useCompareLimit()
  const [compareLimitDraft, setCompareLimitDraft] = useState(compareLimit)
  const [storedCompareCount, setStoredCompareCount] = useState(0)
  const [health, setHealth] = useState<HealthView>({ phase: 'checking', details: null, error: '', checkedAt: null })
  const [computeNodes, setComputeNodes] = useState<ComputeNode[]>([])
  const [computeNodesPhase, setComputeNodesPhase] = useState<'loading' | 'ready' | 'error'>('loading')
  const [computeNodesError, setComputeNodesError] = useState('')
  const [computeNodesCheckedAt, setComputeNodesCheckedAt] = useState<Date | null>(null)
  const [storageHealth, setStorageHealth] = useState<PlatformStorageHealth | null>(null)
  const [storagePhase, setStoragePhase] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle')
  const [storageError, setStorageError] = useState('')
  const [storageCheckedAt, setStorageCheckedAt] = useState<Date | null>(null)
  const [diagnosticsNotice, setDiagnosticsNotice] = useState('')
  const healthController = useRef<AbortController | null>(null)
  const nodesController = useRef<AbortController | null>(null)
  const storageController = useRef<AbortController | null>(null)
  const settingsTrigger = useRef<HTMLButtonElement>(null)
  const settingsDialog = useRef<HTMLElement>(null)

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

  const loadComputeNodes = useCallback(async () => {
    nodesController.current?.abort()
    const controller = new AbortController()
    nodesController.current = controller
    setComputeNodesPhase('loading')
    setComputeNodesError('')
    try {
      const nodes = await api.computeNodes(controller.signal)
      if (controller.signal.aborted) return
      setComputeNodes(nodes)
      setComputeNodesPhase('ready')
      setComputeNodesCheckedAt(new Date())
    } catch (reason) {
      if (controller.signal.aborted) return
      setComputeNodesPhase('error')
      setComputeNodesError(reason instanceof Error ? reason.message : '无法读取算力节点状态')
    } finally {
      if (nodesController.current === controller) nodesController.current = null
    }
  }, [])

  const loadStorageHealth = useCallback(async () => {
    storageController.current?.abort()
    const controller = new AbortController()
    storageController.current = controller
    setStoragePhase('loading')
    setStorageError('')
    try {
      const details = await api.storageHealth(controller.signal)
      if (controller.signal.aborted) return
      setStorageHealth(details)
      setStoragePhase('ready')
      setStorageCheckedAt(new Date())
    } catch (reason) {
      if (controller.signal.aborted) return
      setStoragePhase('error')
      setStorageError(reason instanceof Error ? reason.message : '无法检查存储目录')
    } finally {
      if (storageController.current === controller) storageController.current = null
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
    if (!settingsOpen || !['overview', 'services', 'storage'].includes(settingsSection)) return
    void loadComputeNodes()
    const timer = window.setInterval(() => void loadComputeNodes(), HEALTH_POLL_INTERVAL_MS)
    return () => {
      window.clearInterval(timer)
      nodesController.current?.abort()
    }
  }, [loadComputeNodes, settingsOpen, settingsSection])

  useEffect(() => {
    if (!settingsOpen || settingsSection !== 'storage') return
    void loadStorageHealth()
    return () => storageController.current?.abort()
  }, [loadStorageHealth, settingsOpen, settingsSection])

  useEffect(() => {
    if (!settingsOpen) return
    const trigger = settingsTrigger.current
    setCompareLimitDraft(compareLimit)
    setStoredCompareCount(readStoredCompareRuns().length)
    setDiagnosticsNotice('')
    settingsDialog.current?.querySelector<HTMLElement>('[data-settings-initial-focus]')?.focus()
    return () => {
      if (trigger?.isConnected) trigger.focus()
    }
  }, [compareLimit, settingsOpen])

  function applyCompareLimit() {
    const current = readStoredCompareRuns()
    const retained = current.slice(0, compareLimitDraft)
    if (retained.length !== current.length) {
      const removed = current.length - retained.length
      if (!window.confirm(`当前浏览器已选择 ${current.length} 个对比运行。继续后将从对比列表移除 ${removed} 个选择，训练任务和记录不会删除。是否继续？`)) return
      writeStoredCompareRuns(retained)
    }
    writeCompareLimit(compareLimitDraft)
    setStoredCompareCount(retained.length)
  }

  function downloadDiagnosticsSummary() {
    const summary = {
      generated_at: new Date().toISOString(),
      core_checked_at: health.checkedAt?.toISOString() || null,
      agent_nodes_checked_at: computeNodesCheckedAt?.toISOString() || null,
      platform_version: health.details?.version || import.meta.env.VITE_PLATFORM_VERSION || null,
      platform_status: health.phase,
      core_services: health.details ? {
        database: health.details.database,
        scheduler: health.details.scheduler,
        local_deletion_worker: health.details.local_deletion_worker,
        inference_worker: health.details.inference_worker,
        database_latency_ms: health.details.database_latency_ms,
      } : null,
      agent_nodes: computeNodes.map((node) => ({
        kind: node.kind,
        status: node.status,
        last_seen_at: node.last_seen_at,
        agent_version: node.agent_version,
        required_agent_version: node.required_agent_version,
        training_version_supported: node.kind === 'local' ? null : node.training_version_supported ?? null,
        platform_training_available: node.kind === 'local' ? null : !node.training_unavailable_reason,
        last_reported_training_ready: node.kind === 'local' ? null : node.training?.ready ?? false,
        gpu_process_attribution_supported: node.process_attribution_supported,
      })),
    }
    const blob = new Blob([JSON.stringify(summary, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `yolo-platform-diagnostics-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 0)
    setDiagnosticsNotice('诊断摘要已下载')
  }

  function refreshSettingsStatus() {
    void checkHealth(true)
    void loadComputeNodes()
    if (settingsSection === 'storage') void loadStorageHealth()
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
  const activeSettingsSection = settingsSections.find((section) => section.id === settingsSection) ?? settingsSections[0]
  const remoteNodes = computeNodes.filter((node) => node.kind === 'agent')
  const onlineRemoteNodeCount = remoteNodes.filter((node) => node.status === 'online').length
  const nodesStatusLabel = computeNodesPhase === 'loading' && !computeNodes.length ? '正在读取远程节点…'
    : computeNodesPhase === 'loading' ? `正在刷新 · 上次结果 ${onlineRemoteNodeCount} / ${remoteNodes.length} 在线`
    : computeNodesPhase === 'error' ? computeNodesCheckedAt
      ? `节点刷新失败 · 上次结果 ${onlineRemoteNodeCount} / ${remoteNodes.length} 在线`
      : '远程节点状态读取失败'
      : `远程节点 ${onlineRemoteNodeCount} / ${remoteNodes.length} 在线`
  const coreCheckedLabel = health.checkedAt ? health.checkedAt.toLocaleTimeString('zh-CN') : '尚未完成'
  const nodesCheckedLabel = computeNodesCheckedAt ? computeNodesCheckedAt.toLocaleTimeString('zh-CN') : '尚未完成'
  const storageCheckedLabel = storageCheckedAt ? storageCheckedAt.toLocaleTimeString('zh-CN') : '尚未完成'
  const settingsCheckedLabel = settingsSection === 'storage'
    ? `存储检查：${storageCheckedLabel} · 节点成功读取：${nodesCheckedLabel}`
    : ['overview', 'services'].includes(settingsSection)
      ? `核心检查：${coreCheckedLabel} · 节点成功读取：${nodesCheckedLabel}`
      : `核心检查：${coreCheckedLabel}`

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Boxes size={19} /></div>
          <div className="brand-copy">
            <div className="brand-title">
              <strong>YOLO Forge</strong>
              <span className="brand-version" title={`平台版本 ${import.meta.env.VITE_PLATFORM_VERSION}`}>v{import.meta.env.VITE_PLATFORM_VERSION}</span>
            </div>
            <span>训练控制台</span>
          </div>
        </div>
        <div className="theme-picker" aria-label="界面主题">
          <span>界面主题</span>
          <div>
            <button type="button" className={appearance.mode === 'light' ? 'active' : ''} onClick={() => updateAppearance({ mode: 'light' })} title="明亮主题" aria-pressed={appearance.mode === 'light'}><Sun size={14} />明亮</button>
            <button type="button" className={appearance.mode === 'dark' ? 'active' : ''} onClick={() => updateAppearance({ mode: 'dark' })} title="深色主题" aria-pressed={appearance.mode === 'dark'}><Moon size={14} />深色</button>
          </div>
          {appearance.mode === 'system' && <small className="theme-system-note"><Monitor size={11} />跟随系统 · 当前{appearance.resolvedTheme === 'dark' ? '深色' : '明亮'}</small>}
        </div>
        <nav>
          <NavLink to="/" end><LayoutDashboard size={18} />任务中心</NavLink>
          <NavLink to="/compare"><Activity size={18} />实验对比</NavLink>
          <NavLink to="/model-tests"><ScanEye size={18} />模型测试</NavLink>
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
          <div className="settings-layout">
            <nav className="settings-nav" aria-label="平台设置分类">
              <span className="settings-nav-heading">设置分类</span>
              {settingsSections.map(({ id, label, icon: Icon }) => (
                <button key={id} type="button" className={settingsSection === id ? 'active' : ''} aria-current={settingsSection === id ? 'page' : undefined} aria-controls="platform-settings-content" onClick={() => setSettingsSection(id)}>
                  <Icon size={16} aria-hidden="true" />
                  <span>{label}</span>
                  {id === 'services' && <i className={`settings-nav-status ${health.phase}`} title={serviceLabel} aria-label={serviceLabel} />}
                </button>
              ))}
            </nav>
            <div className="modal-body settings-content" id="platform-settings-content" role="region" aria-labelledby="settings-section-title" key={settingsSection}>
              <section className="settings-panel">
                <div className="parameter-section settings-page-heading">
                  <h3 id="settings-section-title">{activeSettingsSection.label}</h3>
                  <p>{activeSettingsSection.description}</p>
                </div>

                {settingsSection === 'overview' && <>
                  <div className={`alert${health.phase === 'error' ? ' error' : health.phase === 'checking' ? ' warning' : ''}`} role={health.phase === 'error' ? 'alert' : 'status'}>{serviceDescription}</div>
                  <div className="form-grid settings-meta-grid">
                    <label>平台实例<input readOnly value={health.details?.instance || '—'} /></label>
                    <label>平台版本<input readOnly value={health.details?.version || import.meta.env.VITE_PLATFORM_VERSION || '—'} /></label>
                  </div>
                  <div className="settings-overview-actions">
                    <span><Server size={14} />{nodesStatusLabel}</span>
                    <button type="button" className="button ghost compact" disabled={!health.details || computeNodesPhase !== 'ready'} onClick={downloadDiagnosticsSummary}><Download size={14} />下载诊断摘要</button>
                  </div>
                  {diagnosticsNotice && <div className="settings-inline-status" role="status">{diagnosticsNotice}</div>}
                  {computeNodesError && <div className="settings-inline-error" role="status">节点状态暂不可用：{computeNodesError}</div>}
                </>}

                {settingsSection === 'appearance' && <AppearanceSettings />}

                {settingsSection === 'training' && <>
                  <div className="parameter-section">
                    <h3>实验对比</h3>
                    <p>只影响当前浏览器的实验对比列表，不控制训练调度或 GPU 并发。</p>
                  </div>
                  <div className="compare-limit-setting">
                    <div>
                      <strong>最大同时对比运行数</strong>
                      <span>允许 {COMPARE_LIMIT_MIN}～{COMPARE_LIMIT_MAX} 个，默认 {COMPARE_LIMIT_DEFAULT} 个</span>
                    </div>
                    <div className="compare-limit-controls">
                      <div className="number-stepper" aria-label="最大同时对比运行数">
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
                      ? `当前已选择 ${storedCompareCount} 个运行；应用后保留前 ${compareLimitDraft} 个，并移除其余 ${storedCompareCount - compareLimitDraft} 个。`
                      : compareLimitDraft === compareLimit ? `当前上限为 ${compareLimit} 个。` : `新上限 ${compareLimitDraft} 个尚未应用。`}
                  </div>
                </>}

                {settingsSection === 'services' && <>
                  <div className={`alert${health.phase === 'error' ? ' error' : health.phase === 'checking' ? ' warning' : ''}`} role={health.phase === 'error' ? 'alert' : 'status'}>{serviceDescription}</div>
                  <div className="metric-strip service-metric-strip" aria-label="核心服务状态">
                    <div><span>数据库</span><strong>{health.details?.database === 'ok' ? '正常' : '—'}</strong></div>
                    <div><span>训练调度器</span><strong>{workerLabel(health.details?.scheduler)}</strong></div>
                    <div><span>文件清理</span><strong>{workerLabel(health.details?.local_deletion_worker)}</strong></div>
                    <div><span>模型测试</span><strong>{workerLabel(health.details?.inference_worker)}</strong></div>
                  </div>
                  <div className="form-grid settings-service-details">
                    <label>数据库延迟<input readOnly value={health.details ? `${health.details.database_latency_ms.toFixed(2)} ms` : '—'} /></label>
                    {health.details?.scheduler_error && <label className="span-2">当前调度错误<input readOnly title={health.details.scheduler_error} value={health.details.scheduler_error} /></label>}
                    {health.details?.scheduler_last_error && health.details.scheduler_last_error !== health.details.scheduler_error && <label className="span-2">最近调度错误<input readOnly title={health.details.scheduler_last_error} value={`${health.details.scheduler_last_error}${health.details.scheduler_last_error_at ? ` · ${dateTimeLabel(health.details.scheduler_last_error_at)}` : ''}`} /></label>}
                  </div>
                  <div className="settings-subsection-heading"><h4>本机与远程节点</h4><span>{computeNodesCheckedAt ? `最近成功读取 ${computeNodesCheckedAt.toLocaleTimeString('zh-CN')}` : '尚未成功读取'}</span></div>
                  {computeNodesError && <div className="settings-inline-error" role="alert">节点刷新失败：{computeNodesError}{computeNodesCheckedAt ? '；以下保留最近一次成功数据' : ''}</div>}
                  {computeNodesPhase === 'loading' && !computeNodes.length ? <div className="settings-empty-state">正在读取节点状态…</div>
                    : computeNodesPhase === 'error' && !computeNodes.length ? <div className="settings-empty-state">没有可用的节点状态数据</div>
                    : computeNodes.length ? <div className="settings-node-list">{computeNodes.map((node) => {
                      const trainingLabel = node.kind === 'local' ? '本机平台' : node.training_unavailable_reason || '训练就绪'
                      return <div className="settings-node-row" key={node.id}>
                        <div className="settings-node-primary"><strong>{node.name}</strong><span className={`settings-status-badge ${node.status}`}>{node.kind === 'local' ? '本机' : nodeStatusLabel(node.status)}</span></div>
                        <div className="settings-node-details">
                          <span>{node.kind === 'local' ? '平台服务节点' : `Agent ${node.agent_version ? `v${node.agent_version}` : '未上报'}`}</span>
                          {node.kind === 'agent' && <span className={node.training_version_supported === false ? 'settings-capability warning' : 'settings-capability'}>{node.training_version_supported == null ? `最低训练版本 ${node.required_agent_version}+` : node.training_version_supported ? `满足训练要求 ${node.required_agent_version}+` : `需升级至 ${node.required_agent_version}+`}</span>}
                          <span className={node.kind === 'agent' && node.training_unavailable_reason ? 'settings-capability warning' : 'settings-capability'}>{trainingLabel}</span>
                          {node.kind === 'agent' && <span className={node.process_attribution_supported ? 'settings-capability' : 'settings-capability warning'}>{node.process_attribution_supported ? 'GPU 进程可关联' : `GPU 进程关联需 ${node.required_process_attribution_agent_version}+`}</span>}
                        </div>
                        {node.kind === 'agent' && <small>最近心跳：{dateTimeLabel(node.last_seen_at) || '尚未上报'}{node.status === 'offline' ? ' · 当前状态依据旧数据' : ''}</small>}
                      </div>
                    })}</div> : <div className="settings-empty-state">尚未登记算力节点</div>}
                </>}

                {settingsSection === 'storage' && <>
                  <DatasetStorageSettings />
                  <div className="form-grid settings-path-grid">
                    <label className="span-2">平台服务器 · 训练输出目录<input readOnly title={storageHealth?.path || health.details?.runs_root || ''} value={storageHealth?.path || health.details?.runs_root || '—'} /></label>
                  </div>
                  {storagePhase === 'loading' && <div className="settings-empty-state">正在检查平台输出目录…</div>}
                  {storagePhase === 'error' && <div className="settings-inline-error" role="alert">{storageError}</div>}
                  {storagePhase === 'ready' && storageHealth && <div className={`settings-storage-summary ${storageHealth.status === 'writable' ? 'ready' : 'warning'}`}>
                    <strong>{storageStatusLabel(storageHealth.status)}</strong>
                    <span>{storageHealth.free_bytes == null ? '可用空间未知' : `可用 ${bytesLabel(storageHealth.free_bytes)} / ${bytesLabel(storageHealth.total_bytes)}`}</span>
                  </div>}
                  <div className="settings-subsection-heading"><h4>远程训练输出目录</h4><span>状态来自 Agent 最近一次上报</span></div>
                  {computeNodesError && <div className="settings-inline-error" role="alert">节点刷新失败：{computeNodesError}{computeNodesCheckedAt ? '；以下保留最近一次成功数据' : ''}</div>}
                  {computeNodesPhase === 'loading' && !computeNodes.length ? <div className="settings-empty-state">正在读取远程 Agent 目录…</div>
                    : computeNodesPhase === 'error' && !computeNodes.length ? <div className="settings-empty-state">没有可用的远程 Agent 目录数据</div>
                      : remoteNodes.length ? <div className="settings-storage-nodes">{remoteNodes.map((node) => {
                    const roots = node.training?.output_roots || []
                    const modelRoots = node.training?.model_test_roots || []
                    const outputStatus = node.status !== 'online'
                      ? `上次上报：${node.training?.output_writable ? '输出目录可写' : '输出目录不可写或未知'}`
                      : node.training?.output_writable ? '输出目录可写' : '输出目录不可写或未上报'
                    return <div className="settings-storage-node" key={node.id}>
                      <div className="settings-storage-node-heading"><strong>{node.name}</strong><span className={`settings-status-badge ${node.status}`}>{nodeStatusLabel(node.status)}</span><span className={node.status === 'online' && node.training?.output_writable ? 'settings-capability' : 'settings-capability warning'}>{outputStatus}</span></div>
                      {roots.length ? <details><summary>训练输出目录 · {roots.length} 项</summary><div className="settings-storage-roots">{roots.map((root) => <code key={root}>{root}</code>)}</div></details> : <small>Agent 尚未上报允许的训练输出目录</small>}
                      {modelRoots.length ? <details><summary>模型测试目录 · {modelRoots.length} 项</summary><div className="settings-storage-roots">{modelRoots.map((root) => <code key={root}>{root}</code>)}</div></details> : <small>Agent 尚未上报允许的模型测试目录</small>}
                    </div>
                  })}</div> : <div className="settings-empty-state">尚未登记远程 Agent</div>}
                  <p className="settings-readonly-note">训练和模型目录来自运行环境；数据集保存位置可在上方配置，目录检查结果含服务器时间。</p>
                </>}
              </section>
            </div>
          </div>
          <footer>
            {settingsSection === 'appearance' ? <>
              <span className="appearance-save-note" role="status">{appearance.storageAvailable ? '即时生效 · 仅保存在当前浏览器' : '即时生效 · 浏览器存储不可用，刷新后可能恢复默认'}</span>
              <button type="button" className="button ghost" onClick={resetAppearance}><RotateCcw size={15} />恢复默认版</button>
            </> : <>
              <span style={{ marginRight: 'auto' }}>{settingsCheckedLabel}</span>
              <button type="button" className="button ghost" disabled={health.phase === 'checking'} onClick={refreshSettingsStatus}><RefreshCw size={15} className={health.phase === 'checking' ? 'spin' : ''} />重新检查</button>
            </>}
            <button type="button" className="button" onClick={closeSettings}>关闭</button>
          </footer>
        </section>
      </div>}
    </div>
  )
}
