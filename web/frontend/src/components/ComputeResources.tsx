import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { Activity, ChevronDown, Cpu, Gauge, HardDrive, MemoryStick, Pencil, Plus, RefreshCw, Search, Server, Settings2, Thermometer, Trash2, Zap } from 'lucide-react'
import { Link } from 'react-router-dom'
import { api, formatBytes } from '../api'
import type { ComputeNode, ComputeNodeRefreshResult, ComputeResources as ComputeData, GpuInfo, GpuProcess } from '../types'
import { ComputeNodeModal } from './ComputeNodeModal'

function memoryLabel(value: number | null | undefined): string {
  if (value == null) return '—'
  if (value >= 1024) return `${(value / 1024).toFixed(1)} GB`
  return `${value} MB`
}

function valueLabel(value: number | null | undefined, suffix: string, digits = 0): string {
  return value == null ? '—' : `${value.toFixed(digits)}${suffix}`
}

function ratio(value: number | null | undefined, total: number | null | undefined): number {
  if (value == null || total == null || total <= 0) return 0
  return Math.min(100, Math.max(0, value / total * 100))
}

function processRows(gpus: GpuInfo[]): Array<GpuProcess & { gpuIndex: number }> {
  return gpus.flatMap((gpu) => (gpu.processes || []).map((process) => ({ ...process, gpuIndex: gpu.index })))
}

function associationText(association: GpuProcess['association']): string {
  if (!association) return '归属无法确认'
  switch (association.state) {
    case 'managed': return `${association.task_name || '平台训练'} · Attempt ${String(association.attempt_number || 1).padStart(3, '0')}`
    case 'other_node': return `${association.task_name || '本平台任务'} · 其他节点`
    case 'other_platform': return '其他 YOLO Forge 实例'
    case 'unmanaged': return '未关联当前实例'
    case 'orphaned': return '任务记录已结束'
    case 'ambiguous': return '归属冲突'
    case 'stale': return association.task_name
      ? `上次采样：${association.task_name} · 当前待确认`
      : '上次采样结果 · 当前待确认'
    default: return '归属无法确认'
  }
}

function associationTitle(association: GpuProcess['association']): string | undefined {
  if (!association) return undefined
  const method = association.match_method === 'runtime_marker' ? '训练运行标识'
    : association.match_method === 'ancestor' ? '进程父链'
      : association.match_method === 'launcher' ? '启动进程' : ''
  return [association.reason, method && `关联依据：${method}`,
    association.root_pid && `启动 PID：${association.root_pid}`,
    association.sampled_at && `采样时间：${new Date(association.sampled_at).toLocaleString('zh-CN', { hour12: false })}`]
    .filter(Boolean).join(' · ') || undefined
}

function AssociationCell({ process }: { process: GpuProcess }) {
  const association = process.association
  const label = associationText(association)
  const title = associationTitle(association)
  if (association?.task_id && (association.state === 'managed' || association.state === 'other_node')) {
    return <Link className={`linked-task ${association.state === 'other_node' ? 'other-node' : ''}`}
      to={`/tasks/${association.task_id}/canvas`} title={title}>
      {label}
    </Link>
  }
  return <span className={`process-association ${association?.state || 'unknown'}`} title={title}>{label}</span>
}

function ComputeDetails({ data, associationSupported }: { data: ComputeData; associationSupported: boolean }) {
  const processes = useMemo(() => processRows(data?.gpus || []), [data])
  const summary = useMemo(() => {
    const gpus = data?.gpus || []
    const utilizations = gpus.map((gpu) => gpu.utilization).filter((value): value is number => value != null)
    return {
      utilization: utilizations.length ? Math.round(utilizations.reduce((sum, value) => sum + value, 0) / utilizations.length) : null,
      memoryUsed: gpus.reduce((sum, gpu) => sum + (gpu.memory_used || 0), 0),
      memoryTotal: gpus.reduce((sum, gpu) => sum + (gpu.memory_total || 0), 0),
    }
  }, [data])

  const sampledAt = data ? new Date(data.sampled_at).toLocaleTimeString('zh-CN', { hour12: false }) : '—'

  return <>
      <section className="compute-overview">
        <article><span><Cpu size={15} />GPU 设备</span><strong>{data.gpus.length}</strong><small>{data.provider || '未检测到设备'}</small></article>
        <article><span><Gauge size={15} />平均利用率</span><strong>{summary.utilization == null ? '—' : `${summary.utilization}%`}</strong><small>当前所有 GPU 平均值</small></article>
        <article><span><MemoryStick size={15} />显存使用</span><strong>{memoryLabel(summary.memoryUsed)}</strong><small>总计 {memoryLabel(summary.memoryTotal)}</small></article>
        <article><span><Activity size={15} />计算进程</span><strong>{processes.length}</strong><small>采样时间 {sampledAt}</small></article>
      </section>

      {data.available && data.message && <div className="node-info">{data.message}</div>}
      {!data.available ? <section className="compute-empty"><Cpu size={38} /><strong>暂无可用的 NVIDIA GPU 信息</strong><span>{data.message}</span><small>请检查这台服务器的 NVIDIA 驱动与 nvidia-smi。</small></section> : <section className="gpu-grid">
        {data.gpus.map((gpu) => {
          const utilization = gpu.utilization || 0
          const memoryPercent = ratio(gpu.memory_used, gpu.memory_total)
          const powerPercent = ratio(gpu.power_draw, gpu.power_limit)
          const busy = utilization > 5 || (gpu.processes?.length || 0) > 0
          return <article className="gpu-card" key={gpu.uuid || gpu.index}>
            <header><div className="gpu-index"><Cpu size={17} /></div><div><span>GPU {gpu.index}</span><h2>{gpu.name}</h2></div><em className={busy ? 'busy' : 'idle'}><i />{busy ? '使用中' : '空闲'}</em></header>
            <div className="gpu-main">
              <div className="usage-gauge" style={{ '--gauge-value': `${utilization * 3.6}deg` } as CSSProperties}><div><strong>{valueLabel(gpu.utilization, '%')}</strong><span>GPU 利用率</span></div></div>
              <div className="gpu-bars">
                <div><label><span>显存</span><strong>{memoryLabel(gpu.memory_used)} / {memoryLabel(gpu.memory_total)}</strong></label><div><i style={{ width: `${memoryPercent}%` }} /></div><small>{memoryPercent.toFixed(1)}% 已使用</small></div>
                <div><label><span>功耗</span><strong>{valueLabel(gpu.power_draw, ' W', 1)} / {valueLabel(gpu.power_limit, ' W', 0)}</strong></label><div className="power"><i style={{ width: `${powerPercent}%` }} /></div><small>{powerPercent.toFixed(1)}% 额定功耗</small></div>
              </div>
            </div>
            <div className="gpu-details">
              <div><Thermometer size={14} /><span>温度</span><strong>{valueLabel(gpu.temperature, '°C')}</strong></div>
              <div><Zap size={14} /><span>图形时钟</span><strong>{valueLabel(gpu.graphics_clock, ' MHz')}</strong></div>
              <div><MemoryStick size={14} /><span>显存时钟</span><strong>{valueLabel(gpu.memory_clock, ' MHz')}</strong></div>
              <div><HardDrive size={14} /><span>驱动版本</span><strong>{gpu.driver_version || '—'}</strong></div>
            </div>
          </article>
        })}
      </section>}

      <section className="compute-lower-grid">
        <article className="compute-panel process-panel">
          <header><div><Activity size={17} /><span><strong>GPU 计算进程</strong><small>来自 nvidia-smi 的实时占用信息</small></span></div><em>{processes.length} 个进程</em></header>
          {!associationSupported && <div className="process-attribution-note">当前 Agent 只上报 GPU PID。升级到 1.9.0 后可识别远程训练任务；现有训练不受影响。</div>}
          {processes.length ? <div className="process-table"><table><thead><tr><th>GPU</th><th>PID</th><th>进程</th><th>关联训练任务</th><th>占用显存</th></tr></thead><tbody>{processes.map((process) => (
            <tr key={`${process.gpuIndex}-${process.pid}`}><td>GPU {process.gpuIndex}</td><td className="mono">{process.pid}</td><td>{process.name}</td><td><AssociationCell process={process} /></td><td>{memoryLabel(process.memory_used)}</td></tr>
          ))}</tbody></table></div> : <div className="no-process"><Activity size={22} /><span>当前没有 GPU 计算进程</span></div>}
        </article>

        <article className="compute-panel host-panel">
          <header><div><Server size={17} /><span><strong>当前主机</strong><small>从运行环境动态读取</small></span></div></header>
          <div className="host-name"><Server size={20} /><span><small>HOSTNAME</small><strong>{data.host.hostname}</strong></span></div>
          <dl><div><dt>逻辑 CPU</dt><dd>{data.host.cpu_count ?? '—'} 核</dd></div><div><dt>系统内存</dt><dd>{data.host.memory_used == null ? '—' : formatBytes(data.host.memory_used)} / {data.host.memory_total == null ? '—' : formatBytes(data.host.memory_total)}</dd></div><div><dt>负载均值</dt><dd>{data.host.load_average.length ? data.host.load_average.join(' / ') : '—'}</dd></div><div><dt>系统</dt><dd title={data.host.platform}>{data.host.platform}</dd></div></dl>
        </article>
      </section>
  </>
}

const statusNames = { online: '在线', offline: '离线', pending: '等待接入', disabled: '已禁用' }
const storageKey = 'yolo-compute-expanded-nodes'
const refreshIntervalMs = 5000

function dateLabel(value: string | null): string {
  return value ? new Date(value).toLocaleString('zh-CN', { hour12: false }) : '尚未上报'
}

function refreshResultLabel(outcome: ComputeNodeRefreshResult, kind: ComputeNode['kind']): string {
  switch (outcome.result) {
    case 'fresh': return kind === 'local' ? '本机已完成新采样' : 'Agent 已上报新采样'
    case 'stale': return '未确认新采样，显示现有数据'
    case 'timeout': return '等待即时采样超时，显示现有数据'
    case 'unsupported': return '无法请求即时采样，显示现有数据'
    case 'error': return '即时采样失败，显示现有数据'
  }
}

export function ComputeResources() {
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [details, setDetails] = useState<Record<string, ComputeNode>>({})
  const [detailErrors, setDetailErrors] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [manualRefreshing, setManualRefreshing] = useState(false)
  const [refreshOutcomes, setRefreshOutcomes] = useState<Record<string, ComputeNodeRefreshResult>>({})
  const [refreshMessage, setRefreshMessage] = useState('')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [countdown, setCountdown] = useState(refreshIntervalMs / 1000)
  const [pageVisible, setPageVisible] = useState(() => !document.hidden)
  const [error, setError] = useState('')
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState('all')
  const [expanded, setExpanded] = useState<string[]>(() => {
    try {
      const saved: unknown = JSON.parse(localStorage.getItem(storageKey) || 'null')
      return Array.isArray(saved) && saved.every((item) => typeof item === 'string') ? saved : ['local']
    } catch { return ['local'] }
  })
  const [modal, setModal] = useState<{ node?: ComputeNode; connect?: boolean } | null>(null)
  const [removing, setRemoving] = useState('')
  const expandedRef = useRef(expanded)
  expandedRef.current = expanded
  const controller = useRef<AbortController | null>(null)
  const requestId = useRef(0)
  const inFlight = useRef(false)
  const manualInFlight = useRef(false)
  const nextRefreshAt = useRef(0)

  const visible = useMemo(() => nodes.filter((node) => (status === 'all' || node.status === status)
    && `${node.name} ${node.address} ${node.hostname || ''} ${node.notes}`.toLowerCase().includes(query.trim().toLowerCase())), [nodes, query, status])
  const filters = useRef({ query, status })
  filters.current = { query, status }
  const lastFilter = useRef(`${status}\0${query}`)

  const load = useCallback(async (force = false) => {
    // Manual sampling owns the page until it finishes; automatic reads stay cache-only.
    if (manualInFlight.current || (inFlight.current && !force)) return
    controller.current?.abort()
    const current = new AbortController()
    controller.current = current
    const version = ++requestId.current
    inFlight.current = true
    setRefreshing(true)
    try {
      const list = await api.computeNodes(current.signal)
      if (version !== requestId.current) return
      setNodes(list); setError('')
      const ids = list.filter((node) => expandedRef.current.includes(node.id)
        && (filters.current.status === 'all' || node.status === filters.current.status)
        && `${node.name} ${node.address} ${node.hostname || ''} ${node.notes}`.toLowerCase().includes(filters.current.query.trim().toLowerCase())).map((node) => node.id)
      const results = await Promise.allSettled(ids.map((id) => api.computeNode(id, current.signal)))
      if (version !== requestId.current) return
      const successful: Record<string, ComputeNode> = {}
      const errors: Record<string, string> = {}
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') successful[ids[index]] = result.value
        else errors[ids[index]] = result.reason instanceof Error ? result.reason.message : '服务器详情读取失败'
      })
      setDetails((previous) => ({ ...previous, ...successful })); setDetailErrors(errors)
      setRefreshOutcomes((previous) => Object.fromEntries(Object.entries(previous).filter(([id, outcome]) => {
        const latest = list.find((node) => node.id === id)
        return latest && latest.sampled_at === outcome.sampled_at
      })))
    } catch (reason) {
      if (current.signal.aborted) return
      setError(reason instanceof SyntaxError ? '算力接口尚未就绪，请重启后端以加载新功能。' : reason instanceof Error ? reason.message : '算力信息加载失败')
    } finally {
      if (version === requestId.current) {
        inFlight.current = false
        nextRefreshAt.current = performance.now() + refreshIntervalMs
        setCountdown(refreshIntervalMs / 1000)
        setLoading(false); setRefreshing(false)
      }
    }
  }, [])

  const refreshNow = useCallback(async () => {
    if (manualInFlight.current) return
    // A click supersedes an in-flight cache read, but never another manual sample.
    controller.current?.abort()
    const current = new AbortController()
    controller.current = current
    const version = ++requestId.current
    manualInFlight.current = true
    inFlight.current = true
    setManualRefreshing(true)
    setRefreshing(true)
    setRefreshMessage('正在等待各服务器的新采样…')
    setRefreshOutcomes({})
    try {
      const response = await api.refreshComputeNodes(current.signal)
      if (version !== requestId.current) return
      const refreshed = response.nodes
      setNodes(refreshed.map((entry) => entry.node))
      setDetails(Object.fromEntries(refreshed.map((entry) => [entry.id, entry.node])))
      setRefreshOutcomes(Object.fromEntries(refreshed.map((entry) => [entry.id, entry])))
      setDetailErrors({})
      setError('')
      const fresh = refreshed.filter((entry) => entry.result === 'fresh').length
      setRefreshMessage(refreshed.length === 0 ? '上次手动刷新：当前没有服务器可刷新。'
        : fresh === refreshed.length ? `上次手动刷新：${fresh} 台服务器已取得新采样。`
          : `上次手动刷新：${fresh}/${refreshed.length} 台服务器已确认即时采样；其余服务器请查看各节点提示。`)
    } catch (reason) {
      if (current.signal.aborted) return
      if ([404, 405].includes((reason as Error & { status?: number })?.status || 0)) {
        // A newly built UI may be served by an older backend until its next safe restart.
        // Read its existing cache, and make clear that a fresh sample was not requested.
        try {
          const list = await api.computeNodes(current.signal)
          const ids = list.filter((node) => expandedRef.current.includes(node.id)).map((node) => node.id)
          const results = await Promise.allSettled(ids.map((id) => api.computeNode(id, current.signal)))
          if (version !== requestId.current) return
          const cached: Record<string, ComputeNode> = {}
          results.forEach((result, index) => {
            if (result.status === 'fulfilled') cached[ids[index]] = result.value
          })
          setNodes(list)
          setDetails((previous) => ({ ...previous, ...cached }))
          setRefreshMessage('上次手动刷新：后端尚未加载即时刷新接口；本次仅重新读取现有缓存，没有触发新采样。请在安全窗口重启开发版后端。')
          setError('')
        } catch (cacheError) {
          if (current.signal.aborted) return
          setRefreshMessage('')
          setError(cacheError instanceof Error ? cacheError.message : '算力信息加载失败')
        }
        return
      }
      setRefreshMessage('')
      setError(reason instanceof Error ? `立即刷新失败：${reason.message}` : '立即刷新失败，请稍后重试')
    } finally {
      if (version === requestId.current) {
        manualInFlight.current = false
        inFlight.current = false
        nextRefreshAt.current = performance.now() + refreshIntervalMs
        setCountdown(refreshIntervalMs / 1000)
        setLoading(false)
        setRefreshing(false)
        setManualRefreshing(false)
      }
    }
  }, [])

  useEffect(() => {
    void load()
    const mountedRequestId = requestId.current
    return () => {
      controller.current?.abort()
      requestId.current = mountedRequestId + 1
      inFlight.current = false
      manualInFlight.current = false
    }
  }, [load])

  useEffect(() => {
    // Count down to the actual request deadline, rather than decrementing a counter.
    // Each completed request starts a fresh interval so slow requests never overlap.
    nextRefreshAt.current = performance.now() + refreshIntervalMs
    setCountdown(refreshIntervalMs / 1000)
    const tick = () => {
      if (document.hidden || inFlight.current) return
      const remaining = Math.max(0, Math.ceil((nextRefreshAt.current - performance.now()) / 1000))
      setCountdown(remaining)
      if (remaining === 0) void load()
    }
    const timer = autoRefresh ? window.setInterval(tick, 100) : undefined
    const focusRefresh = () => {
      setPageVisible(!document.hidden)
      if (autoRefresh && !document.hidden) void load()
    }
    document.addEventListener('visibilitychange', focusRefresh)
    return () => { window.clearInterval(timer); document.removeEventListener('visibilitychange', focusRefresh) }
  }, [load, autoRefresh])

  useEffect(() => {
    const key = `${status}\0${query}`
    if (lastFilter.current === key) return
    const timer = window.setTimeout(() => { lastFilter.current = key; void load(true) }, 250)
    return () => window.clearTimeout(timer)
  }, [query, status, load])

  useEffect(() => {
    try { localStorage.setItem(storageKey, JSON.stringify(expanded)) } catch { /* Browser storage may be disabled. */ }
  }, [expanded])

  function toggle(id: string) {
    const next = expanded.includes(id) ? expanded.filter((item) => item !== id) : [...expanded, id]
    expandedRef.current = next
    setExpanded(next)
    if (next.includes(id)) void load(true)
  }

  async function remove(node: ComputeNode) {
    if (!window.confirm(`移除服务器“${node.name}”？该节点的 Token 会失效；远程文件和训练进程不会被删除。`)) return
    setRemoving(node.id)
    try {
      await api.deleteComputeNode(node.id)
      setExpanded((previous) => previous.filter((id) => id !== node.id))
      setDetails((previous) => { const next = { ...previous }; delete next[node.id]; return next })
      await load(true)
    } catch (reason) { setError(reason instanceof Error ? reason.message : '移除失败') }
    finally { setRemoving('') }
  }

  return <div className="page compute-page">
    <header className="page-header compute-header">
      <div><span className="eyebrow">COMPUTE MONITOR</span><h1>算力资源</h1><p>按服务器查看 GPU 负载、显存、功耗和计算进程，点击名称展开详情。</p></div>
      <div className="compute-actions">
        <button className={`auto-refresh ${autoRefresh ? 'active' : ''}`} aria-pressed={autoRefresh} title={autoRefresh ? '点击暂停自动刷新' : '点击开启自动刷新'} onClick={() => setAutoRefresh((value) => !value)}><i />{!autoRefresh ? '自动刷新已暂停' : !pageVisible ? '后台暂停刷新' : refreshing ? '刷新中…' : `下次刷新 ${countdown}s`}</button>
        <button className="button" disabled={manualRefreshing} onClick={() => void refreshNow()} title="等待本机新采样及远程 Agent 上报"><RefreshCw size={15} className={manualRefreshing ? 'spin' : ''} />{manualRefreshing ? '正在采样…' : '立即刷新'}</button>
        <button className="button primary" onClick={() => setModal({})}><Plus size={16} />添加服务器</button>
      </div>
    </header>
    <div className="node-toolbar">
      <label className="node-search"><Search size={16} /><input aria-label="搜索服务器" placeholder="搜索名称、地址或主机名" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
      <select className="node-status-filter" aria-label="服务器状态筛选" value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">全部状态</option>{Object.entries(statusNames).map(([key, value]) => <option key={key} value={key}>{value}</option>)}</select>
      <span>{nodes.length} 台服务器 · {nodes.filter((node) => node.status === 'online').length} 台在线</span>
      <button className="button compact" onClick={() => { const next = [...new Set([...expanded, ...visible.map((node) => node.id)])]; expandedRef.current = next; setExpanded(next); void load(true) }}>展开全部</button>
      <button className="button compact" onClick={() => { expandedRef.current = []; setExpanded([]) }}>折叠全部</button>
    </div>
    {error && <div className="alert error" role="alert">{error} {nodes.length > 0 && '当前保留的是此前数据，请刷新重试。'}</div>}
    {refreshMessage && <div className="node-refresh-message" role="status">{refreshMessage}</div>}
    {loading && !nodes.length && <div className="compute-loading"><RefreshCw className="spin" size={23} />正在读取服务器算力信息…</div>}
    <div className="node-list">{visible.map((node) => {
      const open = expanded.includes(node.id)
      const data = details[node.id]?.compute
      const refreshOutcome = refreshOutcomes[node.id]
      const trainingUnavailableReason = node.training_unavailable_reason
      return <section className={`node-section ${open ? 'expanded' : ''}`} key={node.id}>
        <div className="node-heading">
          <button className="node-toggle" aria-expanded={open} aria-controls={`node-${node.id}`} onClick={() => toggle(node.id)}>
            <ChevronDown size={18} className={open ? 'rotated' : ''} /><Server size={20} />
            <span className="node-identity"><strong>{node.name}<em>{node.kind === 'local' ? '本机' : '远程'}</em></strong><small>{node.address || node.hostname || '等待 Agent 上报主机信息'}{node.notes ? ` · ${node.notes}` : ''}</small></span>
            <span className={`node-status ${node.status}`}><i />{statusNames[node.status]}</span>
            {refreshOutcome && <span className={`node-refresh-badge ${refreshOutcome.result}`} title={refreshOutcome.reason || undefined}>{refreshOutcome.result === 'fresh' ? '已更新' : '未确认即时采样'}</span>}
            <span className="node-brief">{node.summary ? `${node.summary.gpu_count} GPU · ${node.summary.utilization == null ? '—' : `${node.summary.utilization}%`} · ${memoryLabel(node.summary.memory_used)} 显存` : '暂无采样'}</span>
          </button>
          <div className="node-controls">
            <button className="icon-button" title="编辑服务器" aria-label={`编辑${node.name}`} onClick={() => setModal({ node })}><Pencil size={15} /></button>
            {node.kind === 'agent' && <><button className="icon-button" title="Agent 接入配置" aria-label={`${node.name}的接入配置`} onClick={() => setModal({ node, connect: true })}><Settings2 size={16} /></button><button className="icon-button" title="移除服务器" aria-label={`移除${node.name}`} disabled={removing === node.id} onClick={() => void remove(node)}><Trash2 size={15} /></button></>}
          </div>
        </div>
        {open && <div className="node-content" id={`node-${node.id}`}>
          <div className="node-meta"><span>{node.kind === 'local' ? '最后采样' : '最后上报'}：{dateLabel(node.last_seen_at)}</span>{node.kind === 'agent' && node.sampled_at && <span>采样时间：{dateLabel(node.sampled_at)}</span>}{node.kind === 'agent' && <span>上报间隔：{node.report_interval_seconds}s · 超过 {node.offline_after_seconds}s 无新数据显示离线</span>}{node.agent_version && <span>Agent v{node.agent_version}</span>}</div>
          {refreshOutcome && <div className={`node-refresh-result ${refreshOutcome.result}`} role="status"><strong>{refreshResultLabel(refreshOutcome, node.kind)}</strong>{refreshOutcome.reason && <span> · {refreshOutcome.reason}</span>}</div>}
          {node.kind === 'agent' && <div className={`node-training-state ${trainingUnavailableReason ? 'not-ready' : 'ready'}`} role="status">
            <strong>{trainingUnavailableReason ? '暂不可训练' : '训练已就绪'}</strong>
            <span>{trainingUnavailableReason || '可接受远程训练任务'}</span>
          </div>}
          {node.kind === 'agent' && <details className="node-info">
            <summary>训练输出允许目录 · {node.training?.output_roots?.length || 0} 项</summary>
            {node.training?.output_roots?.length ? <>
              <p>以下为 Agent 最近上报的实际目录；新建训练任务可选择这些目录或其已存在的子目录。如需新目录，请先在远程服务器创建。</p>
              {node.training.output_roots.map((root) => <div key={root}><code style={{ overflowWrap: 'anywhere' }}>{root}</code></div>)}
            </> : <p>Agent 尚未上报允许目录。请核对远程配置；升级 Agent 后会显示实际解析的目录。</p>}
          </details>}
          {node.status === 'offline' && <div className="node-warning">服务器已离线。下方为最后一次采样，当前 GPU 负载和进程状态待确认。</div>}
          {node.status === 'disabled' && <div className="node-warning">监控已禁用，Agent 上报将被拒绝。下方为此前保存的数据。</div>}
          {node.status === 'pending' && <div className="node-info">等待 Agent 接入。点击右侧“接入配置”，在对应服务器启动只读 Agent。</div>}
          {detailErrors[node.id] && <div className="alert error">{detailErrors[node.id]} · 此前采样尚未刷新</div>}
          {data ? <ComputeDetails data={data} associationSupported={node.process_attribution_supported} />
            : <div className="node-no-data"><Server size={26} /><span>{details[node.id] ? '这台服务器尚未上报算力数据' : '正在读取服务器详情…'}</span>{node.kind === 'agent' && <button className="button compact" onClick={() => setModal({ node, connect: true })}>查看接入步骤</button>}</div>}
        </div>}
      </section>
    })}</div>
    {!loading && !visible.length && <div className="node-no-data"><Search size={26} /><span>没有匹配的服务器</span></div>}
    {modal && <ComputeNodeModal node={modal.node} connect={modal.connect} onClose={() => setModal(null)} onSaved={() => void load(true)} />}
  </div>
}
