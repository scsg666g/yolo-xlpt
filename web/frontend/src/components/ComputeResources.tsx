import { useCallback, useEffect, useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import { Activity, Cpu, Gauge, HardDrive, MemoryStick, RefreshCw, Server, Thermometer, Zap } from 'lucide-react'
import { api, formatBytes } from '../api'
import type { ComputeResources as ComputeData, GpuInfo, GpuProcess, TrainingTask } from '../types'

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

export function ComputeResources() {
  const [data, setData] = useState<ComputeData | null>(null)
  const [tasks, setTasks] = useState<TrainingTask[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [error, setError] = useState('')

  const loadCompute = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true)
    try {
      setData(await api.compute())
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '算力信息加载失败')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void loadCompute()
    api.tasks().then(setTasks).catch(() => undefined)
  }, [loadCompute])

  useEffect(() => {
    if (!autoRefresh) return
    const timer = window.setInterval(() => void loadCompute(), 5000)
    return () => window.clearInterval(timer)
  }, [autoRefresh, loadCompute])

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

  return <div className="page compute-page">
    <header className="page-header compute-header">
      <div><span className="eyebrow">COMPUTE MONITOR</span><h1>算力资源</h1><p>实时查看当前服务器的 GPU 负载、显存、功耗和计算进程。</p></div>
      <div className="compute-actions">
        <button className={`auto-refresh ${autoRefresh ? 'active' : ''}`} onClick={() => setAutoRefresh((value) => !value)}><i />自动刷新 5s</button>
        <button className="button" disabled={refreshing} onClick={() => void loadCompute(true)}><RefreshCw size={15} className={refreshing ? 'spin' : ''} />立即刷新</button>
      </div>
    </header>

    {error && <div className="alert error">{error}</div>}
    {loading && !data ? <div className="compute-loading"><RefreshCw className="spin" size={23} />正在读取服务器算力信息…</div> : data && <>
      <section className="compute-overview">
        <article><span><Cpu size={15} />GPU 设备</span><strong>{data.gpus.length}</strong><small>{data.provider || '未检测到设备'}</small></article>
        <article><span><Gauge size={15} />平均利用率</span><strong>{summary.utilization == null ? '—' : `${summary.utilization}%`}</strong><small>当前所有 GPU 平均值</small></article>
        <article><span><MemoryStick size={15} />显存使用</span><strong>{memoryLabel(summary.memoryUsed)}</strong><small>总计 {memoryLabel(summary.memoryTotal)}</small></article>
        <article><span><Activity size={15} />计算进程</span><strong>{processes.length}</strong><small>采样时间 {sampledAt}</small></article>
      </section>

      {!data.available ? <section className="compute-empty"><Cpu size={38} /><strong>暂无可用的 NVIDIA GPU 信息</strong><span>{data.message}</span><small>部署到其他服务器时，页面会自动读取新机器的设备。</small></section> : <section className="gpu-grid">
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
          {processes.length ? <div className="process-table"><table><thead><tr><th>GPU</th><th>PID</th><th>进程</th><th>关联训练任务</th><th>占用显存</th></tr></thead><tbody>{processes.map((process) => { const task = tasks.find((item) => item.pid === process.pid); return <tr key={`${process.gpuIndex}-${process.pid}`}><td>GPU {process.gpuIndex}</td><td className="mono">{process.pid}</td><td>{process.name}</td><td>{task ? <span className="linked-task">{task.name}</span> : <span className="muted-value">非平台任务</span>}</td><td>{memoryLabel(process.memory_used)}</td></tr> })}</tbody></table></div> : <div className="no-process"><Activity size={22} /><span>当前没有 GPU 计算进程</span></div>}
        </article>

        <article className="compute-panel host-panel">
          <header><div><Server size={17} /><span><strong>当前主机</strong><small>从运行环境动态读取</small></span></div></header>
          <div className="host-name"><Server size={20} /><span><small>HOSTNAME</small><strong>{data.host.hostname}</strong></span></div>
          <dl><div><dt>逻辑 CPU</dt><dd>{data.host.cpu_count ?? '—'} 核</dd></div><div><dt>系统内存</dt><dd>{data.host.memory_used == null ? '—' : formatBytes(data.host.memory_used)} / {data.host.memory_total == null ? '—' : formatBytes(data.host.memory_total)}</dd></div><div><dt>负载均值</dt><dd>{data.host.load_average.length ? data.host.load_average.join(' / ') : '—'}</dd></div><div><dt>系统</dt><dd title={data.host.platform}>{data.host.platform}</dd></div></dl>
        </article>
      </section>
    </>}
  </div>
}
