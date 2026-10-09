import { useCallback, useEffect, useMemo, useState } from 'react'
import { AlertTriangle, ArrowUpRight, CheckCircle2, Clock3, Copy, Cpu, GitCompareArrows, MoreHorizontal, Plus, RefreshCw, Search, Square, Trash2, Zap } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { api, metricValue, statusLabel } from '../api'
import type { TrainingTask } from '../types'
import { CreateTaskModal } from './CreateTaskModal'

const statusIcon = (status: string) => status === 'completed' ? <CheckCircle2 size={15} /> : status === 'failed' || status === 'interrupted' ? <AlertTriangle size={15} /> : status === 'running' ? <Zap size={15} /> : <Clock3 size={15} />

export function Dashboard() {
  const [tasks, setTasks] = useState<TrainingTask[]>([])
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all')
  const [creating, setCreating] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  const refresh = useCallback(async () => {
    try { setTasks(await api.tasks()); setError('') } catch (reason) { setError(reason instanceof Error ? reason.message : '加载失败') } finally { setLoading(false) }
  }, [])

  useEffect(() => {
    refresh()
    const source = new EventSource('/api/stream')
    source.addEventListener('tasks', (event) => setTasks(JSON.parse((event as MessageEvent).data)))
    source.onerror = () => undefined
    return () => source.close()
  }, [refresh])

  const filtered = useMemo(() => tasks.filter((task) => {
    const matches = task.name.toLowerCase().includes(search.toLowerCase()) || task.model_path.toLowerCase().includes(search.toLowerCase())
    const state = filter === 'all' || task.status === filter || (filter === 'active' && ['queued', 'starting', 'running', 'stopping'].includes(task.status))
    return matches && state
  }), [tasks, search, filter])

  const active = tasks.filter((task) => ['queued', 'starting', 'running', 'stopping'].includes(task.status)).length
  const completed = tasks.filter((task) => task.status === 'completed').length
  const attention = tasks.filter((task) => ['failed', 'interrupted'].includes(task.status)).length

  async function taskAction(event: React.MouseEvent, action: () => Promise<unknown>) {
    event.stopPropagation()
    try { await action(); await refresh() } catch (reason) { setError(reason instanceof Error ? reason.message : '操作失败') }
  }

  return <div className="page dashboard-page">
    <header className="page-header">
      <div><span className="eyebrow">TRAINING OPERATIONS</span><h1>训练任务中心</h1><p>创建、调度并观察每一次模型实验。</p></div>
      <div className="header-actions"><button className="button ghost" onClick={() => api.importRuns().then(refresh)}><RefreshCw size={16} />同步历史任务</button><button className="button primary" onClick={() => setCreating(true)}><Plus size={17} />新建训练任务</button></div>
    </header>

    <section className="stat-row">
      <div className="stat-card"><span>全部任务</span><strong>{tasks.length}</strong><small>训练记录总数</small></div>
      <div className="stat-card accent"><span>正在进行</span><strong>{active}</strong><small><i />实时运行与排队</small></div>
      <div className="stat-card"><span>已完成</span><strong>{completed}</strong><small>成功产出模型</small></div>
      <div className="stat-card warning"><span>需要关注</span><strong>{attention}</strong><small>失败或意外中断</small></div>
    </section>

    <section className="task-section">
      <div className="section-toolbar">
        <div><h2>所有训练</h2><span>{filtered.length} 条记录</span></div>
        <div className="toolbar-actions"><div className="search"><Search size={16} /><input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="搜索任务或模型" /></div><select value={filter} onChange={(e) => setFilter(e.target.value)}><option value="all">全部状态</option><option value="active">进行中</option><option value="completed">已完成</option><option value="interrupted">已中断</option><option value="failed">失败</option><option value="draft">草稿</option></select></div>
      </div>
      {error && <div className="alert error">{error}</div>}
      {loading ? <div className="empty">正在加载任务…</div> : filtered.length === 0 ? <div className="empty"><Cpu size={28} /><strong>没有匹配的训练任务</strong><span>创建任务后，每一次训练都会拥有独立画布。</span></div> :
        <div className="task-grid">{filtered.map((task) => <article key={task.id} className="task-card" onClick={() => navigate(`/tasks/${task.id}/canvas`)}>
          <div className="task-card-top"><span className={`status ${task.status}`}>{statusIcon(task.status)}{statusLabel(task.status)}</span><button className="icon-button"><MoreHorizontal size={18} /></button></div>
          <h3>{task.name}</h3><p className="task-path">{task.model_path.split('/').pop()} · {task.task_type}</p>
          <div className="progress-heading"><span>Epoch {task.completed_epochs} / {task.epochs}</span><strong>{Math.round(task.progress)}%</strong></div>
          <div className="progress"><i style={{ width: `${task.progress}%` }} /></div>
          <div className="metric-strip"><div><span>mAP50</span><strong>{metricValue(task, ['metrics/mAP50(B)', 'metrics/mAP50(M)'])}</strong></div><div><span>Box loss</span><strong>{metricValue(task, ['train/box_loss'])}</strong></div><div><span>设备</span><strong>GPU {task.device}</strong></div></div>
          {task.error_message && <div className="task-warning"><AlertTriangle size={14} />{task.error_message}</div>}
          <footer><span>{task.source === 'imported' ? '历史导入' : new Date(task.created_at).toLocaleString('zh-CN')}</span><div>
            {['draft', 'stopped', 'failed', 'interrupted'].includes(task.status) && <button title="启动" onClick={(e) => taskAction(e, () => api.start(task.id))}><Zap size={15} /></button>}
            {['queued', 'starting', 'running'].includes(task.status) && <button title="停止" onClick={(e) => taskAction(e, () => api.stop(task.id))}><Square size={14} /></button>}
            <button title="克隆" onClick={(e) => taskAction(e, () => api.clone(task.id))}><Copy size={15} /></button>
            <button title="加入实验对比" onClick={(e) => { e.stopPropagation(); navigate(`/compare?tasks=${task.id}`) }}><GitCompareArrows size={15} /></button>
            {!['queued', 'starting', 'running', 'stopping'].includes(task.status) && <button title="删除" onClick={(e) => taskAction(e, () => api.remove(task.id))}><Trash2 size={15} /></button>}
            <button title="打开画布"><ArrowUpRight size={15} /></button>
          </div></footer>
        </article>)}</div>}
    </section>
    <CreateTaskModal open={creating} onClose={() => setCreating(false)} onCreated={(task) => { setCreating(false); navigate(`/tasks/${task.id}/canvas`) }} />
  </div>
}
