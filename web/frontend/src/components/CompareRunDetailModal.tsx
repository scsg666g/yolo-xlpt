import { useMemo, useState } from 'react'
import { Clock3, FileArchive, Info, ListTree, LoaderCircle, Monitor, RefreshCw, Server, Terminal, Trophy, X } from 'lucide-react'
import { formatBytes, statusLabel } from '../api'
import type { Artifact, EffectiveBestWeight, MetricPoint, TaskEvent, TrainingRun, TrainingTask } from '../types'
import { useModalFocus } from '../useModalFocus'
import { FullLogDownload } from './FullLogDownload'

export interface CompareRunDetails {
  loading: boolean
  loaded: boolean
  errors: string[]
  metrics?: MetricPoint[]
  artifacts?: Artifact[]
  bestWeight?: EffectiveBestWeight
  events?: TaskEvent[]
  logs?: { lines: string[]; path: string | null }
}

type DetailTab = 'overview' | 'artifacts' | 'events' | 'logs'

type Props = {
  task: TrainingTask
  run: TrainingRun
  view: TrainingTask
  details?: CompareRunDetails
  onClose: () => void
  onRefresh: () => void
}

const tabs: Array<{ id: DetailTab; label: string; icon: typeof Info }> = [
  { id: 'overview', label: '概览', icon: Info },
  { id: 'artifacts', label: '产物', icon: FileArchive },
  { id: 'events', label: '事件', icon: ListTree },
  { id: 'logs', label: '日志', icon: Terminal },
]

const launchModeLabel = { fresh: '首次训练', resume: '继续训练', retry: '重新训练' }

function formatDate(value: string | null | undefined) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function eventLabel(eventType: string): string {
  return ({
    stop_requested: '停止请求', force_interrupted: '手动标记中断', interrupted: '训练中断',
    stopped: '训练停止', failed: '训练失败', completed: '训练完成', remote_claimed: '远程节点领取',
    queued: '加入队列', started: '开始训练',
  } as Record<string, string>)[eventType] || eventType
}

function artifactPriority(artifact: Artifact) {
  const name = artifact.name.split('/').at(-1)
  if (artifact.kind !== 'weight') return 3
  return name === 'best.pt' ? 0 : name === 'last.pt' ? 1 : 2
}

function statusDetailTitle(status: TrainingRun['status']) {
  if (status === 'interrupted') return '中断运行详情'
  if (status === 'failed') return '失败运行详情'
  if (status === 'stopped') return '停止运行详情'
  return '训练运行详情'
}

export function CompareRunDetailModal({ task, run, view, details, onClose, onRefresh }: Props) {
  const [tab, setTab] = useState<DetailTab>('overview')
  const dialogRef = useModalFocus<HTMLDivElement>(true, onClose, { initialFocusSelector: '[data-modal-initial-focus]' })
  const remote = Boolean(run.execution_node_id)
  const lastMetricEpoch = details?.metrics?.at(-1)?.epoch
  const artifacts = useMemo(() => [...(details?.artifacts || [])]
    .sort((left, right) => artifactPriority(left) - artifactPriority(right) || left.name.localeCompare(right.name)), [details?.artifacts])

  return <div className="modal-backdrop compare-run-detail-backdrop" onMouseDown={onClose}>
    <div ref={dialogRef} className="modal compare-run-detail-modal" role="dialog" aria-modal="true" aria-labelledby="compare-run-detail-title" tabIndex={-1} onMouseDown={(event) => event.stopPropagation()}>
      <header>
        <div className="modal-heading"><div className="heading-icon compare-run-detail-icon">{remote ? <Server size={19} /> : <Monitor size={19} />}</div><div><span className="eyebrow">RUN DIAGNOSTICS · {remote ? 'REMOTE' : 'LOCAL'}</span><h2 id="compare-run-detail-title">{statusDetailTitle(run.status)}</h2><p>{task.name} · Attempt {String(run.attempt_number).padStart(3, '0')}</p></div></div>
        <div className="compare-run-detail-actions"><button type="button" className="button ghost compact" onClick={onRefresh} disabled={details?.loading} title="刷新运行详情">{details?.loading ? <LoaderCircle size={14} className="spin" /> : <RefreshCw size={14} />}刷新</button><button type="button" className="icon-button" data-modal-initial-focus title="关闭" aria-label="关闭运行详情弹窗" onClick={onClose}><X size={20} /></button></div>
      </header>

      <nav className="compare-run-detail-tabs" aria-label="运行详情分类">{tabs.map(({ id, label, icon: Icon }) => <button type="button" key={id} className={tab === id ? 'active' : ''} aria-pressed={tab === id} onClick={() => setTab(id)}><Icon size={15} />{label}{id === 'artifacts' && details?.artifacts ? <small>{details.artifacts.length}</small> : id === 'events' && details?.events ? <small>{details.events.length}</small> : null}</button>)}</nav>

      <div className="compare-run-detail-body">
        {details?.loading && !details.loaded && <div className="compare-run-detail-loading"><LoaderCircle size={18} className="spin" /><span>正在读取指标、产物、事件和日志…</span></div>}
        {details?.errors.length ? <div className="compare-run-detail-load-error" role="alert"><strong>部分详情读取失败</strong><span>{details.errors.join('；')}</span></div> : null}

        {tab === 'overview' && <div className="compare-run-detail-sections">
          <div className="compare-run-detail-summary">
            <div><span>运行状态</span><strong className={`status ${run.status}`}>{statusLabel(run.status)}</strong></div>
            <div><span>执行位置</span><strong>{remote ? '远程服务器' : '本机服务器'}</strong></div>
            <div><span>累计进度</span><strong>{view.completed_epochs} / {view.epochs} 轮</strong></div>
            <div><span>本次指标</span><strong>{details?.metrics ? `${details.metrics.length} 条${lastMetricEpoch != null ? ` · 最近第 ${lastMetricEpoch} 轮` : ''}` : details?.loading ? '读取中' : '暂无数据'}</strong></div>
          </div>

          <section className="compare-run-detail-section">
            <h3>运行信息</h3>
            <div className="compare-run-detail-grid">
              <div><span>启动方式</span><strong>{launchModeLabel[run.launch_mode]}</strong></div>
              <div><span>退出代码</span><strong>{run.exit_code ?? '—'}</strong></div>
              <div><span>开始时间</span><strong>{formatDate(run.started_at)}</strong></div>
              <div><span>结束时间</span><strong>{formatDate(run.finished_at)}</strong></div>
              <div><span>平台最后更新</span><strong>{formatDate(run.updated_at)}</strong></div>
              <div><span>运行标识</span><strong title={run.id}>{run.id}</strong></div>
              {remote && <div className="wide"><span>远程节点标识</span><strong title={run.execution_node_id || ''}>{run.execution_node_id}</strong></div>}
            </div>
          </section>

          {(run.error_message || ['failed', 'stopped', 'interrupted'].includes(run.status)) && <section className="compare-run-detail-reason">
            <h3>结束说明</h3><p>{run.error_message || '平台没有记录额外的结束说明，请结合运行事件和日志末尾判断。'}</p>
          </section>}

          <section className="compare-run-detail-weight">
            <h3><Trophy size={15} />最佳权重记录</h3>{details?.bestWeight?.artifact
              ? <p><strong>{details.bestWeight.artifact.name}</strong>{details.bestWeight.inherited && details.bestWeight.source_attempt_number ? ` · 来自 Attempt ${String(details.bestWeight.source_attempt_number).padStart(3, '0')}` : ''} · {formatBytes(details.bestWeight.artifact.size)}。这是产物记录，尚未验证文件可读或模型可加载。</p>
              : <p>{details?.loading ? '正在读取权重记录…' : remote ? '没有找到 best.pt 产物记录；远程 Agent 可能尚未完成产物同步。' : '没有找到 best.pt 产物记录。'}</p>}
          </section>

          {remote && <div className="compare-run-detail-source-note"><Info size={16} /><div><strong>远程同步记录</strong><p>当前展示平台已经收到的数据。日志是 Agent 最近同步的内容；如果节点在最终上报前离线，末尾指标、日志或产物清单可能不完整。</p></div></div>}
        </div>}

        {tab === 'artifacts' && <section className="compare-run-detail-section compare-run-detail-tab-section">
          <h3>产物记录 <span>{artifacts.length}</span></h3>
          {details?.artifacts ? artifacts.length ? <div className="compare-run-detail-artifacts">{artifacts.map((artifact) => <article key={artifact.id}><FileArchive size={16} /><div><strong>{artifact.name}</strong><span>{artifact.kind} · {formatBytes(artifact.size)} · {formatDate(artifact.modified_at)}</span></div></article>)}</div> : <div className="compare-run-detail-empty">{remote ? '当前没有已同步到平台的产物记录。' : '当前没有已登记的产物。'}</div> : <div className="compare-run-detail-empty">正在读取产物记录…</div>}
          <p className="compare-run-detail-footnote">产物记录只表示平台已登记该文件，不能证明文件仍可读取或模型能够正常加载。</p>
        </section>}

        {tab === 'events' && <section className="compare-run-detail-section compare-run-detail-tab-section">
          <h3>运行事件 <span>{details?.events?.length || 0}</span></h3>
          {details?.events ? details.events.length ? <div className="compare-run-detail-events">{details.events.map((event) => <article key={event.id}><i className={event.level} /><time>{formatDate(event.created_at)}</time><div><strong>{eventLabel(event.event_type)}</strong><p>{event.message}</p></div></article>)}</div> : <div className="compare-run-detail-empty">该 Attempt 没有独立运行事件记录。</div> : <div className="compare-run-detail-empty">正在读取运行事件…</div>}
        </section>}

        {tab === 'logs' && <section className="compare-run-detail-section compare-run-detail-tab-section compare-run-detail-log-section">
          <div className="compare-run-detail-section-heading"><div><h3>日志末尾</h3><p title={details?.logs?.path || ''}>{details?.logs?.path || '没有记录日志路径'}</p></div><Clock3 size={17} /></div>
          {remote && <p className="compare-run-detail-log-note">远程日志最多显示 Agent 最近同步到平台的内容。</p>}
          <FullLogDownload taskId={task.id} runId={run.id} />
          <pre>{details?.logs ? details.logs.lines.length ? details.logs.lines.join('\n') : '暂无日志内容' : '正在读取日志…'}</pre>
        </section>}
      </div>

      <footer><span>{remote ? '远程历史记录可离线查看；下载未缓存产物时仍需远程 Agent 在线' : '详情来自本机运行记录与训练文件'}</span><button type="button" className="button ghost" onClick={onClose}>关闭</button></footer>
    </div>
  </div>
}
