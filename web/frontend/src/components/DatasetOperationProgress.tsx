import type { DatasetOperation } from '../types'
import { formatBytes } from '../api'
import { datasetPhaseLabel } from '../datasetUploadSession'
import { datasetOperationWait } from '../datasetOperationState'

const stages: Record<string, string> = {
  scan: '扫描与文件校验', source_verify: '核对源文件一致性', pack: '打包文件',
  archive_hash: '校验归档', transfer: '传输数据', unpack: '解包与文件校验',
  content_validate: '检查图片与标签', target_verify: '校验目标文件',
  delete_verify: '核对删除前的文件指纹', remove: '清理文件', check: '检查保存位置',
  waiting_lock: '等待文件或工作区锁', throttled: '按资源预算限速读取',
  paused_resource: '资源繁忙，后台操作已暂停',
}

function duration(seconds: number) {
  if (seconds < 60) return `${Math.max(0, Math.floor(seconds))} 秒`
  if (seconds < 3600) return `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒`
  return `${Math.floor(seconds / 3600)} 小时 ${Math.floor(seconds % 3600 / 60)} 分`
}

function age(time?: string | null) {
  return time ? Math.max(0, (Date.now() - new Date(time).getTime()) / 1000) : null
}

export function DatasetOperationProgress({ operation: job, detailed = false }: { operation: DatasetOperation; detailed?: boolean }) {
  const active = ['pending', 'running', 'committing'].includes(job.status)
  const waiting = datasetOperationWait(job)
  const progress = job.progress
  const stage = progress && progress.phase === job.phase ? progress : null
  const pendingLabel = job.queue_ahead ? `排队中 · 前面还有 ${job.queue_ahead} 个操作` : job.node_id === 'local' ? '等待本机执行' : '等待远程 Agent 领取'
  const label = waiting?.label || (job.status === 'pending' ? pendingLabel : stages[stage?.stage || ''] || datasetPhaseLabel[job.phase] || job.phase)
  const unitLabel = stage?.unit === 'directories' ? '个目录' : '个文件'
  const measured = stage && (stage.total != null || ['scan', 'source_verify', 'target_verify', 'delete_verify', 'discover'].includes(stage.stage))
  const count = stage ? stage.unit === 'bytes' ? formatBytes(stage.processed) : `${stage.processed.toLocaleString()} ${unitLabel}` : ''
  const total = stage?.total != null ? stage.unit === 'bytes' ? formatBytes(stage.total) : `${stage.total.toLocaleString()} ${unitLabel}` : ''
  const percent = stage?.total != null && stage.total > 0 ? Math.min(100, stage.processed / stage.total * 100) : null
  const elapsed = age(job.started_at)
  const waited = age(job.queued_at || job.created_at)
  const heartbeat = age(job.heartbeat_at)
  const changed = age(stage?.changed_at)
  const executor = stage?.executor === 'local' ? '本机服务器' : stage?.executor === job.node_id ? job.node_name || '远程服务器' : '远程服务器'
  const terminal = job.status === 'completed' ? '已完成' : job.status === 'failed' ? '失败时阶段' : job.status === 'cancelled' ? '已取消' : ''
  const stageLabel = stages[stage?.stage || ''] || datasetPhaseLabel[job.phase] || job.phase
  const showProgress = !waiting || waiting.reason === 'resource'
  return <div className={`dataset-operation-progress ${detailed ? 'detailed' : ''} ${waiting?.reason === 'resource' ? 'paused' : ''}`}>
    <div className="dataset-progress-heading"><strong>{terminal ? `${terminal} · ${stageLabel}` : label}</strong>{showProgress && measured && <span>{count}{total && ` / ${total}`}{percent != null && ` · ${percent.toFixed(1)}%（本阶段）`}</span>}</div>
    {active && showProgress && job.status !== 'pending' && <progress aria-label={`${stageLabel}阶段进度`} max={100} {...(percent == null ? {} : { value: percent })} />}
    {active && !waiting && !stage && <small>当前阶段：{datasetPhaseLabel[job.phase] || job.phase} · 详细进度暂不可用</small>}
    {active && !waiting && changed != null && changed > 60 && <small className="dataset-progress-stale">进度暂未变化，保留最后记录；不代表操作失败。</small>}
    {active && waiting?.reason === 'resource' && <small>资源恢复后自动继续，已完成的进度会保留。</small>}
    {detailed && <>
      <div className="dataset-progress-meta">
        {stage && <span>执行位置：{executor}</span>}
        {active && job.status === 'pending' && waited != null && <span>已排队 {duration(waited)}</span>}
        {elapsed != null && active && <span>已运行 {duration(elapsed)}</span>}
        {heartbeat != null && <span>最近心跳：{duration(heartbeat)}前</span>}
        {changed != null && <span>最近进展：{duration(changed)}前</span>}
        {!waiting && active && stage?.rate != null && stage.rate > 0 && <span>阶段平均速度：{stage.unit === 'bytes' ? formatBytes(stage.rate) : stage.rate.toFixed(1) + (stage.unit === 'directories' ? ' 目录' : ' 文件')}/秒{stage.eta_seconds != null && ` · 本阶段预计剩余 ${duration(stage.eta_seconds)}`}</span>}
      </div>
      {!!job.progress_steps?.length && <ol className="dataset-progress-steps">{job.progress_steps.map((step, index) => <li key={index}>✓ {stages[step.stage] || datasetPhaseLabel[step.phase] || step.stage} · {step.executor === 'local' ? '本机' : '远程'}</li>)}<li>{job.status === 'completed' ? '✓' : '●'} {stageLabel}</li></ol>}
      {stage?.current_file && <code title={stage.current_file}>当前文件：{stage.current_file}</code>}
      {active && heartbeat != null && heartbeat > 30 && <small className="dataset-progress-stale">执行心跳暂未更新，请结合节点连接状态确认。</small>}
      {job.kind === 'refresh' && active && !waiting && <small>重新加载使用只读校验；支持并行读取的节点可在训练期间执行，完成前沿用上次统计。</small>}
      {job.kind === 'refresh' && ['failed', 'cancelled'].includes(job.status) && <small>本次未更新数据集统计，仍保留上次成功校验的结果。</small>}
    </>}
  </div>
}
