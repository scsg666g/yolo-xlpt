import { datasetPhaseLabel } from '../datasetUploadSession'
import { useEffect, useState } from 'react'
import { api, formatBytes } from '../api'
import type { ComputeNode, DatasetLocation, DatasetOperation, ManagedDataset } from '../types'
import { useModalFocus } from '../useModalFocus'


const jobStatus: Record<string, string> = { pending: '等待处理', running: '正在处理', committing: '正在提交文件', completed: '已完成', failed: '失败，可重试', cancelled: '已取消' }
const kindLabel: Record<string, string> = { check: '目录检查', upload: '上传', register: '登记', migrate: '复制 / 迁移', delete: '永久删除' }

export function DatasetStorageSettings() {
  const [locations, setLocations] = useState<DatasetLocation[]>([])
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [adding, setAdding] = useState(false)
  const [form, setForm] = useState({ node_id: 'local', name: '', path: '', create: false, is_default: false })
  useEffect(() => {
    let active = true
    const load = () => { void Promise.all([api.datasetLocations(), api.computeNodes()]).then(([ls, ns]) => { if (active) { setLocations(ls); setNodes(ns) } }).catch((e: Error) => { if (active) setError(e.message) }) }
    load()
    const timer = window.setInterval(load, 5000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])
  async function action(work: () => Promise<unknown>) {
    setBusy(true); setError('')
    try { await work(); setLocations(await api.datasetLocations()) } catch (e) { setError(e instanceof Error ? e.message : '目录操作失败') } finally { setBusy(false) }
  }
  return <section className="dataset-storage-panel">
    <div className="dataset-section-heading"><strong>数据集保存位置</strong><button type="button" className="button compact ghost" onClick={() => setAdding(!adding)}>{adding ? '收起' : '添加保存位置'}</button></div>
    <p>默认位置仅影响新上传；旧数据仍按登记的服务器和实际路径管理。相对路径以所选服务器项目目录为基准。</p>
    {error && <div className="alert error" role="alert">{error}</div>}
    {adding && <div className="form-grid dataset-location-form">
      <label>服务器<select value={form.node_id} disabled={busy} onChange={e => setForm({ ...form, node_id: e.target.value })}><option value="local">本机（中台）</option>{nodes.filter(n => n.kind === 'agent' && n.enabled).map(n => <option key={n.id} value={n.id}>{n.name} · {n.status}</option>)}</select></label>
      <label>位置名称<input value={form.name} disabled={busy} maxLength={160} placeholder="例如：训练数据盘" onChange={e => setForm({ ...form, name: e.target.value })} /></label>
      <label className="span-2">服务器目录<input value={form.path} disabled={busy} placeholder="例如：数据集，或已配置白名单内的绝对路径" onChange={e => setForm({ ...form, path: e.target.value })} /></label>
      <label><span><input type="checkbox" checked={form.create} onChange={e => setForm({ ...form, create: e.target.checked })} /> 目录不存在时创建</span></label>
      <label><span><input type="checkbox" checked={form.is_default} onChange={e => setForm({ ...form, is_default: e.target.checked })} /> 校验通过后设为默认</span></label>
      <button type="button" className="button primary" disabled={busy || !form.name.trim() || !form.path.trim()} onClick={() => void action(async () => { await api.createDatasetLocation(form); setAdding(false) })}>登记并校验</button>
    </div>}
    <div className="dataset-location-list">{locations.map(l => <article key={l.id}>
      <div><strong>{nodes.find(n => n.id === l.node_id)?.name || (l.node_id === 'local' ? '本机' : l.node_id)} · {l.name}{l.is_default ? ' · 默认' : ''}{!l.enabled ? ' · 已禁用' : ''}</strong>
        <code>{l.resolved_path || l.path}</code><small>{l.status === 'ready' ? '上次校验通过' : l.status === 'error' ? '校验失败' : '等待检查'}{l.details.free_bytes != null ? ` · 可用 ${formatBytes(l.details.free_bytes)} / ${formatBytes(l.details.total_bytes || 0)}` : ''}{l.details.checked_at ? ` · ${new Date(l.details.checked_at).toLocaleString()}` : ''}</small>
        {l.details.message && <small role="alert">{l.details.message}</small>}</div>
      <div className="dataset-inline-actions"><button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.checkDatasetLocation(l.id))}>检查</button>{l.status !== 'ready' && <button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.checkDatasetLocation(l.id, true))}>创建并检查</button>}
        <button type="button" className="button compact ghost" disabled={busy || l.status !== 'ready' || l.is_default} onClick={() => void action(() => api.updateDatasetLocation(l.id, true, true))}>设为默认</button>
        <button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.updateDatasetLocation(l.id, !l.enabled, false))}>{l.enabled ? '禁用' : '启用'}</button>
        <button type="button" className="button compact danger" disabled={busy || l.is_default} onClick={() => void action(() => api.forgetDatasetLocation(l.id))}>移除配置</button></div>
    </article>)}</div>
  </section>
}

export function DatasetOperationHistory({ onChanged }: { onChanged: () => void }) {
  const [jobs, setJobs] = useState<DatasetOperation[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')
  useEffect(() => {
    let active = true; let previous = ''
    const load = () => { void api.datasetOperations().then(incoming => {
      if (!active) return
      setJobs(incoming)
      const signature = incoming.map(j => `${j.id}:${j.status}:${j.phase}`).join(',')
      if (previous && previous !== signature) onChanged()
      previous = signature
    }).catch((e: Error) => { if (active) setError(e.message) }) }
    load(); const timer = window.setInterval(load, 3000)
    return () => { active = false; window.clearInterval(timer) }
  }, [onChanged])
  async function control(id: string, action: 'retry' | 'cancel') {
    setBusy(id); setError('')
    try { await api.controlDatasetOperation(id, action); setJobs(await api.datasetOperations()); onChanged() } catch (e) { setError(e instanceof Error ? e.message : '操作失败') } finally { setBusy('') }
  }
  return <details className="dataset-storage-panel" open={jobs.some(j => !['completed', 'cancelled'].includes(j.status))}><summary>操作历史 · {jobs.length} 条（最近 200 条）</summary>
    {error && <div className="alert error">{error}</div>}
    <div className="dataset-operation-list">{jobs.map(j => <article key={j.id}>
      <div><strong>{kindLabel[j.kind] || j.kind} {j.name} · {j.waiting_offline ? '等待远程节点上线' : j.waiting_training ? '等待正在训练的任务结束' : jobStatus[j.status] || j.status}</strong>
        <small>{datasetPhaseLabel[j.phase] || j.phase} · {new Date(j.created_at).toLocaleString()}</small>
        {(j.source_path || j.destination) && <code>{j.source_path || '浏览器上传'}{j.destination ? ` → ${j.destination}` : ''}</code>}
        {j.transferred > 0 && <small>已传 {formatBytes(j.transferred)}{j.total_bytes ? ` / ${formatBytes(j.total_bytes)}` : ''}</small>}
        {j.error && <p role="alert" className="dataset-operation-error">{j.error}</p>}
        {j.result.directory && <code>目标已保存：{j.result.directory}</code>}
      </div><div className="dataset-inline-actions">{j.status === 'failed' && <button type="button" className="button compact" disabled={busy === j.id} onClick={() => void control(j.id, 'retry')}>重试当前阶段</button>}{j.can_cancel && <button type="button" className="button compact ghost" disabled={busy === j.id} onClick={() => void control(j.id, 'cancel')}>取消</button>}</div>
    </article>)}{!jobs.length && <p>暂无数据集操作记录。</p>}</div>
  </details>
}

export function DatasetActionModal({ dataset, onClose, onDone }: { dataset?: ManagedDataset; onClose: () => void; onDone: () => void }) {
  const [locations, setLocations] = useState<DatasetLocation[]>([])
  const [target, setTarget] = useState('')
  const [name, setName] = useState(dataset?.name || '')
  const [remove, setRemove] = useState(false)
  const [ownership, setOwnership] = useState('external')
  const [acknowledge, setAcknowledge] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const ref = useModalFocus<HTMLFormElement>(true, onClose, { canClose: !busy })
  useEffect(() => { let active = true; void api.datasetLocations().then(ls => { if (active) { const valid = ls.filter(l => l.enabled && l.status === 'ready'); setLocations(valid); setTarget(valid.find(l => l.is_default)?.id || valid[0]?.id || '') } }).catch((e: Error) => { if (active) setError(e.message) }); return () => { active = false } }, [])
  async function submit(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError('')
    try {
      if (dataset) await api.migrateDataset(dataset.id, target, name, remove)
      else await api.registerDataset(target, name, ownership, acknowledge)
      onDone(); onClose()
    } catch (e) { setError(e instanceof Error ? e.message : '提交失败') } finally { setBusy(false) }
  }
  const location = locations.find(l => l.id === target)
  return <div className="modal-backdrop" onMouseDown={() => { if (!busy) onClose() }}><form ref={ref} className="modal" role="dialog" aria-modal="true" aria-labelledby="dataset-action-title" onSubmit={submit} onMouseDown={e => e.stopPropagation()}>
    <header><h2 id="dataset-action-title">{dataset ? '复制或迁移数据集' : '登记服务器已有数据集'}</h2></header>
    <div className="modal-body form-grid">
      {dataset && <p className="span-2">源目录：<code>{dataset.directory}</code>。原任务配置和历史路径保留；有引用时只能复制。</p>}
      <label className="span-2">{dataset ? '目标保存位置' : '已有目录所在位置'}<select required value={target} disabled={busy} onChange={e => setTarget(e.target.value)}>{locations.map(l => <option key={l.id} value={l.id}>{l.node_name || (l.node_id === 'local' ? '本机' : '远程')} · {l.name} · {l.resolved_path}</option>)}</select></label>
      <label className="span-2">数据集文件夹名称<input required maxLength={160} value={name} disabled={busy} onChange={e => setName(e.target.value)} placeholder="例如：安全帽_v2" /></label>
      <p className="span-2 dataset-destination-preview">实际目录：<code>{location?.resolved_path || '请先在保存位置中添加并校验目录'}/{name || '文件夹名称'}</code></p>
      {!dataset && <><p className="span-2">登记要求自包含 YOLO 检测目录（images/train、images/val、labels 和 data.yaml）；会完整校验图片、标签及 SHA-256。远程校验需经过中台传输。</p><label className="span-2">文件管理方式<select value={ownership} onChange={e => setOwnership(e.target.value)}><option value="external">仅登记：保留外部文件所有权</option><option value="managed">接管：允许平台永久删除文件</option></select></label>{ownership === 'managed' && <label className="span-2"><span><input type="checkbox" checked={acknowledge} onChange={e => setAcknowledge(e.target.checked)} />我确认该目录可由平台管理与删除</span></label>}</>}
      {dataset && <label className="span-2"><span><input type="checkbox" checked={remove} disabled={dataset.references.length > 0 || dataset.ownership !== 'managed'} onChange={e => setRemove(e.target.checked)} />目标校验保存成功后，删除源目录</span><small>未勾选时只复制。任何任务（含历史和回收站）引用源目录时不会清理源文件。</small></label>}
      {error && <div className="span-2 alert error" role="alert">{error}</div>}
    </div><footer><button type="button" className="button ghost" disabled={busy} onClick={onClose}>取消</button><button type="submit" className="button primary" disabled={busy || !target || (ownership === 'managed' && !dataset && !acknowledge)}>{busy ? '正在提交…' : '提交后台处理'}</button></footer>
  </form></div>
}
