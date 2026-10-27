import { useEffect, useRef, useState } from 'react'
import { api, formatBytes } from '../api'
import type { ComputeNode, DatasetLocation, ManagedDataset } from '../types'
import { useModalFocus } from '../useModalFocus'
import { MoreHorizontal, X } from 'lucide-react'
import { DatasetServerSelect } from './DatasetServerSelect'


export function DatasetStorageSettings() {
  const section = useRef<HTMLElement>(null)
  const [locations, setLocations] = useState<DatasetLocation[]>([])
  const [nodes, setNodes] = useState<ComputeNode[]>([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [adding, setAdding] = useState(false)
  const [form, setForm] = useState({ node_id: 'local', name: '', path: '', create: false, is_default: false })
  useEffect(() => {
    let active = true
    let running = false
    let timer: ReturnType<typeof setTimeout> | undefined
    let controller: AbortController | null = null
    async function load() {
      if (!active || running || document.hidden) return
      running = true; controller = new AbortController()
      try {
        const [ls, ns] = await Promise.all([api.datasetLocations(controller.signal), api.computeNodes(controller.signal)])
        if (active) { setLocations(ls); setNodes(ns) }
      } catch (e) { if (active && !controller.signal.aborted) setError(e instanceof Error ? e.message : '保存位置读取失败') }
      finally { running = false; if (active) timer = setTimeout(load, 5000) }
    }
    function visible() { if (!document.hidden) { clearTimeout(timer); void load() } }
    void load(); document.addEventListener('visibilitychange', visible)
    return () => { active = false; controller?.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible) }
  }, [])
  useEffect(() => {
    const root = section.current
    function outside(event: PointerEvent) {
      root?.querySelectorAll<HTMLDetailsElement>('.dataset-location-more[open]').forEach(menu => {
        if (!menu.contains(event.target as Node)) menu.open = false
      })
    }
    function escape(event: KeyboardEvent) {
      if (event.key !== 'Escape' || event.isComposing) return
      const menu = root?.querySelector<HTMLDetailsElement>('.dataset-location-more[open]')
      if (menu) { event.preventDefault(); event.stopPropagation(); menu.open = false; menu.querySelector('summary')?.focus() }
    }
    document.addEventListener('pointerdown', outside)
    root?.addEventListener('keydown', escape)
    return () => { document.removeEventListener('pointerdown', outside); root?.removeEventListener('keydown', escape) }
  }, [])
  async function action(work: () => Promise<unknown>) {
    setBusy(true); setError('')
    try { await work(); setLocations(await api.datasetLocations()) } catch (e) { setError(e instanceof Error ? e.message : '目录操作失败') } finally { setBusy(false) }
  }
  return <section ref={section} className="dataset-storage-panel" aria-busy={busy}>
    <div className="dataset-section-heading"><strong>数据集保存位置</strong><button type="button" className="button compact ghost" onClick={() => setAdding(!adding)}>{adding ? '收起' : '添加保存位置'}</button></div>
    <p>默认位置仅影响新上传；旧数据仍按登记的服务器和实际路径管理。相对路径以所选服务器项目目录为基准。</p>
    {error && <div className="alert error" role="alert">{error}</div>}
    {adding && <div className="form-grid dataset-location-form">
      <label>服务器<DatasetServerSelect label="保存位置所在服务器" value={form.node_id} disabled={busy} onChange={v => setForm({ ...form, node_id: v })} options={[{ id: 'local', name: '本机（中台）' }, ...nodes.filter(n => n.kind === 'agent' && n.enabled).map(n => ({ id: n.id, name: n.name, status: n.status === 'online' ? '在线' : '离线' }))]} /></label>
      <label>位置名称<input value={form.name} disabled={busy} maxLength={160} placeholder="例如：训练数据盘" onChange={e => setForm({ ...form, name: e.target.value })} /></label>
      <label className="span-2">服务器目录<input value={form.path} disabled={busy} placeholder="例如：数据集，或已配置白名单内的绝对路径" onChange={e => setForm({ ...form, path: e.target.value })} /></label>
      <label><span><input type="checkbox" disabled={busy} checked={form.create} onChange={e => setForm({ ...form, create: e.target.checked })} /> 目录不存在时创建</span></label>
      <label><span><input type="checkbox" disabled={busy} checked={form.is_default} onChange={e => setForm({ ...form, is_default: e.target.checked })} /> 校验通过后设为默认</span></label>
      <button type="button" className="button primary" disabled={busy || !form.name.trim() || !form.path.trim()} onClick={() => void action(async () => { await api.createDatasetLocation(form); setAdding(false) })}>登记并校验</button>
    </div>}
    <div className="dataset-location-groups">{Array.from(new Set(locations.map(l => l.node_id))).map(nodeId => <section key={nodeId} className="dataset-location-group"><h3>{nodes.find(n => n.id === nodeId)?.name || (nodeId === 'local' ? '本机（中台）' : nodeId)}<small>{locations.filter(l => l.node_id === nodeId).length} 个位置</small></h3><div className="dataset-location-list">{locations.filter(l => l.node_id === nodeId).map(l => <article key={l.id}>
      <div><strong>{l.name}{l.is_default ? ' · 默认' : ''}{!l.enabled ? ' · 已禁用' : ''}</strong>
        <code>{l.resolved_path || l.path}</code><small>{l.status === 'ready' ? '上次校验通过' : l.status === 'error' ? '校验失败' : '等待检查'}{l.details.free_bytes != null ? ` · 可用 ${formatBytes(l.details.free_bytes)} / ${formatBytes(l.details.total_bytes || 0)}` : ''}{l.details.checked_at ? ` · ${new Date(l.details.checked_at).toLocaleString()}` : ''}</small>
        {l.details.message && <small role="alert">{l.details.message}</small>}</div>
      <div className="dataset-inline-actions"><button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.checkDatasetLocation(l.id))}>检查</button>{l.status !== 'ready' && <button type="button" className="button compact ghost" disabled={busy} onClick={() => void action(() => api.checkDatasetLocation(l.id, true))}>创建并检查</button>}
        <button type="button" className="button compact ghost" disabled={busy || l.status !== 'ready' || l.is_default} onClick={() => void action(() => api.updateDatasetLocation(l.id, true, true))}>设为默认</button>
        <details className="dataset-location-more"><summary aria-label={`更多操作：${l.name}`} title="更多操作"><MoreHorizontal size={18} /></summary><div><button type="button" className="button compact ghost" disabled={busy} onClick={e => { e.currentTarget.closest('details')?.removeAttribute('open'); void action(() => api.updateDatasetLocation(l.id, !l.enabled, false)) }}>{l.enabled ? '禁用' : '启用'}</button>
        <button type="button" className="button compact danger" disabled={busy || l.is_default} onClick={e => { e.currentTarget.closest('details')?.removeAttribute('open'); void action(() => api.forgetDatasetLocation(l.id)) }}>移除配置</button></div></details></div>
    </article>)}</div></section>)}</div>
  </section>
}

export function DatasetStorageModal({ onClose }: { onClose: () => void }) {
  const ref = useModalFocus<HTMLDivElement>(true, onClose)
  useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = previous }
  }, [])
  return <div className="modal-backdrop" onMouseDown={onClose}><div ref={ref} className="modal dataset-storage-modal" role="dialog" aria-modal="true" aria-labelledby="dataset-storage-title" tabIndex={-1} onMouseDown={e => e.stopPropagation()}>
    <header><h2 id="dataset-storage-title">保存位置配置</h2><button type="button" className="icon-button" aria-label="关闭保存位置" onClick={onClose}><X size={20} /></button></header>
    <div className="dataset-storage-modal-body"><DatasetStorageSettings /></div>
    <footer><button type="button" className="button ghost" onClick={onClose}>关闭</button></footer>
  </div></div>
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
