import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { ArrowLeft, ArrowUpRight, FolderOpen, RefreshCw, Search, Trash2, X } from 'lucide-react'
import { api, formatBytes } from '../api'
import type { DatasetSettingsResponse, DatasetTemporaryDetail } from '../types'
import { useModalFocus } from '../useModalFocus'
import '../datasetSettings.css'

type TemporaryRow = DatasetSettingsResponse['storage']['operations'][number]
const operationStatus: Record<string, string> = { pending: '等待处理', running: '正在处理', committing: '提交中', failed: '失败', completed: '已完成', cancelled: '已取消' }

export function DatasetTemporaryModal({ rows, truncated, sampledAt, initialSelected, onClose, onCleared, onOpenOperation }: {
  rows: TemporaryRow[]; truncated: number; sampledAt: string | null
  initialSelected?: string
  onClose: () => void; onCleared: () => void; onOpenOperation?: (id: string) => void
}) {
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string | null>(initialSelected || null)
  const [detail, setDetail] = useState<DatasetTemporaryDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [confirm, setConfirm] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [refresh, setRefresh] = useState(0)
  const dialog = useModalFocus<HTMLDivElement>(true, onClose, { canClose: !busy, initialFocusSelector: '[data-temporary-initial]' })

  useEffect(() => {
    if (!selected) return
    const controller = new AbortController()
    setLoading(true)
    setError('')
    void api.datasetTemporaryDetail(selected, controller.signal).then(value => {
      if (!controller.signal.aborted) setDetail(value)
    }).catch(reason => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '临时文件详情读取失败')
    }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [selected, refresh])

  const visible = rows.filter(row => `${row.name} ${row.id} ${row.kind} ${row.status}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()))
  async function clear() {
    if (!detail?.can_clear || busy) return
    setBusy(true)
    setError('')
    try {
      const result = await api.clearDatasetTemporary(detail.id)
      setNotice(result.message)
      setConfirm(false)
      onCleared()
      setDetail(null)
      setRefresh(value => value + 1)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '临时文件清理失败，请刷新详情确认状态')
      setConfirm(false)
      setRefresh(value => value + 1)
    } finally { setBusy(false) }
  }

  return createPortal(<div className="dataset-temporary-backdrop" onMouseDown={() => { if (!busy) onClose() }}>
    <div ref={dialog} className="dataset-temporary-modal" role="dialog" aria-modal="true" aria-labelledby="dataset-temporary-title" tabIndex={-1} onKeyDown={event => event.stopPropagation()} onMouseDown={event => event.stopPropagation()}>
      <header><div><span className="eyebrow">TEMPORARY FILES</span><h2 id="dataset-temporary-title">{selected ? '临时文件详情' : '平台临时文件'}</h2><p>仅查看平台中心的中转快照与上传暂存，原始数据集和训练文件不在此范围。</p></div><button data-temporary-initial type="button" className="icon-button" aria-label="关闭临时文件详情" disabled={busy} onClick={onClose}><X size={18} /></button></header>
      <div className="dataset-temporary-body">
        {notice && <p className="settings-inline-status" role="status">{notice}</p>}
        {error && <p className="alert error" role="alert">{error}</p>}
        {selected ? <>
          <button type="button" className="button ghost compact" onClick={() => { if (!rows.length) { onClose(); return } setSelected(null); setDetail(null); setConfirm(false); setError('') }}><ArrowLeft size={14} />{rows.length ? '返回列表' : '返回操作记录'}</button>
          {loading && <p role="status">正在核对临时文件…</p>}
          {detail && !loading && <>
            <div className="dataset-temporary-operation"><strong>{detail.operation?.name || (detail.operation ? detail.id : '未关联操作记录')}</strong><span>{detail.operation ? `${operationStatus[detail.operation.status] || detail.operation.status} · ${detail.operation.node_name || detail.operation.node_id} · 操作编号 ${detail.id}` : `临时目录编号 ${detail.id}`}</span><p>{detail.cleanup_reason}</p>{detail.operation && onOpenOperation && <button type="button" className="button ghost compact" onClick={() => onOpenOperation(detail.id)}><ArrowUpRight size={14} />查看关联操作记录</button>}</div>
            <div className="dataset-temporary-file-heading"><strong>文件和目录</strong><span>{detail.sampled_complete ? `${detail.files.length} 项` : `已观察到 ${detail.files.length} 项，统计可能不完整`}</span></div>
            {detail.files.length ? <div className="dataset-temporary-files">{detail.files.map((file, index) => <article key={`${file.kind}-${file.path}-${index}`}><FolderOpen size={15} aria-hidden="true" /><div><strong>{file.label} / {file.path}</strong><small>{file.directory ? '目录' : '文件'} · {file.complete ? '' : '至少 '}{formatBytes(file.bytes)}</small>{file.examples.length > 0 && <details><summary>查看目录内文件示例 · {file.examples.length} 项</summary><ul>{file.examples.map(name => <li key={name}>{name}</li>)}</ul></details>}</div></article>)}</div> : <p className="dataset-settings-note">当前未发现该操作的临时文件，操作记录仍然保留。</p>}
            <p className="dataset-settings-note">{detail.clear_reason}</p>
            {confirm && <div className="dataset-temporary-confirm" role="alertdialog" aria-labelledby="dataset-temporary-confirm-title"><strong id="dataset-temporary-confirm-title">确认清理这些临时文件？</strong><p>只清理当前操作可安全释放的平台中心文件，保留操作历史和数据集文件。失败操作下次重试将重新生成中转快照。</p><div><button type="button" className="button ghost compact" disabled={busy} onClick={() => setConfirm(false)}>返回查看</button><button type="button" className="button danger compact" disabled={busy} onClick={() => void clear()}>{busy ? '正在清理…' : '确认清理'}</button></div></div>}
          </>}
        </> : <>
          <label className="dataset-temporary-search"><Search size={15} aria-hidden="true" /><input type="search" aria-label="搜索临时文件关联操作" placeholder="搜索数据集名称或操作编号" value={query} onChange={event => setQuery(event.target.value)} /></label>
          <p className="dataset-settings-note">最近采样：{sampledAt ? new Date(sampledAt).toLocaleString('zh-CN') : '尚未完成'}。{truncated ? `还有 ${truncated} 项未展示，可在操作记录中按编号查找。` : '列表显示当前采样关联的操作。'}</p>
          {visible.length ? <div className="dataset-temporary-list">{visible.map(row => <button key={row.id} type="button" onClick={() => { setSelected(row.id); setDetail(null); setNotice('') }}><span><strong>{row.name || row.id}</strong><small>{operationStatus[row.status] || row.status} · {row.cleanup_reason}</small><small>操作编号 {row.id}</small></span><span>{row.sampled_complete ? '' : '至少 '}{formatBytes(row.total_bytes)}<ArrowUpRight size={14} /></span></button>)}</div> : <p className="dataset-settings-note">{query ? '没有匹配的临时文件记录。' : '当前采样未发现与操作记录关联的临时文件。'}</p>}
        </>}
      </div>
      <footer><span>清理不会删除操作记录</span>{selected && <button type="button" className="button ghost compact" disabled={loading || busy} onClick={() => setRefresh(value => value + 1)}><RefreshCw size={14} />重新核对</button>}{detail?.can_clear && !confirm && <button type="button" className="button danger compact" disabled={loading || busy} onClick={() => setConfirm(true)}><Trash2 size={14} />清理临时文件</button>}<button type="button" className="button ghost compact" disabled={busy} onClick={onClose}>关闭</button></footer>
    </div>
  </div>, document.body)
}
