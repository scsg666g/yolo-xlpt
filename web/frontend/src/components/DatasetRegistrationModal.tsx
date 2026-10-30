import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, RefreshCw, X } from 'lucide-react'
import { api } from '../api'
import type { ManagedDataset } from '../types'
import { useModalFocus } from '../useModalFocus'

export function DatasetRegistrationModal({ dataset, mode, onClose, onDone }: {
  dataset: ManagedDataset
  mode: 'unregister' | 'adopt'
  onClose: () => void
  onDone: () => void
}) {
  const [path, setPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const submitting = useRef(false)
  const adopting = mode === 'adopt'
  const ref = useModalFocus<HTMLFormElement>(true, onClose, {
    canClose: !busy, initialFocusSelector: adopting ? 'input' : '[data-cancel]',
  })
  useEffect(() => {
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = previous }
  }, [])

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    if (submitting.current || (adopting && path !== dataset.directory)) return
    submitting.current = true
    setBusy(true); setError('')
    try {
      if (adopting) await api.adoptDataset(dataset.id, path)
      else await api.unregisterDataset(dataset.id)
      onDone()
    } catch (reason) {
      const message = reason && typeof reason === 'object' && 'message' in reason && typeof reason.message === 'string'
        ? reason.message : typeof reason === 'string' ? reason : ''
      setError(message || (adopting ? '接管失败' : '删除登记失败'))
    } finally {
      submitting.current = false
      setBusy(false)
    }
  }

  return <div className="modal-backdrop" onMouseDown={() => { if (!busy) onClose() }}>
    <form ref={ref} className="modal dataset-action-modal" role={adopting ? 'dialog' : 'alertdialog'} aria-modal="true" aria-labelledby="dataset-registration-title" aria-describedby="dataset-registration-description" aria-busy={busy} tabIndex={-1} onSubmit={event => void submit(event)} onMouseDown={event => event.stopPropagation()}>
      <header><div><span className="eyebrow">{adopting ? 'MANAGE FILES' : 'REMOVE REGISTRATION'}</span><h2 id="dataset-registration-title">{adopting ? '接管文件' : '删除数据集登记'}</h2><p>{dataset.name}</p></div><button type="button" className="icon-button" disabled={busy} onClick={onClose} aria-label="关闭登记操作"><X size={19} /></button></header>
      <div className="modal-body form-grid dataset-action-body">
        {error && <div className="dataset-feedback error" role="alert"><AlertTriangle size={17} /><span>{error}</span></div>}
        <section className="dataset-action-path"><strong>服务器 · {dataset.node_name || (dataset.node_id === 'local' ? '本机服务器' : dataset.node_id)}</strong><small>{dataset.node_id}</small><code>{dataset.directory}</code></section>
        <aside className="dataset-action-help" id="dataset-registration-description">
          <strong>{adopting ? '授予平台文件管理权限' : '只删除数据库中的登记记录，保留原始文件'}</strong>
          {adopting ? <p>接管后平台可执行迁移源清理和永久删除文件，仍会检查任务引用及文件状态。此次操作仅更新管理权限，可在训练期间执行。</p>
            : <><p>图片、标签、data.yaml、训练任务和历史实验均保留，正在进行的训练继续使用原目录。</p><p>删除后卡片立即消失，刷新和重启不会自动恢复。以后可通过“扫描已有数据集”或“登记已有目录”重新登记。</p></>}
        </aside>
        {!!dataset.references.length && <p className="dataset-action-help">当前有 {dataset.references.length} 个任务引用此数据集；{adopting ? '接管不会停止训练，也不会允许绕过引用保护删除文件。' : '删除登记不会解除任务与实际数据路径的关联。'}</p>}
        {adopting && <label>输入完整目录以确认接管<input aria-label="确认接管的数据集完整目录" value={path} onChange={event => setPath(event.target.value)} disabled={busy} required autoComplete="off" spellCheck={false} /><small>输入内容需与上方目录完全一致。</small></label>}
      </div>
      <footer><button type="button" className="button ghost" data-cancel disabled={busy} onClick={onClose}>取消</button><button type="submit" className={`button ${adopting ? 'primary' : 'danger'}`} disabled={busy || (adopting && path !== dataset.directory)}>{busy && <RefreshCw size={15} className="spin" />}{busy ? '正在提交…' : adopting ? '确认接管' : '确认删除登记'}</button></footer>
    </form>
  </div>
}
