import { useEffect, useRef, useState } from 'react'
import { Copy, Download, LoaderCircle, RefreshCw, X } from 'lucide-react'
import { api } from '../api'
import type { ComputeNode, ComputeNodeSettings } from '../types'

interface Props {
  node?: ComputeNode
  connect?: boolean
  onClose: () => void
  onSaved: () => void
}

export function ComputeNodeModal({ node, connect = false, onClose, onSaved }: Props) {
  const [settings, setSettings] = useState<ComputeNodeSettings>({
    name: node?.name || '', address: node?.address || '', notes: node?.notes || '',
    enabled: node?.enabled ?? true, report_interval_seconds: node?.report_interval_seconds || 5,
  })
  const [savedNode, setSavedNode] = useState(node)
  const [setup, setSetup] = useState(connect)
  const [token, setToken] = useState('')
  const [downloaded, setDownloaded] = useState(false)
  const [center, setCenter] = useState(window.location.origin)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const dialog = useRef<HTMLDivElement>(null)
  const submitting = useRef(false)

  useEffect(() => {
    const previous = document.activeElement
    dialog.current?.querySelector<HTMLElement>('input, button')?.focus()
    return () => { if (previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [])

  function close() {
    if (submitting.current) return
    if (token && !downloaded && !window.confirm('Agent Token 只在本窗口显示。还没有下载配置，确认关闭吗？关闭后需要重新生成 Token。')) return
    onClose()
  }

  function handleKeys(event: React.KeyboardEvent<HTMLDivElement>) {
    if (event.nativeEvent.isComposing) return
    if (event.key === 'Escape') { event.preventDefault(); close() }
    if (event.key === 'Tab') {
      const controls = event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href]')
      const first = controls[0], last = controls[controls.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
    }
  }

  async function save(event: React.FormEvent) {
    event.preventDefault()
    if (submitting.current) return
    submitting.current = true; setBusy(true); setError('')
    try {
      if (node) {
        await api.updateComputeNode(node.id, settings)
        onSaved(); onClose()
      } else {
        const result = await api.createComputeNode(settings)
        setSavedNode(result.node); setToken(result.token); setSetup(true); onSaved()
      }
    } catch (reason) { setError(reason instanceof Error ? reason.message : '服务器保存失败') }
    finally { submitting.current = false; setBusy(false) }
  }

  async function rotate() {
    if (!savedNode || submitting.current) return
    if (!window.confirm('重新生成后旧 Token 会立即失效，需要替换远程 Agent 配置并重启 Agent。是否继续？')) return
    submitting.current = true; setBusy(true); setError('')
    try {
      const result = await api.rotateComputeToken(savedNode.id)
      setSavedNode(result.node); setToken(result.token); setDownloaded(false); setNotice('新 Token 已生成，请下载配置。'); onSaved()
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Token 生成失败') }
    finally { submitting.current = false; setBusy(false) }
  }

  function configText() {
    const url = new URL(center)
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('请输入有效的中心平台 HTTP(S) 地址')
    if (!token || !savedNode) throw new Error('请先生成该服务器的 Agent Token')
    return JSON.stringify({
      _comments: {
        format: '本文件是标准 JSON；中文说明保存在 _comments 中，不要添加 // 或 # 注释。相对路径配置需要 Agent 1.8.1 或更高版本；旧版 Agent 请先升级或使用绝对路径。修改配置后需重启 Agent。',
        center_url: '远程服务器能够访问的中心平台地址。开发版和使用版应连接各自的平台。',
        node_id: '中心平台生成的节点 ID，应与 token 配套使用。',
        token: '节点凭证，请保密；不要提交到 Git 或向他人分享。',
        report_interval_seconds: '算力采样和状态上报间隔，单位为秒。',
        nvidia_smi_bin: 'GPU 信息采集程序；默认从运行 Agent 的 PATH 查找 nvidia-smi。',
        enable_training: '默认 false，仅上报算力。配置好训练环境和允许目录后再改为 true。',
        training_cwd: '训练工作目录；相对路径以此配置文件所在目录为基准。默认 . 表示配置文件所在目录。',
        allowed_training_roots: '允许访问模型、数据集及输出的目录列表；相对路径以 training_cwd 为基准。启用训练前必须填写。',
        training_output_roots: '训练结果允许写入的目录列表，必须包含在 allowed_training_roots 中；相对路径以 training_cwd 为基准。开发版与使用版应使用互不重叠的输出目录。空列表会兼容使用 allowed_training_roots，建议明确配置。',
        yolo_bin: 'YOLO 命令。仅填 yolo 时从 PATH 查找；包含路径分隔符的相对路径以 training_cwd 为基准。',
        training_lock_file: '训练互斥锁文件；相对路径以 training_cwd 为基准。同一训练工作区的多个 Agent 应共用锁文件，不能通过改锁名绕过占用保护。',
        enable_training_trace: '是否采样真实 Batch 的训练过程；仅受支持的新训练任务生成 Trace，会增加少量采样开销。',
      },
      center_url: center.trim().replace(/\/+$/, ''), node_id: savedNode.id, token,
      report_interval_seconds: savedNode.report_interval_seconds, nvidia_smi_bin: 'nvidia-smi',
      enable_training: false, training_cwd: '.', allowed_training_roots: [], training_output_roots: [],
      yolo_bin: 'yolo', training_lock_file: '.yolo-forge-training.lock', enable_training_trace: true,
    }, null, 2)
  }

  function downloadConfig() {
    try {
      const url = URL.createObjectURL(new Blob([configText()], { type: 'application/json' }))
      const link = document.createElement('a'); link.href = url; link.download = 'agent-config.json'; link.click()
      window.setTimeout(() => URL.revokeObjectURL(url), 1000)
      setDownloaded(true); setError(''); setNotice('配置已下载，请妥善保存。')
    } catch (reason) { setError(reason instanceof Error ? reason.message : '配置下载失败') }
  }

  async function copyConfig() {
    try { await navigator.clipboard.writeText(configText()); setDownloaded(true); setError(''); setNotice('配置已复制。') }
    catch (reason) { setError(reason instanceof Error ? reason.message : '复制失败，请使用下载配置') }
  }

  const update = <K extends keyof ComputeNodeSettings>(key: K, value: ComputeNodeSettings[K]) => setSettings((previous) => ({ ...previous, [key]: value }))

  return <div className="modal-backdrop" onMouseDown={close}>
    <div ref={dialog} className="modal compute-node-modal" role="dialog" aria-modal="true" aria-labelledby="compute-node-title" onKeyDown={handleKeys} onMouseDown={(event) => event.stopPropagation()}>
      <header><h2 id="compute-node-title">{setup ? `接入服务器 · ${savedNode?.name || ''}` : node ? '编辑服务器' : '添加服务器'}</h2><button className="icon-button" disabled={busy} aria-label="关闭服务器窗口" onClick={close}><X size={18} /></button></header>
      {setup ? <>
        <div className="modal-body">
          <p className="node-help">在对应服务器运行 Agent，即可显示算力数据；训练执行默认关闭，启用前需配置该机的环境路径和允许访问的目录。无需提供 SSH 密码。</p>
          <div className="form-grid"><label className="span-2">中心平台地址<input value={center} onChange={(event) => setCenter(event.target.value)} placeholder="http://中心服务器:18088" /></label></div>
          <p className="node-help">请填写远程服务器能访问的地址。当前浏览器地址仅作为初始值。</p>
          {token ? <div className="node-info">专属 Token 已生成，仅在本窗口提供。配置中包含 Token，请勿提交到 Git 或分享给他人。</div>
            : <div className="node-info">已有 Token 不会再次显示。保留原 Agent 配置即可继续使用；丢失配置时，可重新生成。</div>}
          <div className="agent-downloads"><a className="button" href="/api/compute/agent/download" download><Download size={15} />下载 Agent</a><button className="button primary" disabled={!token} onClick={downloadConfig}><Download size={15} />下载专属配置</button><button className="button" disabled={!token} onClick={() => void copyConfig()}><Copy size={15} />复制配置 JSON</button></div>
          <ol className="agent-steps">
            <li>把 <code>monitor_agent.py</code> 和 <code>agent-config.json</code> 放到远程服务器的同一目录。</li>
            <li>验证权限和连通性，然后上报一次：<pre>chmod 600 agent-config.json{'\n'}python3 monitor_agent.py --config agent-config.json --once</pre></li>
            <li>看到“采样上报成功”后，持续运行：<pre>python3 monitor_agent.py --config agent-config.json</pre></li>
            <li>下载的 JSON 包含中文 <code>_comments</code> 说明。如需执行训练，填写 <code>training_cwd</code>、<code>allowed_training_roots</code> 和 <code>training_output_roots</code> 后，再将 <code>enable_training</code> 改为 <code>true</code>。输出允许目录必须位于访问允许目录内，开发版与使用版应使用互不重叠的输出目录。</li>
            <li>相对路径配置需要 Agent 1.8.1 或更高版本。<code>training_cwd</code> 的相对路径以配置文件目录为基准；允许目录、锁文件及带路径的 <code>yolo_bin</code> 的相对路径以训练工作目录为基准。仅填 <code>yolo</code> 时使用 PATH 中的命令。修改后重启 Agent，在算力页核对上报的实际输出目录；创建任务时可直接选择该目录。</li>
          </ol>
          <p className="node-help">Agent 使用 Python 标准库；GPU 指标由本机 nvidia-smi 提供。训练会调用配置的 YOLO CLI，不执行任意 Shell。日志与产物留在远程服务器；当前平台接收进度、指标和最近日志，不会自动复制模型文件。</p>
          <button className="button compact" disabled={busy} onClick={() => void rotate()}><RefreshCw size={14} />重新生成 Token</button>
          {notice && <div className="node-info" role="status">{notice}</div>}
          {error && <div className="alert error" role="alert">{error}</div>}
        </div>
        <footer><button className="button primary" disabled={busy} onClick={close}>完成</button></footer>
      </> : <form onSubmit={save}>
        <div className="modal-body">
          <div className="form-grid">
            <label className="span-2">服务器名称<input autoFocus required maxLength={160} value={settings.name} disabled={busy} onChange={(event) => update('name', event.target.value)} placeholder="例如：训练服务器-02" /></label>
            <label className="span-2">地址 / IP（可选）<input maxLength={255} value={settings.address} disabled={busy} onChange={(event) => update('address', event.target.value)} placeholder="例如：10.180.0.123，仅用于识别服务器" /></label>
            <label className="span-2">备注（可选）<input maxLength={1000} value={settings.notes} disabled={busy} onChange={(event) => update('notes', event.target.value)} placeholder="位置、用途或负责人" /></label>
            {node?.kind !== 'local' && <><label>Agent 上报间隔<select value={settings.report_interval_seconds} disabled={busy} onChange={(event) => update('report_interval_seconds', Number(event.target.value))}>{[5, 10, 15, 30, 60, 120, 300].map((value) => <option key={value} value={value}>{value} 秒</option>)}</select></label><label className="check"><input type="checkbox" checked={settings.enabled} disabled={busy} onChange={(event) => update('enabled', event.target.checked)} />启用监控</label></>}
          </div>
          <p className="node-help">{node?.kind === 'local' ? '本机算力自动采集，可以修改显示名称、地址和备注。' : '保存后提供专属 Agent 和配置下载。服务器资料保存在中心数据库，刷新或换浏览器仍会保留。'}</p>
          {error && <div className="alert error" role="alert">{error}</div>}
        </div>
        <footer><button type="button" className="button" disabled={busy} onClick={close}>取消</button><button type="submit" className="button primary" disabled={busy || !settings.name.trim()}>{busy && <LoaderCircle size={15} className="spin" />}{node ? '保存' : '保存并获取 Agent'}</button></footer>
      </form>}
    </div>
  </div>
}
