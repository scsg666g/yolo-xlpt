import { useEffect, useMemo, useState } from 'react'
import { Check, ChevronDown, Folder, HardDrive, LoaderCircle, Search, Server, Trash2 } from 'lucide-react'
import { formatBytes, statusLabel } from '../api'
import type { ComputeNode, ModelAsset, TaskGroup, TrainingTask } from '../types'

const collapsedTaskGroupsKey = 'yolo-model-library-collapsed-groups-v1'
const collapsedAssetGroupsKey = 'yolo-model-library-collapsed-assets-v1'

function readCollapsed(key: string): Set<string> {
  try {
    const value = JSON.parse(window.localStorage.getItem(key) || '[]')
    return new Set(Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [])
  } catch { return new Set() }
}

function dateLabel(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '更新时间未知' : `更新于 ${date.toLocaleDateString('zh-CN')}`
}

function nodeLabel(task: TrainingTask, nodes: ComputeNode[]) {
  if (!task.execution_node_id) return '本机'
  const node = nodes.find((item) => item.id === task.execution_node_id)
  return node ? `远程 · ${node.name}` : '远程节点'
}

interface TrainingTaskLibraryProps {
  tasks: TrainingTask[]
  groups: TaskGroup[]
  nodes: ComputeNode[]
  selectedTaskId: string
  disabled?: boolean
  onSelect: (task: TrainingTask) => void
}

export function TrainingTaskLibrary({ tasks, groups, nodes, selectedTaskId, disabled = false, onSelect }: TrainingTaskLibraryProps) {
  const [search, setSearch] = useState('')
  const [collapsed, setCollapsed] = useState(() => readCollapsed(collapsedTaskGroupsKey))
  const term = search.trim().toLocaleLowerCase()

  useEffect(() => {
    try { window.localStorage.setItem(collapsedTaskGroupsKey, JSON.stringify([...collapsed])) } catch { /* Storage is optional. */ }
  }, [collapsed])

  const sections = useMemo(() => {
    const knownGroups = new Set(groups.map((group) => group.id))
    const definitions = [
      ...groups.map((group) => ({ id: group.id, name: group.name, icon: group.name, tasks: tasks.filter((task) => task.group_id === group.id) })),
      { id: '__ungrouped__', name: '未分组', icon: '', tasks: tasks.filter((task) => !task.group_id || !knownGroups.has(task.group_id)) },
    ]
    return definitions.map((section) => {
      const matches = section.tasks.filter((task) => !term || [
        task.name, task.model_path, task.data_path, section.name, nodeLabel(task, nodes),
      ].some((value) => value.toLocaleLowerCase().includes(term)))
        .sort((left, right) => right.updated_at.localeCompare(left.updated_at))
      return { ...section, matches }
    }).filter((section) => section.matches.length)
  }, [groups, nodes, tasks, term])

  function toggle(id: string) {
    setCollapsed((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return <section className="model-library-task-browser" aria-label="按任务分组选择训练模型">
    <label className="model-library-search"><Search size={14} /><input value={search} onChange={(event) => setSearch(event.target.value)} disabled={disabled} placeholder="搜索任务、数据集或模型" /></label>
    {!tasks.length ? <p className="model-library-empty">没有可用的训练任务。</p>
      : !sections.length ? <p className="model-library-empty">没有匹配的训练任务。</p>
        : <div className="model-library-sections">{sections.map((section) => {
          const isCollapsed = !term && collapsed.has(section.id)
          return <section className={`model-library-section${isCollapsed ? ' collapsed' : ''}`} key={section.id}>
            <header>
              <button type="button" aria-expanded={!isCollapsed} onClick={() => toggle(section.id)}>
                <ChevronDown size={14} /><Folder size={15} /><strong>{section.name}</strong><span>{section.matches.length}</span>
              </button>
            </header>
            {!isCollapsed && <div className="model-library-task-list">{section.matches.map((task) => {
              const selected = task.id === selectedTaskId
              return <button type="button" key={task.id} className={`model-library-task${selected ? ' selected' : ''}`} disabled={disabled} aria-pressed={selected} onClick={() => onSelect(task)}>
                <span className={`model-library-task-status ${task.status}`}><i />{statusLabel(task.status)}</span>
                <strong>{task.name}{selected && <Check size={14} />}</strong>
                <span className="model-library-task-context">{nodeLabel(task, nodes)} · {dateLabel(task.updated_at)}</span>
                <small title={task.data_path}>数据集 · {task.data_path.split(/[\\/]/).filter(Boolean).pop() || task.data_path}</small>
                <small title={task.model_path}>模型 · {task.model_path.split(/[\\/]/).filter(Boolean).pop() || task.model_path}</small>
              </button>
            })}</div>}
          </section>
        })}</div>}
  </section>
}

interface RegisteredModelLibraryProps {
  assets: ModelAsset[]
  sourceType: ModelAsset['source_type']
  nodes: ComputeNode[]
  selectedAssetId: string | null
  disabled?: boolean
  loading?: boolean
  removingAssetId?: string | null
  onRefresh?: () => void
  onSelect: (asset: ModelAsset) => void
  onRemove: (asset: ModelAsset) => void
}

export function RegisteredModelLibrary({ assets, sourceType, nodes, selectedAssetId, disabled = false, loading = false, removingAssetId = null, onRefresh, onSelect, onRemove }: RegisteredModelLibraryProps) {
  const [search, setSearch] = useState('')
  const [collapsed, setCollapsed] = useState(() => readCollapsed(collapsedAssetGroupsKey))
  const term = search.trim().toLocaleLowerCase()

  useEffect(() => {
    try { window.localStorage.setItem(collapsedAssetGroupsKey, JSON.stringify([...collapsed])) } catch { /* Storage is optional. */ }
  }, [collapsed])

  const sections = useMemo(() => {
    const scoped = assets.filter((asset) => asset.source_type === sourceType)
    const nodeIds = [...new Set(scoped.map((asset) => asset.execution_node_id || '__local__'))]
    return nodeIds.map((nodeId) => {
      const groupAssets = scoped.filter((asset) => (asset.execution_node_id || '__local__') === nodeId)
      const node = nodes.find((item) => item.id === nodeId)
      const name = sourceType === 'local_path' ? '本机模型' : node?.name || groupAssets[0]?.execution_node_name || '已登记远程节点'
      const matches = groupAssets.filter((asset) => !term || [asset.name, asset.path, asset.execution_node_name || '', name]
        .some((value) => value.toLocaleLowerCase().includes(term)))
        .sort((left, right) => right.updated_at.localeCompare(left.updated_at))
      return { id: nodeId, name, matches }
    }).filter((section) => section.matches.length)
  }, [assets, nodes, sourceType, term])

  function toggle(id: string) {
    setCollapsed((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return <section className="model-library-asset-browser" aria-label="已登记模型库">
    <div className="model-library-asset-toolbar"><label className="model-library-search"><Search size={14} /><input value={search} onChange={(event) => setSearch(event.target.value)} disabled={disabled} placeholder="搜索模型名称或路径" /></label>{onRefresh && <button type="button" className="icon-button" disabled={disabled || loading} title="刷新已登记模型" aria-label="刷新已登记模型" onClick={onRefresh}>{loading ? <span className="model-library-spinner" /> : <HardDrive size={15} />}</button>}</div>
    {!sections.length ? <p className="model-library-empty">{loading ? '正在读取模型…' : '还没有已登记模型。输入路径并完成校验后会显示在这里。'}</p>
      : <div className="model-library-sections">{sections.map((section) => {
        const isCollapsed = !term && collapsed.has(section.id)
        return <section className={`model-library-section${isCollapsed ? ' collapsed' : ''}`} key={section.id}>
          <header><button type="button" aria-expanded={!isCollapsed} onClick={() => toggle(section.id)}><ChevronDown size={14} />{sourceType === 'remote_path' ? <Server size={15} /> : <HardDrive size={15} />}<strong>{section.name}</strong><span>{section.matches.length}</span></button></header>
          {!isCollapsed && <div className="model-library-task-list">{section.matches.map((asset) => {
            const selected = asset.id === selectedAssetId
            const removing = removingAssetId === asset.id
            const blockedReason = asset.removal_blocked_reason === undefined
              ? asset.status === 'preparing' ? '模型正在准备或传输，完成后可移除登记' : ''
              : asset.removal_blocked_reason || ''
            return <div key={asset.id} className={`model-library-asset ${selected ? 'selected' : ''}`}>
              <button type="button" className="model-library-asset-select" disabled={disabled || removing} aria-pressed={selected} onClick={() => onSelect(asset)}>
                <span className={`asset-status ${asset.status}`}><i />{asset.status === 'ready' ? '可用' : asset.status === 'preparing' ? '准备中' : '不可用'}</span>
                <strong>{asset.name}{asset.size !== null && <small>{formatBytes(asset.size)}</small>}</strong>
                <small className="model-library-asset-path" title={asset.path}>{asset.path}</small>
                <small className="model-library-asset-version">版本 {asset.sha256 ? `${asset.sha256.slice(0, 12)}…` : '待校验'} · {dateLabel(asset.updated_at)}</small>
              </button>
              <button type="button" className="icon-button model-library-asset-remove" title={blockedReason || (disabled ? '请等待当前操作完成后移除登记' : '移除模型登记，保留源文件和测试历史')} disabled={disabled || Boolean(removingAssetId) || Boolean(blockedReason)} aria-label={`移除模型登记：${asset.name}`} aria-describedby={blockedReason ? `model-removal-blocked-${asset.id}` : undefined} onClick={() => onRemove(asset)}>{removing ? <LoaderCircle className="spin" size={14} /> : <Trash2 size={14} />}</button>
              {blockedReason && <small className="model-library-asset-blocked" id={`model-removal-blocked-${asset.id}`}>{blockedReason}</small>}
            </div>
          })}</div>}
        </section>
      })}</div>}
    {sourceType === 'remote_path' && nodes.filter((node) => node.kind === 'agent' && node.enabled).length > 1 && <p className="model-library-node-note">远程模型按 Agent 节点分组；相同路径位于不同节点时仍视为不同模型。</p>}
  </section>
}
