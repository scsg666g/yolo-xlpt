import type { ReactNode } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'

export type PanelNodeData = {
  title: string
  kicker: string
  icon: ReactNode
  accent?: string
  content: ReactNode
  activity?: 'queued' | 'starting' | 'running' | 'stopping' | 'stopped' | 'interrupted' | 'failed' | 'completed'
  pulse?: 'a' | 'b'
  onOpen?: () => void
  openLabel?: string
  openText?: string
}

export function CanvasNode({ data, selected }: NodeProps) {
  const panel = data as PanelNodeData
  const classes = ['canvas-node', selected ? 'selected' : '', panel.activity ? `activity-${panel.activity}` : '', panel.pulse ? `node-pulse-${panel.pulse}` : ''].filter(Boolean).join(' ')
  return <div className={classes} style={{ '--node-accent': panel.accent || '#70f0bd' } as React.CSSProperties}>
    <Handle type="target" position={Position.Left} isConnectable={false} />
    <header><div className="node-icon">{panel.icon}</div><div><span>{panel.kicker}</span><strong>{panel.title}</strong></div>{panel.onOpen && <button type="button" className="node-open-button nodrag nopan" aria-label={panel.openLabel || `查看${panel.title}`} onClick={(event) => { event.stopPropagation(); panel.onOpen?.() }} onDoubleClick={(event) => event.stopPropagation()}>{panel.openText || '查看详情'}</button>}</header>
    <div className="node-content">{panel.content}</div>
    <Handle type="source" position={Position.Right} isConnectable={false} />
  </div>
}
