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
}

export function CanvasNode({ data, selected }: NodeProps) {
  const panel = data as PanelNodeData
  const classes = ['canvas-node', selected ? 'selected' : '', panel.activity ? `activity-${panel.activity}` : '', panel.pulse ? `node-pulse-${panel.pulse}` : ''].filter(Boolean).join(' ')
  return <div className={classes} style={{ '--node-accent': panel.accent || '#70f0bd' } as React.CSSProperties}>
    <Handle type="target" position={Position.Left} />
    <header><div className="node-icon">{panel.icon}</div><div><span>{panel.kicker}</span><strong>{panel.title}</strong></div></header>
    <div className="node-content">{panel.content}</div>
    <Handle type="source" position={Position.Right} />
  </div>
}
