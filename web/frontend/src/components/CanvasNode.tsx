import type { ReactNode } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'

export type PanelNodeData = { title: string; kicker: string; icon: ReactNode; accent?: string; content: ReactNode }

export function CanvasNode({ data, selected }: NodeProps) {
  const panel = data as PanelNodeData
  return <div className={`canvas-node ${selected ? 'selected' : ''}`} style={{ '--node-accent': panel.accent || '#70f0bd' } as React.CSSProperties}>
    <Handle type="target" position={Position.Left} />
    <header><div className="node-icon">{panel.icon}</div><div><span>{panel.kicker}</span><strong>{panel.title}</strong></div></header>
    <div className="node-content">{panel.content}</div>
    <Handle type="source" position={Position.Right} />
  </div>
}

