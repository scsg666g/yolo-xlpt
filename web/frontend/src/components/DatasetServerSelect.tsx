import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { Check, ChevronDown, FolderOpen, Search, Server } from 'lucide-react'

export interface DatasetServerOption { id: string; name: string; count?: number; status?: string; description?: string; badge?: string }

/** Keep the popup in the dialog DOM so keyboard focus remains inside its focus trap. */
export function DatasetServerSelect({ value, onChange, options, label = '按服务器筛选', disabled = false, icon = 'server', placeholder = '选择服务器', searchLabel = '搜索服务器', emptyMessage = '没有匹配的服务器' }: {
  value: string; onChange: (value: string) => void; options: DatasetServerOption[]; label?: string; disabled?: boolean
  icon?: 'server' | 'folder'; placeholder?: string; searchLabel?: string; emptyMessage?: string
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [highlight, setHighlight] = useState(0)
  const [position, setPosition] = useState({ left: 0, top: 0, width: 260, maxHeight: 320 })
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const popup = useRef<HTMLDivElement>(null)
  const id = useId()
  const filtered = options.filter(o => `${o.name} ${o.id} ${o.description || ''}`.toLowerCase().includes(query.trim().toLowerCase()))
  const index = Math.min(highlight, Math.max(filtered.length - 1, 0))
  const selected = options.find(o => o.id === value)
  function close(restore = false) { setOpen(false); if (restore) trigger.current?.focus() }
  function show() { setQuery(''); setHighlight(Math.max(0, options.findIndex(o => o.id === value))); setOpen(true) }
  function choose(identity: string) { onChange(identity); close(true) }
  useLayoutEffect(() => {
    if (!open) return
    function place() {
      const rect = trigger.current?.getBoundingClientRect()
      if (!rect) return
      const width = Math.min(Math.max(260, rect.width), window.innerWidth - 24)
      const below = window.innerHeight - rect.bottom - 12
      const above = rect.top - 12
      const up = below < 230 && above > below
      const maxHeight = Math.min(360, Math.max(120, up ? above - 6 : below - 6))
      setPosition({ left: Math.min(Math.max(12, rect.left), window.innerWidth - width - 12), top: up ? rect.top - 6 : rect.bottom + 6, width, maxHeight })
      if (popup.current) popup.current.style.transform = up ? 'translateY(-100%)' : ''
    }
    function escape(event: KeyboardEvent) {
      if (event.key === 'Escape' && !event.isComposing) {
        event.preventDefault(); event.stopPropagation(); setOpen(false); trigger.current?.focus()
      }
    }
    const menu = root.current
    menu?.addEventListener('keydown', escape)
    place()
    window.addEventListener('resize', place)
    // Scrolling the options must not close the popup; scrolling the page repositions it.
    window.addEventListener('scroll', place, true)
    ;(popup.current?.querySelector<HTMLElement>('input, [role="listbox"]') || popup.current)?.focus()
    function outside(e: PointerEvent) { if (!root.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('pointerdown', outside)
    return () => { menu?.removeEventListener('keydown', escape); window.removeEventListener('resize', place); window.removeEventListener('scroll', place, true); document.removeEventListener('pointerdown', outside) }
  }, [open])
  useEffect(() => {
    if (open) document.getElementById(`${id}-${index}`)?.scrollIntoView({ block: 'nearest' })
  }, [open, index, id])
  function keyboard(e: React.KeyboardEvent) {
    if (e.nativeEvent.isComposing) return
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); e.nativeEvent.stopImmediatePropagation(); close(true) }
    else if (e.key === 'Tab') close()
    else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) {
      e.preventDefault()
      setHighlight(e.key === 'Home' ? 0 : e.key === 'End' ? filtered.length - 1 : (index + (e.key === 'ArrowDown' ? 1 : -1) + filtered.length) % Math.max(1, filtered.length))
    } else if (e.key === 'Enter' || (e.key === ' ' && e.target === popup.current)) {
      e.preventDefault(); if (filtered[index]) choose(filtered[index].id)
    }
  }
  return <div ref={root} className="dataset-server-select" onKeyDown={e => { if (open) keyboard(e) }} onBlur={e => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setOpen(false) }}>
    <button ref={trigger} type="button" className={`dataset-server-trigger ${open ? 'open' : ''}`} disabled={disabled} title={selected ? [selected.name, selected.description].filter(Boolean).join(' · ') : placeholder} aria-label={label} aria-haspopup="listbox" aria-expanded={open} aria-controls={open ? id : undefined} onClick={() => open ? close() : show()} onKeyDown={e => { if (!open && ['ArrowDown', 'ArrowUp'].includes(e.key)) { e.preventDefault(); show() } }}>
      {icon === 'folder' ? <FolderOpen size={16} /> : <Server size={15} />}<span>{selected?.name || placeholder}</span>{selected?.badge && <small className="dataset-select-badge">{selected.badge}</small>}<ChevronDown size={14} />
    </button>
    {open && <div ref={popup} className="dataset-server-popup" style={position} tabIndex={-1}>
      {options.length > 6 && <label className="dataset-server-search"><Search size={14} /><input role="combobox" aria-expanded="true" aria-controls={id} aria-activedescendant={filtered[index] ? `${id}-${index}` : undefined} aria-autocomplete="list" aria-label={searchLabel} value={query} placeholder={searchLabel} onChange={e => { setQuery(e.target.value); setHighlight(0) }} /></label>}
      <div id={id} tabIndex={-1} role="listbox" aria-label={label} aria-activedescendant={filtered[index] ? `${id}-${index}` : undefined}>
        {filtered.map((o, i) => <button type="button" tabIndex={-1} id={`${id}-${i}`} role="option" aria-selected={value === o.id} className={index === i ? 'highlight' : ''} key={o.id} title={o.name} onMouseEnter={() => setHighlight(i)} onClick={() => choose(o.id)}>
          <span><strong>{o.name}</strong>{o.description && <small className="dataset-select-description">{o.description}</small>}{o.status && <small>{o.status}</small>}</span>{o.badge && <small className="dataset-select-badge">{o.badge}</small>}{o.count != null && <small>{o.count}</small>}{value === o.id && <Check size={15} />}
        </button>)}
        {!filtered.length && <p>{emptyMessage}</p>}
      </div>
    </div>}
  </div>
}
