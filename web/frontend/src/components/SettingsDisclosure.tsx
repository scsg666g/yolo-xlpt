import type { ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

type SettingsDisclosureProps = {
  title: string
  description?: string
  badge?: string
  defaultOpen?: boolean
  children: ReactNode
}

export function SettingsDisclosure({ title, description, badge, defaultOpen = false, children }: SettingsDisclosureProps) {
  return <details className="settings-disclosure" open={defaultOpen}>
    <summary>
      <span className="settings-disclosure-copy"><strong>{title}</strong>{description && <small>{description}</small>}</span>
      {badge && <span className="settings-disclosure-badge">{badge}</span>}
      <ChevronDown className="settings-disclosure-chevron" size={16} aria-hidden="true" />
    </summary>
    <div className="settings-disclosure-body">{children}</div>
  </details>
}
