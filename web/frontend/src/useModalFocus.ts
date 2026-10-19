import { useEffect, useRef, type RefObject } from 'react'

const focusableSelector = [
  'a[href]',
  'button:not(:disabled)',
  'input:not(:disabled):not([type="hidden"])',
  'select:not(:disabled)',
  'textarea:not(:disabled)',
  '[tabindex]:not([tabindex="-1"])',
].join(',')

interface ModalFocusOptions {
  canClose?: boolean
  initialFocusSelector?: string
}

export function useModalFocus<T extends HTMLElement>(
  open: boolean,
  onClose: () => void,
  { canClose = true, initialFocusSelector }: ModalFocusOptions = {},
): RefObject<T | null> {
  const dialogRef = useRef<T>(null)
  const closeRef = useRef(onClose)
  const canCloseRef = useRef(canClose)

  closeRef.current = onClose
  canCloseRef.current = canClose

  useEffect(() => {
    if (!open) return
    const dialog = dialogRef.current
    if (!dialog) return
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const focusables = () => Array.from(dialog.querySelectorAll<HTMLElement>(focusableSelector))
      .filter((element) => element.getClientRects().length > 0 && element.getAttribute('aria-hidden') !== 'true')
    const frame = window.requestAnimationFrame(() => {
      const initial = initialFocusSelector ? dialog.querySelector<HTMLElement>(initialFocusSelector) : null
      ;(initial || focusables()[0] || dialog).focus()
    })

    function handleKeyDown(event: KeyboardEvent) {
      if (event.isComposing) return
      if (event.key === 'Escape') {
        event.stopPropagation()
        if (!canCloseRef.current) return
        event.preventDefault()
        closeRef.current()
        return
      }
      if (event.key !== 'Tab') return
      event.stopPropagation()
      const controls = focusables()
      const first = controls[0]
      const last = controls.at(-1)
      if (!first || !last) {
        event.preventDefault()
        dialog?.focus()
      } else if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog)) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    dialog.addEventListener('keydown', handleKeyDown)
    return () => {
      window.cancelAnimationFrame(frame)
      dialog.removeEventListener('keydown', handleKeyDown)
      if (previousFocus?.isConnected) previousFocus.focus()
    }
  }, [initialFocusSelector, open])

  return dialogRef
}
