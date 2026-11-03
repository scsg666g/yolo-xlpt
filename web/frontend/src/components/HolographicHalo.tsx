import { useEffect, useRef } from 'react'

export function HolographicHalo({ paused }: { paused: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const pausedRef = useRef(paused)
  useEffect(() => { pausedRef.current = paused }, [paused])
  useEffect(() => {
    const element = canvas.current
    const context = element?.getContext('2d')
    if (!element || !context) return
    const media = matchMedia('(prefers-reduced-motion: reduce)')
    let frame = 0, previous = 0, time = 0, width = 0, height = 0, drawn = false
    const parent = element.parentElement!
    const visual = parent.querySelector<HTMLElement>('.auth-visual-region')
    let pointerX = 0, pointerY = 0, haloX = 0, haloY = 0, radius = 0
    const stars = Array.from({ length: 80 }, (_, i) => ({
      x: ((i * 719 + 71) % 997) / 997, y: ((i * 367 + 89) % 991) / 991,
      r: i % 5 === 0 ? 1.3 : .65, phase: i * .73,
    }))
    function resize() {
      const rect = element!.getBoundingClientRect()
      width = rect.width; height = rect.height
      // Size and align the halo within its column, including when form height changes.
      const area = visual?.getBoundingClientRect()
      if (area && area.width > 0 && area.height > 0) {
        haloX = area.left - rect.left + area.width / 2
        haloY = area.top - rect.top + area.height * .41
        radius = Math.min(area.width * .58, area.height * .36)
      } else {
        haloX = width * .92; haloY = height * .24
        radius = Math.min(width * .47, 245)
      }
      const dpr = Math.min(devicePixelRatio || 1, 1.5)
      element!.width = Math.round(width * dpr); element!.height = Math.round(height * dpr)
      context!.setTransform(dpr, 0, 0, dpr, 0, 0); drawn = false
    }
    function draw(now: number) {
      frame = requestAnimationFrame(draw)
      if (document.hidden || now - previous < 33) return
      const delta = Math.min((now - previous) / 1000, .05); previous = now
      const moving = !pausedRef.current && !media.matches
      if (!moving && drawn) return
      if (moving) time += delta
      const ctx = context!
      ctx.clearRect(0, 0, width, height)
      for (const star of stars) {
        ctx.globalAlpha = .2 + .22 * (.5 + .5 * Math.sin(time * .6 + star.phase))
        ctx.fillStyle = '#d4ddff'; ctx.beginPath()
        ctx.arc(star.x * width, star.y * height, star.r, 0, Math.PI * 2); ctx.fill()
      }
      ctx.globalAlpha = 1
      ctx.save()
      ctx.translate(haloX + pointerX * 10, haloY + pointerY * 10)
      ctx.rotate(-.32 + Math.sin(time * .15) * .05)
      const glow = ctx.createRadialGradient(0, 0, radius * .5, 0, 0, radius * 1.25)
      glow.addColorStop(0, '#111a3700'); glow.addColorStop(.45, '#456ed61a')
      glow.addColorStop(.7, '#6d6cfa12'); glow.addColorStop(1, '#223a7c00')
      ctx.fillStyle = glow; ctx.fillRect(-radius * 1.4, -radius * 1.4, radius * 2.8, radius * 2.8)
      for (let j = 0; j < 30; j++) {
        const r = radius * (.69 + j * .0105)
        ctx.lineWidth = j % 6 === 0 ? 1.3 : .5
        ctx.strokeStyle = `rgba(${110 + j * 3},${169 - j},255,${j % 6 === 0 ? .43 : .16})`
        ctx.beginPath(); ctx.ellipse(0, 0, r * .74, r, 0, 0, Math.PI * 2); ctx.stroke()
      }
      for (let k = 0; k < 5; k++) {
        ctx.beginPath(); ctx.lineWidth = k % 2 ? 2 : 3
        ctx.shadowColor = k % 2 ? '#80ccff' : '#b7a0ff'; ctx.shadowBlur = 12
        ctx.strokeStyle = k % 2 ? '#9be0ff' : '#ada8ff'
        const a = time * .24 + k * Math.PI * .41
        ctx.ellipse(0, 0, radius * (.74 + k * .002), radius * (1 + k * .003), 0, a, a + .45); ctx.stroke()
      }
      ctx.shadowBlur = 0
      for (let j = 0; j < 60; j++) {
        const a = j * Math.PI * 2 / 60
        ctx.strokeStyle = j % 5 === 0 ? '#a9caff77' : '#a9caff22'; ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(Math.cos(a) * radius * .84, Math.sin(a) * radius * 1.13)
        ctx.lineTo(Math.cos(a) * radius * .85, Math.sin(a) * radius * 1.15); ctx.stroke()
      }
      ctx.restore(); drawn = true
    }
    function pointer(event: PointerEvent) {
      const rect = parent.getBoundingClientRect()
      pointerX = (event.clientX - rect.left) / rect.width - .5
      pointerY = (event.clientY - rect.top) / rect.height - .5
    }
    function leave() { pointerX = 0; pointerY = 0 }
    function motion() { drawn = false }
    const observer = new ResizeObserver(resize); observer.observe(element)
    if (visual) observer.observe(visual)
    // A form error or a late font can shift the centered column without resizing it.
    const content = parent.querySelector('.auth-content')
    if (content) observer.observe(content)
    const form = parent.querySelector('.auth-form-region')
    if (form) observer.observe(form)
    parent.addEventListener('pointermove', pointer); parent.addEventListener('pointerleave', leave)
    media.addEventListener('change', motion)
    resize(); frame = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(frame); observer.disconnect()
      parent.removeEventListener('pointermove', pointer); parent.removeEventListener('pointerleave', leave)
      media.removeEventListener('change', motion)
    }
  }, [])
  return <canvas ref={canvas} className="auth-halo" aria-hidden="true" />
}
