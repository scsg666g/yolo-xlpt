import { useEffect, useState, type CSSProperties } from 'react'
import { Check, Monitor, Moon, Palette, Sun } from 'lucide-react'
import { ACCENT_PRESETS, updateAppearance, useAppearance, type ThemeMode } from '../appearance'

const modes = [
  { id: 'light', label: '明亮', description: '清晰、轻盈的工作空间', icon: Sun },
  { id: 'dark', label: '深色', description: '适合夜间与长时间查看', icon: Moon },
  { id: 'system', label: '跟随系统', description: '随设备外观自动切换', icon: Monitor },
] satisfies { id: ThemeMode; label: string; description: string; icon: typeof Sun }[]

export function AppearanceSettings() {
  const appearance = useAppearance()
  const [colorDraft, setColorDraft] = useState(appearance.customColor)
  const [colorError, setColorError] = useState(false)

  useEffect(() => {
    setColorDraft(appearance.customColor)
    setColorError(false)
  }, [appearance.customColor, appearance.accent])

  function applyColor() {
    const value = colorDraft.trim()
    if (!/^#[0-9a-f]{6}$/i.test(value)) {
      setColorError(true)
      return
    }
    setColorError(false)
    updateAppearance({ accent: 'custom', customColor: value })
  }

  return <div className="appearance-settings">
    <fieldset className="appearance-section">
      <legend>显示模式</legend>
      <p>默认版保留当前配色、字号与间距。所有调整即时生效。</p>
      <div className="appearance-modes">
        {modes.map(({ id, label, description, icon: Icon }) => <button
          key={id} type="button" className={`appearance-mode ${appearance.mode === id ? 'selected' : ''}`}
          aria-pressed={appearance.mode === id} onClick={() => updateAppearance({ mode: id })}
        >
          <span className={`appearance-mode-image ${id}`} aria-hidden="true"><i /><span><b /><b /><b /></span></span>
          <span className="appearance-mode-label"><Icon size={15} /><strong>{label}</strong>{appearance.mode === id && <Check size={14} />}</span>
          <small>{description}</small>
        </button>)}
      </div>
    </fieldset>

    <fieldset className="appearance-section">
      <legend>主题配色</legend>
      <p>调整按钮和选中项的颜色；训练状态与图表曲线保持各自的配色。</p>
      <div className="appearance-swatches">
        {ACCENT_PRESETS.map(({ id, label, color }) => <button
          key={id} type="button" className={`appearance-swatch ${appearance.accent === id ? 'selected' : ''}`}
          aria-pressed={appearance.accent === id} onClick={() => updateAppearance({ accent: id })}
          style={{ '--swatch-color': color } as CSSProperties}
        ><i aria-hidden="true" /><span>{label}</span>{appearance.accent === id && <Check size={13} />}</button>)}
        <button type="button" className={`appearance-swatch ${appearance.accent === 'custom' ? 'selected' : ''}`} aria-pressed={appearance.accent === 'custom'} onClick={() => updateAppearance({ accent: 'custom' })}><Palette size={17} />自定义{appearance.accent === 'custom' && <Check size={13} />}</button>
      </div>
      <div className="appearance-custom-color">
        <label className="appearance-color-picker" title="选择自定义主题色">
          <input type="color" aria-label="选择自定义主题色" value={appearance.customColor} onChange={(event) => updateAppearance({ accent: 'custom', customColor: event.target.value })} />
        </label>
        <label className="appearance-hex-label" htmlFor="appearance-color">自定义色值</label>
        <input id="appearance-color" className="appearance-hex" value={colorDraft} maxLength={7} spellCheck={false} aria-invalid={colorError} aria-describedby="appearance-color-help"
          onChange={(event) => { setColorDraft(event.target.value); setColorError(false) }}
          onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); applyColor() } }} />
        <button type="button" className="button ghost compact" onClick={applyColor}>应用颜色</button>
      </div>
      <p id="appearance-color-help" className={colorError ? 'appearance-error' : ''} role={colorError ? 'alert' : undefined}>{colorError ? '请输入完整的六位十六进制颜色，例如 #347BD1。' : '支持取色或输入 HEX 色值，文字颜色会自动适配。'}</p>
    </fieldset>

    <div className="appearance-reading">
      <fieldset className="appearance-section">
        <legend>界面字号</legend>
        <div className="appearance-segmented">
          {([{ id: 'standard', label: '标准（默认）' }, { id: 'large', label: '较大' }] as const).map(({ id, label }) => <button key={id} type="button" aria-pressed={appearance.fontSize === id} onClick={() => updateAppearance({ fontSize: id })}>{label}</button>)}
        </div>
        <p>调整界面文字与日志的阅读大小。</p>
      </fieldset>
      <fieldset className="appearance-section">
        <legend>显示密度</legend>
        <div className="appearance-segmented">
          {([{ id: 'compact', label: '紧凑' }, { id: 'default', label: '默认' }, { id: 'comfortable', label: '舒适' }] as const).map(({ id, label }) => <button key={id} type="button" aria-pressed={appearance.density === id} onClick={() => updateAppearance({ density: id })}>{label}</button>)}
        </div>
        <p>调整任务卡片、列表与表格的间距。</p>
      </fieldset>
    </div>

    <section className="appearance-preview" aria-label="外观效果预览">
      <div className="appearance-preview-heading"><strong>效果预览</strong><span>示例内容</span></div>
      <div className="appearance-preview-card">
        <div><span className="appearance-preview-kicker">TRAINING</span><h4>目标检测训练</h4><p>本机与远程任务使用相同的显示偏好</p></div>
        <div className="appearance-preview-statuses"><span className="status running">运行中</span><span className="status failed">失败</span></div>
        <div className="appearance-preview-controls"><input readOnly aria-label="预览任务名称" value="YOLO 训练实验" /><button type="button" className="button primary" aria-label="预览主按钮" onClick={(event) => event.preventDefault()}>主要按钮</button></div>
      </div>
    </section>
  </div>
}
