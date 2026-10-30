import { useState } from 'react'
import { LoaderCircle, RotateCcw, Trash2 } from 'lucide-react'
import { api } from '../api'
import {
  resetTaskCenterPreferences,
  TASK_CENTER_PREFERENCES_DEFAULTS,
  updateTaskCenterPreferences,
  useTaskCenterPreferences,
} from '../preferences'
import { SettingsDisclosure } from './SettingsDisclosure'

const zoomPercent = (value: number) => `${Math.round(value * 100)}%`

export function TaskCenterSettings() {
  const preferences = useTaskCenterPreferences()
  const [resetPhase, setResetPhase] = useState<'idle' | 'saving' | 'success' | 'error'>('idle')
  const [resetMessage, setResetMessage] = useState('')

  async function requestWorkspaceReset() {
    if (!window.confirm('确定重置任务总览中已保存的卡片位置、连线和视口吗？训练任务及历史记录不会删除。')) return
    setResetPhase('saving')
    setResetMessage('')
    try {
      const current = await api.taskWorkspace()
      await api.saveTaskWorkspace({ revision: current.revision, modes: {} })
      setResetPhase('success')
      setResetMessage('任务总览布局已重置；下次打开时会按当前默认偏好重新排列。')
    } catch (reason) {
      setResetPhase('error')
      setResetMessage(reason instanceof Error ? reason.message : '任务总览布局重置失败，请稍后重试。')
    }
  }

  return <div className="task-center-settings">
    <SettingsDisclosure title="默认视图" description="控制每次打开任务总览时的初始排列和缩放。" badge={preferences.layoutMode === 'free' ? '自由布局' : '自动网格'} defaultOpen>
      <div className="settings-choice-row">
        <div><strong>布局模式</strong><span>自由布局可拖动卡片；自动网格会按任务顺序整齐排列。</span></div>
        <div className="settings-segmented" role="group" aria-label="任务总览默认布局">
          <button type="button" aria-pressed={preferences.layoutMode === 'free'} onClick={() => updateTaskCenterPreferences({ layoutMode: 'free' })}>自由布局</button>
          <button type="button" aria-pressed={preferences.layoutMode === 'grid'} onClick={() => updateTaskCenterPreferences({ layoutMode: 'grid' })}>自动网格</button>
        </div>
      </div>
      <label className="settings-range-row" htmlFor="task-center-default-zoom">
        <span><strong>默认缩放</strong><small>范围 60%～120%，避免任务较多时卡片缩得难以阅读。</small></span>
        <span className="settings-range-control">
          <input id="task-center-default-zoom" type="range" min="0.6" max="1.2" step="0.05" value={preferences.defaultZoom} onChange={(event) => updateTaskCenterPreferences({ defaultZoom: Number(event.target.value) })} />
          <output htmlFor="task-center-default-zoom">{zoomPercent(preferences.defaultZoom)}</output>
        </span>
      </label>
    </SettingsDisclosure>

    <SettingsDisclosure title="画布行为" description="辅助导航和任务关系展示，不改变训练调度。" defaultOpen>
      <label className="settings-toggle-row">
        <span><strong>显示小地图</strong><small>任务较多时快速定位画布区域。</small></span>
        <input type="checkbox" role="switch" checked={preferences.showMiniMap} onChange={(event) => updateTaskCenterPreferences({ showMiniMap: event.target.checked })} />
      </label>
      <label className="settings-toggle-row">
        <span><strong>显示关联线</strong><small>展示手动创建的任务关系；关联线仅用于整理和说明。</small></span>
        <input type="checkbox" role="switch" checked={preferences.showConnections} onChange={(event) => updateTaskCenterPreferences({ showConnections: event.target.checked })} />
      </label>
    </SettingsDisclosure>

    <SettingsDisclosure title="卡片显示" description="调整任务卡片的信息密度。">
      <div className="settings-choice-row">
        <div><strong>卡片密度</strong><span>紧凑模式适合同时观察更多任务，舒适模式更便于阅读。</span></div>
        <div className="settings-segmented" role="group" aria-label="任务卡片显示密度">
          <button type="button" aria-pressed={preferences.cardDensity === 'comfortable'} onClick={() => updateTaskCenterPreferences({ cardDensity: 'comfortable' })}>舒适</button>
          <button type="button" aria-pressed={preferences.cardDensity === 'compact'} onClick={() => updateTaskCenterPreferences({ cardDensity: 'compact' })}>紧凑</button>
        </div>
      </div>
    </SettingsDisclosure>

    <SettingsDisclosure title="布局数据管理" description="浏览器偏好和平台共享布局可以分别恢复。">
      <div className="settings-action-row">
        <div><strong>恢复浏览器默认偏好</strong><span>恢复默认布局、缩放、卡片密度和辅助显示。</span></div>
        <button type="button" className="button ghost compact" onClick={resetTaskCenterPreferences} disabled={
          preferences.layoutMode === TASK_CENTER_PREFERENCES_DEFAULTS.layoutMode
          && preferences.cardDensity === TASK_CENTER_PREFERENCES_DEFAULTS.cardDensity
          && preferences.defaultZoom === TASK_CENTER_PREFERENCES_DEFAULTS.defaultZoom
          && preferences.showMiniMap === TASK_CENTER_PREFERENCES_DEFAULTS.showMiniMap
          && preferences.showConnections === TASK_CENTER_PREFERENCES_DEFAULTS.showConnections
        }><RotateCcw size={14} />恢复默认</button>
      </div>
      <div className="settings-action-row danger">
        <div><strong>重置任务总览布局</strong><span>清除已保存的卡片位置、连线和视口；不会删除任务。</span></div>
        <button type="button" className="button ghost compact" disabled={resetPhase === 'saving'} onClick={() => void requestWorkspaceReset()}>
          {resetPhase === 'saving' ? <LoaderCircle size={14} className="spin" /> : <Trash2 size={14} />}
          {resetPhase === 'saving' ? '正在重置' : '重置布局'}
        </button>
      </div>
      {resetMessage && <p className={resetPhase === 'error' ? 'settings-inline-error' : 'settings-inline-status'} role={resetPhase === 'error' ? 'alert' : 'status'}>{resetMessage}</p>}
      <p className="settings-browser-note">{preferences.storageAvailable ? '以上显示偏好仅保存在当前浏览器。' : '浏览器存储不可用，以上显示偏好在刷新后可能恢复默认。'}</p>
    </SettingsDisclosure>
  </div>
}
