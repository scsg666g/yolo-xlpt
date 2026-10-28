import type { DatasetOperation } from './types'

type WaitState = { label: string; badge: string; reason: 'offline' | 'resource' | 'training' | 'lock' }

/** Optional fields are deliberately ignored for old operations and older Agents. */
export function datasetOperationWait(job: DatasetOperation): WaitState | null {
  if (!['pending', 'running', 'committing'].includes(job.status)) return null
  if (job.waiting_offline) return { label: '等待远程节点上线', badge: '节点离线', reason: 'offline' }
  if (job.waiting_resource || job.progress?.stage === 'paused_resource') {
    const reason = job.progress?.resource_reason
    const label = reason === 'low_disk_space' ? '可用磁盘空间不足，后台操作已暂停'
      : reason === 'disk_pressure' ? '磁盘压力较高，后台操作已暂停'
        : job.status === 'pending' ? '等待后台资源配额' : '资源繁忙，后台操作已暂停'
    return { label, badge: '资源暂停', reason: 'resource' }
  }
  if (job.waiting_path_lock) return { label: '等待同路径操作完成', badge: '等待同路径', reason: 'lock' }
  if (job.waiting_training) return { label: '等待训练任务结束', badge: '等待训练', reason: 'training' }
  if (job.waiting_lock || job.progress?.stage === 'waiting_lock') return { label: '等待文件或工作区锁', badge: '等待文件锁', reason: 'lock' }
  return null
}
