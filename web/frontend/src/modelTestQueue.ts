import type { ModelTest } from './types'

export interface ImageTestReference {
  test: ModelTest
  index: number
  key: string
}

export interface SelectedImage {
  id: number
  file: File
  previewUrl: string
  attempt?: ImageTestReference
  result?: ImageTestReference
}

const active = new Set(['uploading', 'preparing', 'queued', 'running', 'cancel_requested'])

export function imageTestKey(model: string, confidence: number, iou: number, classification: boolean) {
  return JSON.stringify([model, confidence, classification ? null : iou])
}

function referenceMatches(reference: ImageTestReference | undefined, key: string) {
  if (!reference) return false
  if (reference.key === key) return true
  // A standalone model's task type can first become known after inference.
  if ((reference.test.task_type || reference.test.model.task_type) !== 'classify') return false
  const previous = JSON.parse(reference.key)
  const current = JSON.parse(key)
  return previous[0] === current[0] && previous[1] === current[1]
}

export function imageResultIsCurrent(image: SelectedImage, key: string) {
  return referenceMatches(image.result, key)
}

export function imageTestState(image: SelectedImage, key: string) {
  if (image.attempt && active.has(image.attempt.test.status)) return 'active'
  if (referenceMatches(image.attempt, key) && ['failed', 'cancelled'].includes(image.attempt!.test.status)) return 'failed'
  if (referenceMatches(image.result, key)) return 'completed'
  return image.result ? 'stale' : 'pending'
}

export function refreshImageTest(images: SelectedImage[], test: ModelTest) {
  return images.map((image) => {
    if (image.attempt?.test.id !== test.id) return image
    const reference = { ...image.attempt, test }
    return { ...image, attempt: reference, ...(test.status === 'completed' ? { result: reference } : {}) }
  })
}

export function pendingTestImages(images: SelectedImage[], key: string) {
  return images.filter((image) => !['completed', 'active'].includes(imageTestState(image, key)))
}

// Store the actual batch index against a stable selection ID, never a UI index.
export function bindImageTest(images: SelectedImage[], ids: number[], key: string, test: ModelTest) {
  return images.map((image) => {
    const index = ids.indexOf(image.id)
    if (index < 0) return image
    const reference = { test, index, key }
    return { ...image, attempt: reference, ...(test.status === 'completed' ? { result: reference } : {}) }
  })
}

export function removeImageTest(images: SelectedImage[], testId: string) {
  return images.map((image) => ({ ...image,
    attempt: image.attempt?.test.id === testId ? undefined : image.attempt,
    result: image.result?.test.id === testId ? undefined : image.result,
  }))
}

export function imageResultUrl(image: SelectedImage | undefined) {
  const result = image?.result
  return result ? `/api/model-tests/${result.test.id}/images/${result.index}/result` : ''
}
