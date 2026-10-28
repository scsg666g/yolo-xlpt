import { sha256 } from '@noble/hashes/sha256'
import { bytesToHex } from '@noble/hashes/utils'

/** Streams hashes without requiring HTTPS-only WebCrypto or loading a whole image. */
export async function hashDatasetFile(file: Blob, signal: AbortSignal): Promise<string> {
  const value = sha256.create()
  for (let offset = 0; offset < file.size; offset += 1024 * 1024) {
    signal.throwIfAborted()
    value.update(new Uint8Array(await file.slice(offset, offset + 1024 * 1024).arrayBuffer()))
  }
  return bytesToHex(value.digest())
}
export function datasetSessionKey(location: string, name: string, manifest: Array<{ path: string; size: number; sha256: string }>) {
  return 'dataset-upload:' + bytesToHex(sha256(new TextEncoder().encode(JSON.stringify([location, name, manifest]))))
}

export const datasetPhaseLabel: Record<string, string> = { discover: '扫描候选数据集', check: '校验保存目录', prepare_upload: '准备上传快照', export: '导出源数据', validate: '校验图片与标签', import: '传输并写入目标', remove: '清理源目录' }
