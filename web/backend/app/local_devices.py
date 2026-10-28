"""Bind physical reservations to CUDA logical indices without widening visibility."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from functools import lru_cache


# The CUDA driver applies both CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER.
# NVML/nvidia-smi indices alone cannot resolve numeric masks on mixed GPU hosts.
_CUDA_UUID_PROBE = '''
import ctypes, json, uuid
cuda = ctypes.CDLL('libcuda.so.1')
def check(result):
    if result != 0:
        raise RuntimeError('CUDA inventory unavailable')
check(cuda.cuInit(0))
count = ctypes.c_int()
check(cuda.cuDeviceGetCount(ctypes.byref(count)))
get_uuid = getattr(cuda, 'cuDeviceGetUuid_v2', cuda.cuDeviceGetUuid)
identities = []
for index in range(count.value):
    device = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(device), index))
    value = (ctypes.c_ubyte * 16)()
    check(get_uuid(ctypes.byref(value), device))
    identities.append('GPU-' + str(uuid.UUID(bytes=bytes(value))))
print(json.dumps(identities))
'''


@lru_cache(maxsize=32)
def _visible_cuda_uuids(mask: str, order: str | None, inventory_ids: tuple[str, ...],
                        generation: int) -> tuple[str, ...]:
    # Cache for one scheduler sampling window; inventory/order changes invalidate
    # it immediately. The subprocess never starts training or changes visibility.
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': mask}
    if order is None:
        environment.pop('CUDA_DEVICE_ORDER', None)
    else:
        environment['CUDA_DEVICE_ORDER'] = order
    try:
        result = subprocess.run([sys.executable, '-B', '-c', _CUDA_UUID_PROBE],
                                env=environment, capture_output=True, text=True,
                                check=True, timeout=5)
        identities = json.loads(result.stdout)
        if (not isinstance(identities, list) or len(identities) != len(mask.split(','))
                or any(not isinstance(value, str) or value not in inventory_ids for value in identities)
                or len(set(identities)) != len(identities)):
            return ()
        return tuple(identities)
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()


def visible_inventory(gpus: list[dict], environment: dict | None = None) -> list[dict]:
    environment = os.environ if environment is None else environment
    mask = environment.get('CUDA_VISIBLE_DEVICES')
    if mask is None:
        return gpus
    tokens = [part.strip() for part in mask.split(',')]
    # CUDA stops at the first invalid index; conservatively reject malformed masks.
    if not tokens or any(not token or token == '-1' for token in tokens):
        return []
    if any(not token.isdigit() and not token.startswith(('GPU-', 'MIG-')) for token in tokens):
        return []
    if any(token.isdigit() for token in tokens):
        identities = _visible_cuda_uuids(mask, environment.get('CUDA_DEVICE_ORDER'),
                                        tuple(str(gpu.get('uuid') or '') for gpu in gpus),
                                        int(time.monotonic() // 30))
        return [gpu for gpu in gpus if gpu.get('uuid') in identities]
    selected = set()
    for token in tokens:
        matches = [gpu for gpu in gpus if (
            int(token) == gpu.get('index') if token.isdigit()
            else str(gpu.get('uuid') or '').startswith(token)
        )]
        if len(matches) != 1 or matches[0].get('index') in selected:
            return []
        selected.add(matches[0].get('index'))
    return [gpu for gpu in gpus if gpu.get('index') in selected]


def bind_command(command: list[str], device: str, assignments: list[dict],
                 inventory: list[dict], environment: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    """Use reservation UUIDs in the child mask and contiguous CLI device indices.

    Existing processes and their saved commands are never rewritten. CPU, MPS
    and automatic selection retain their established behavior.
    """
    parts = [part.strip() for part in device.split(',')]
    if not parts or not all(part.isdigit() for part in parts):
        return command, environment
    if not any(item.startswith('device=') for item in command):
        return command, environment
    allowed = visible_inventory(inventory, environment)
    by_index = {item.get('index'): item for item in assignments}
    mask = []
    for part in parts:
        index = int(part)
        assignment = by_index.get(index)
        if assignment is None:
            raise ValueError('本机 GPU 预约缺失，请重新排队训练')
        identity = assignment.get('uuid')
        matches = [gpu for gpu in allowed if (
            gpu.get('uuid') == identity if identity else gpu.get('index') == index
        )]
        if len(matches) != 1:
            raise ValueError('预约 GPU 已变化或不在 CUDA_VISIBLE_DEVICES 允许范围，请检查设备配置')
        resolved_uuid = matches[0].get('uuid')
        if not resolved_uuid:
            raise ValueError('本机 GPU UUID 无法核验，请检查驱动并刷新设备清单')
        mask.append(resolved_uuid)
    logical = ','.join(str(index) for index in range(len(mask)))
    def replace(items):
        return [f'device={logical}' if item.startswith('device=') else item for item in items]
    updated = replace(command)
    if '--command-json' in updated:
        position = updated.index('--command-json') + 1
        original = json.loads(updated[position])
        updated[position] = json.dumps(replace(original), ensure_ascii=False)
    return updated, {**environment, 'CUDA_VISIBLE_DEVICES': ','.join(mask)}
