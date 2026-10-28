import json
import subprocess
from types import SimpleNamespace

import pytest

from app.local_devices import bind_command, visible_inventory
from app import local_devices


GPUS = [{'index': 0, 'uuid': 'GPU-A'}, {'index': 1, 'uuid': 'GPU-B'}]


@pytest.fixture(autouse=True)
def cuda_driver(monkeypatch):
    local_devices._visible_cuda_uuids.cache_clear()

    def run_probe(command, **kwargs):
        mask = kwargs['env']['CUDA_VISIBLE_DEVICES']
        identities = {'0': 'GPU-A', '1': 'GPU-B', 'GPU-A': 'GPU-A', 'GPU-B': 'GPU-B'}
        values = [identities.get(part, 'unknown') for part in mask.split(',')]
        return SimpleNamespace(stdout=json.dumps(values))

    monkeypatch.setattr(local_devices.subprocess, 'run', run_probe)
    yield
    local_devices._visible_cuda_uuids.cache_clear()


@pytest.mark.parametrize('mask,indices', [(None, [0, 1]), ('1,0', [0, 1]), ('1', [1]),
    ('GPU-B', [1]), ('', []), ('-1', []), ('bad', []), ('1,,0', []),
    ('9,0', []), ('0,9,1', []), ('GPU-', []), ('0,0', [])])
def test_visibility_preserves_physical_indices(mask, indices):
    environment = {} if mask is None else {'CUDA_VISIBLE_DEVICES': mask}
    assert [gpu['index'] for gpu in visible_inventory(GPUS, environment)] == indices


def test_reserved_uuid_and_wrapper_command_agree_under_reordered_parent_mask():
    cli = ['yolo', 'detect', 'train', 'device=0']
    wrapper = ['python', 'training_trace.py', '--command-json', json.dumps(cli), *cli]
    command, environment = bind_command(wrapper, '0', [GPUS[0]], GPUS,
                                        {'CUDA_VISIBLE_DEVICES': '1,0', 'MARKER': 'keep'})
    assert environment == {'CUDA_VISIBLE_DEVICES': 'GPU-A', 'MARKER': 'keep'}
    assert command[-1] == 'device=0'
    assert json.loads(command[command.index('--command-json') + 1])[-1] == 'device=0'


def test_multi_gpu_order_is_requested_order_even_if_inventory_and_reservations_reorder():
    command, env = bind_command(['yolo', 'device=1,0'], '1,0', GPUS, list(reversed(GPUS)), {})
    assert command == ['yolo', 'device=0,1']
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-B,GPU-A'


def test_uuid_survives_index_change_and_never_widens_parent_visibility():
    changed = [{'index': 1, 'uuid': 'GPU-A'}, {'index': 0, 'uuid': 'GPU-B'}]
    _, env = bind_command(['yolo', 'device=0'], '0', [GPUS[0]], changed,
                          {'CUDA_VISIBLE_DEVICES': 'GPU-A'})
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-A'
    with pytest.raises(ValueError, match='允许范围'):
        bind_command(['yolo', 'device=0'], '0', [GPUS[0]], GPUS,
                     {'CUDA_VISIBLE_DEVICES': '1'})


def test_cpu_and_automatic_selection_preserve_environment_and_command():
    for device in ['cpu', 'mps', '', '-1']:
        command, env = bind_command(['yolo', f'device={device}'], device, [], GPUS,
                                    {'CUDA_VISIBLE_DEVICES': '1'})
        assert command == ['yolo', f'device={device}']
        assert env['CUDA_VISIBLE_DEVICES'] == '1'


def test_missing_reservation_cannot_launch_gpu_work():
    with pytest.raises(ValueError, match='预约缺失'):
        bind_command(['yolo', 'device=0'], '0', [], GPUS, {})


def test_legacy_index_reservation_uses_current_uuid_and_never_reverts_to_unverified_number():
    _, env = bind_command(['yolo', 'device=0'], '0', [{'index': 0, 'uuid': None}], GPUS, {})
    assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-A'
    with pytest.raises(ValueError, match='UUID'):
        bind_command(['yolo', 'device=0'], '0', [{'index': 0}], [{'index': 0}], {})


def test_numeric_mask_uses_cuda_uuid_enumeration_instead_of_nvml_index(monkeypatch):
    seen = []

    def different_driver_order(command, **kwargs):
        seen.append(kwargs['env'])
        return SimpleNamespace(stdout=json.dumps(['GPU-B']))

    monkeypatch.setattr(local_devices.subprocess, 'run', different_driver_order)
    environment = {'CUDA_VISIBLE_DEVICES': '0', 'CUDA_DEVICE_ORDER': 'FASTEST_FIRST'}
    assert visible_inventory(GPUS, environment) == [GPUS[1]]
    assert seen[0]['CUDA_VISIBLE_DEVICES'] == '0'
    assert seen[0]['CUDA_DEVICE_ORDER'] == 'FASTEST_FIRST'
    with pytest.raises(ValueError, match='允许范围'):
        bind_command(['yolo', 'device=0'], '0', [GPUS[0]], GPUS, environment)
    command, child = bind_command(['yolo', 'device=1'], '1', [GPUS[1]], GPUS, environment)
    assert command[-1] == 'device=0'
    assert child['CUDA_VISIBLE_DEVICES'] == 'GPU-B'


@pytest.mark.parametrize('failure', [OSError('no driver'), subprocess.TimeoutExpired('probe', 5),
                                   ValueError('invalid output')])
def test_unverifiable_numeric_visibility_cannot_launch_outside_parent_mask(monkeypatch, failure):
    def failed_probe(*args, **kwargs):
        raise failure

    monkeypatch.setattr(local_devices.subprocess, 'run', failed_probe)
    assert visible_inventory(GPUS, {'CUDA_VISIBLE_DEVICES': '0'}) == []
    with pytest.raises(ValueError, match='允许范围'):
        bind_command(['yolo', 'device=0'], '0', [GPUS[0]], GPUS, {'CUDA_VISIBLE_DEVICES': '0'})
