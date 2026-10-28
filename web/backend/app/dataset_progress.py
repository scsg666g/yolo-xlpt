"""Bounded optional progress stored in existing operation JSON; no schema migration."""
from datetime import datetime, timezone
import math
import time

STAGES = {'scan', 'source_verify', 'pack', 'archive_hash', 'transfer', 'unpack',
          'content_validate', 'target_verify', 'delete_verify', 'remove', 'check',
          'discover', 'prepare_upload', 'export', 'validate', 'import', 'waiting_lock'}


def clean_progress(value):
    if not isinstance(value, dict) or not isinstance(value.get('stage'), str) or value['stage'] not in STAGES:
        raise ValueError('进度阶段无效')
    result = {'stage': value['stage'], 'unit': value.get('unit', 'files'),
              'current_file': str(value.get('current_file', ''))[-400:]}
    if result['unit'] not in ('files', 'bytes', 'directories'):
        raise ValueError('进度单位无效')
    for key in ('processed', 'total'):
        number = value.get(key, 0 if key == 'processed' else None)
        if (key == 'processed' and number is None) or (number is not None and (type(number) is not int or not 0 <= number <= 10**15)):
            raise ValueError('进度计数无效')
        result[key] = number
    if result['total'] is not None and result['processed'] > result['total']:
        raise ValueError('进度超过总量')
    sequence = value.get('sequence')
    if sequence is not None and (type(sequence) is not int or not 0 <= sequence <= 10**15):
        raise ValueError('进度序号无效')
    result['sequence'] = sequence
    return result


def record_progress(op, value, run_id, executor):
    """Late heartbeats from a previous phase/retry cannot overwrite current progress."""
    if not run_id or run_id != op.payload.get('progress_run_id'):
        return False
    value = clean_progress(value)
    now = datetime.now(timezone.utc).isoformat()
    payload = dict(op.payload)
    previous = payload.get('progress') or {}
    if (previous.get('run_id') == run_id and value['sequence'] is not None
            and value['sequence'] <= (previous.get('sequence') or 0)):
        return False
    same = (previous.get('stage'), previous.get('phase'), previous.get('run_id')) == (
        value['stage'], payload['phase'], run_id)
    same = same and previous.get('executor') == executor and value['processed'] >= previous.get('processed', 0)
    steps = list(payload.get('progress_steps', []))[-15:]
    if (previous and not same and previous.get('run_id') == run_id
            and previous.get('stage') != value['stage']
            and previous.get('stage') not in {'export', 'prepare_upload', 'validate', 'import'}):
        steps.append({k: previous.get(k) for k in ('stage', 'phase', 'executor')})
    changed = not same or previous.get('processed') != value['processed']
    started = previous['started_at'] if same else now
    rate = None
    elapsed = (datetime.fromisoformat(now) - datetime.fromisoformat(started)).total_seconds()
    baseline = previous.get('baseline', value['processed']) if same else value['processed']
    if elapsed >= 3 and value['processed'] > baseline:
        rate = (value['processed'] - baseline) / elapsed
    eta = ((value['total'] - value['processed']) / rate
           if rate and value['total'] is not None else None)
    payload['progress'] = {**value, 'phase': payload['phase'], 'run_id': run_id,
                           'executor': executor, 'started_at': started, 'sampled_at': now,
                           'changed_at': now if changed else previous.get('changed_at', now),
                           'baseline': baseline, 'rate': rate,
                           'eta_seconds': round(eta) if eta is not None and math.isfinite(eta) else None}
    payload['progress_steps'] = steps[-16:]
    op.payload = payload
    return True


def progress_reporter(persist, interval=2, start_sequence=0):
    """Throttle work callbacks without delaying cancellation/lease checks."""
    last = [0.0, None, None]
    sequence = [start_sequence]
    def report(value):
        now = time.monotonic()
        stage, done, total = value['stage'], value.get('processed', 0), value.get('total')
        finished = total is not None and done == total and (stage, done) != (last[1], last[2])
        if stage != last[1] or now - last[0] >= interval or finished:
            sequence[0] += 1
            persist({**value, 'sequence': sequence[0]})
            last[:] = [now, stage, done]
    return report
