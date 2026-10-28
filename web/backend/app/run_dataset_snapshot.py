"""Attach last-validated catalogue evidence to launch-time config evidence."""
from pathlib import PurePosixPath

from sqlalchemy import inspect, select

from .models import DatasetEntry


def enrich(db, snapshot, data_path, *, launch_mode=None):
    result = dict(snapshot)
    if launch_mode == 'resume':
        # Ultralytics may restore args.data from the checkpoint. The submitted
        # path cannot prove which config the trainer actually resolved.
        result['configuration_scope'] = 'resume_request'
        return result
    result['configuration_scope'] = 'launch_configuration'
    if not inspect(db.get_bind()).has_table('dataset_entries'):
        return result
    directory = str(PurePosixPath(data_path).parent)
    entry = db.scalar(select(DatasetEntry).where(
        DatasetEntry.node_id == result['node_id'], DatasetEntry.path == directory,
        DatasetEntry.status == 'ready',
    ).limit(1))
    details = (entry.details or {}) if entry else {}
    # A stale catalogue must not be presented as evidence for a changed YAML.
    if (not entry or not result.get('config_sha256') or
            result['config_sha256'] != details.get('config_sha256')):
        return result
    result.update(source='managed_catalog', verification='catalog_snapshot', dataset_id=entry.id,
                  fingerprint=details.get('fingerprint'), checked_at=details.get('checked_at'),
                  stats=dict(details.get('stats') or {}), classes=list(details.get('names') or []))
    return result
