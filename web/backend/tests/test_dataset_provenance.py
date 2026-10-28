import hashlib
import importlib.util
from datetime import datetime, timezone
from pathlib import Path

from app.dataset_provenance import capture, clean
from app.models import DatasetEntry
from app.run_dataset_snapshot import enrich
from test_services import make_session


def test_launch_snapshot_hashes_only_configuration_and_is_bounded(tmp_path):
    config = tmp_path / 'data.yaml'
    config.write_text('path: .\ntrain: images/train\n', encoding='utf-8')
    raw = capture(config, task_type='detect', node_id='node-a')
    assert raw['config_sha256'] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert raw['verification'] == 'config_only'
    assert clean(raw)['node_id'] == 'node-a'
    assert capture(tmp_path, task_type='classify')['verification'] == 'path_only'


def test_catalogue_snapshot_is_used_only_when_yaml_matches(tmp_path):
    session = make_session(tmp_path)
    config = tmp_path / 'dataset' / 'data.yaml'
    config.parent.mkdir()
    config.write_text('names: [cat]\n', encoding='utf-8')
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    session.add(DatasetEntry(
        id='dataset', location_id='location', node_id='local', name='dataset',
        path=str(config.parent), status='ready', ownership='managed',
        details={'config_sha256': digest, 'fingerprint': 'f' * 64,
                 'checked_at': '2026-10-01T00:00:00+00:00', 'stats': {'train_images': 4},
                 'names': ['cat']},
    ))
    session.commit()
    raw = {'version': 1, 'node_id': 'local', 'task_type': 'detect', 'source': 'runtime_config',
           'verification': 'config_only', 'captured_at': datetime.now(timezone.utc).isoformat(),
           'config_sha256': digest}
    snapshot = enrich(session, clean(raw), config)
    assert snapshot['source'] == 'managed_catalog'
    assert snapshot['fingerprint'] == 'f' * 64 and snapshot['stats'] == {'train_images': 4}
    resumed = enrich(session, clean(raw), config, launch_mode='resume')
    assert resumed['configuration_scope'] == 'resume_request'
    assert resumed['source'] == 'runtime_config'
    assert 'fingerprint' not in resumed and 'stats' not in resumed
    raw['config_sha256'] = '0' * 64
    assert enrich(session, clean(raw), config)['source'] == 'runtime_config'
    session.close()


def test_agent_core_generator_preserves_embedded_launchers_and_is_idempotent():
    source = Path(__file__).resolve().parents[2] / 'agent' / 'build_dataset_cores.py'
    spec = importlib.util.spec_from_file_location('portable_core_builder', source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    original = '''LAUNCHER = \"\"\"if __name__ == \"__main__\": main()\"\"\"
def main(): pass
if __name__ == \"__main__\": main()
'''
    backend = Path(__file__).resolve().parents[1] / 'app'
    generated = builder.embed_sources(original, backend)
    namespace = {'__name__': 'test_agent'}
    exec(generated, namespace)
    assert namespace['LAUNCHER'] == 'if __name__ == \"__main__\": main()'
    for name, module in [('DATASET_FILES', 'dataset_files'), ('DATASET_DISCOVERY', 'dataset_discovery_files'),
                         ('DATASET_PROVENANCE', 'dataset_provenance'), ('METRIC_FILES', 'metric_files')]:
        assert namespace[name]
        assert namespace[name + '_SOURCE'] == (backend / (module + '.py')).read_text(encoding='utf-8')
    assert builder.embed_sources(generated, backend) == generated
