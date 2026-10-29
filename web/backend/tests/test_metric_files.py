from pathlib import Path
import importlib.util
import pytest

from app.metric_files import MetricCsvReader


@pytest.fixture
def agent():
    script = Path(__file__).resolve().parents[2] / 'agent/monitor_agent.py'
    spec = importlib.util.spec_from_file_location('metric_agent_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_incremental_csv_waits_for_complete_epoch_and_reuses_unchanged_file(tmp_path, monkeypatch):
    path = tmp_path / 'results.csv'
    path.write_text(' epoch, time,score\n1,1,0.1\n2,2,')
    reader = MetricCsvReader()
    first = reader.read(path)
    assert first['completed'] == 1
    assert first['latest'] == {'score': 0.1}
    original_open = Path.open
    # A bounded suffix check still runs to detect same-timestamp rewrites.
    original_metric = reader._metric
    monkeypatch.setattr(reader, '_metric', lambda row: (_ for _ in ()).throw(AssertionError('unchanged rows reparsed')))
    assert reader.read(path) == first
    monkeypatch.setattr(reader, '_metric', original_metric)
    with path.open('a') as stream:
        stream.write('0.2\n')
    updated = reader.read(path)
    assert [item['epoch'] for item in updated['metrics']] == [1, 2]
    assert updated['latest'] == {'score': 0.2}
    assert updated['generation'] == first['generation']


def test_csv_reset_and_final_partial_record(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('epoch,time,score\n1,1,0.1\n')
    reader = MetricCsvReader()
    first = reader.read(path)
    path.write_text('epoch,time,score\n2,2,0.2\n')
    changed = reader.read(path)
    assert changed['generation'] > first['generation']
    assert changed['metrics'] == [{'epoch': 2, 'elapsed': 2.0, 'values': {'score': 0.2}}]
    path.write_text('epoch,time,score\n3,3,0.3')
    assert reader.read(path)['metrics'] == []
    final = reader.read(path, final=True)
    assert final['completed'] == 3
    assert final['latest'] == {'score': 0.3}
    assert reader.read(path, final=True) == final


def test_csv_replacement_and_partial_rows_do_not_publish_invalid_metrics(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('epoch,time,score\n1,1,0.1\n')
    reader = MetricCsvReader()
    reader.read(path)
    replacement = tmp_path / 'replacement'
    replacement.write_text('epoch,time,score\n2,NaN,Infinity\n3,3\n')
    replacement.replace(path)
    snapshot = reader.read(path)
    assert snapshot['metrics'] == [{'epoch': 2, 'elapsed': 0.0, 'values': {'score': None}}]
    path.unlink()
    assert reader.read(path)['metrics'] == []


def test_csv_rejects_negative_epoch_before_sending_reset_snapshot(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('epoch,time,score\n-1,1,0.1\n1,2,0.2\n')
    snapshot = MetricCsvReader().read(path, include_metadata=True)
    assert snapshot['has_metrics']
    assert [point['epoch'] for point in snapshot['metrics']] == [1]


def test_metric_reader_marks_complete_replacement_but_not_missing_file_as_authoritative(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('epoch,time,score\n1,1,0.1\n2,2,0.2\n')
    reader = MetricCsvReader()
    first = reader.read(path, include_metadata=True)
    assert first['reset'] and first['has_metrics']
    path.write_text('epoch,time,score\n1,3,0.8\n')
    replaced = reader.read(path, include_metadata=True)
    assert replaced['reset'] and replaced['has_metrics']
    assert [point['epoch'] for point in replaced['metrics']] == [1]
    path.unlink()
    missing = reader.read(path, include_metadata=True)
    assert not missing['authoritative'] and not missing['has_metrics']


def test_incremental_consumer_receives_only_new_rows_and_full_reader_remains_compatible(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('epoch,time,score\n1,1,0.1\n2,2,0.2\n')
    reader = MetricCsvReader()
    first = reader.read(path, incremental=True)
    assert [item['epoch'] for item in first['metrics']] == [1, 2]
    assert reader.read(path, incremental=True)['metrics'] == []
    with path.open('a') as stream:
        stream.write('3,3,0.3\n')
    delta = reader.read(path, incremental=True)
    assert [item['epoch'] for item in delta['metrics']] == [3]
    assert delta['completed'] == 3
    assert [item['epoch'] for item in reader.read(path)['metrics']] == [1, 2, 3]


def test_agent_resends_same_epoch_after_results_rewrite(agent, tmp_path):
    run_dir = tmp_path / 'attempt-001'
    run_dir.mkdir()
    path = run_dir / 'results.csv'
    path.write_text('epoch,time,score\n1,1,0.1\n')
    sent = set()
    assert agent.read_remote_progress(run_dir, sent_epochs=sent)[2][0]['values'] == {'score': 0.1}
    sent.add(1)
    replacement = tmp_path / 'replacement.csv'
    replacement.write_text('epoch,time,score\n1,1,0.9\n')
    replacement.replace(path)
    _, _, rows = agent.read_remote_progress(run_dir, sent_epochs=sent)
    assert sent == set()
    assert rows == [{'epoch': 1, 'elapsed': 1.0, 'values': {'score': 0.9}}]


def test_agent_reset_requires_valid_snapshot_and_resends_after_csv_rewrite(agent, tmp_path):
    run_dir = tmp_path / 'attempt-001'
    run_dir.mkdir()
    path = run_dir / 'results.csv'
    sent, pending, state = set(), {}, {}
    path.write_text('epoch,time,score\n1,1,0.1\n2,2,0.2\n')
    completed, _, rows = agent.read_remote_progress(
        run_dir, sent_epochs=sent, pending_metrics=pending, metric_state=state)
    assert completed == 2 and state['reset_pending']
    sent.update(point['epoch'] for point in rows)
    state.pop('reset_pending')  # Simulate a successful center acknowledgement.
    path.write_text('epoch,time,score\n1,3,0.9\n')
    completed, _, rows = agent.read_remote_progress(
        run_dir, sent_epochs=sent, pending_metrics=pending, metric_state=state)
    assert completed == 1 and state['reset_pending'] and sent == set()
    assert rows == [{'epoch': 1, 'elapsed': 3.0, 'values': {'score': 0.9}}]
    # A failed center request must leave the reset flag for the next heartbeat.
    assert state['reset_pending']
    path.write_text('epoch,time,score\n')
    agent.read_remote_progress(run_dir, sent_epochs=sent, pending_metrics=pending, metric_state=state)
    assert not state.get('reset_pending')


def test_agent_allows_appended_correction_for_acknowledged_epoch(agent, tmp_path):
    run_dir = tmp_path / 'attempt-001'
    run_dir.mkdir()
    path = run_dir / 'results.csv'
    path.write_text('epoch,time,score\n1,1,0.1\n')
    sent = set()
    agent.read_remote_progress(run_dir, sent_epochs=sent)
    sent.add(1)
    with path.open('a') as stream:
        stream.write('1,2,0.9\n')
    _, latest, rows = agent.read_remote_progress(run_dir, sent_epochs=sent)
    assert sent == set()
    assert rows == [{'epoch': 1, 'elapsed': 2.0, 'values': {'score': 0.9}}]
    assert latest == {'score': 0.9}


def test_standalone_agent_embeds_same_metric_core_and_bounds_log_reads(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'agent/monitor_agent.py'
    spec = importlib.util.spec_from_file_location('metric_agent_test', script)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    import app.metric_files as core
    assert agent.METRIC_FILES_SOURCE == Path(core.__file__).read_text()
    calls = []
    monkeypatch.setattr(agent, 'recent_log_text', lambda path, maximum_bytes: calls.append(maximum_bytes) or '\n'.join(str(i) for i in range(120)))
    assert agent.recent_log_lines(tmp_path / 'log') == [str(i) for i in range(20, 120)]
    assert calls == [1024 * 1024]
