from types import SimpleNamespace

from app.training_trace import TraceCollector


def test_non_primary_ddp_rank_does_not_collect(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    collector = TraceCollector(tmp_path / "trace")

    collector.on_train_start(SimpleNamespace())
    collector.on_train_batch_start(SimpleNamespace())
    collector.on_train_batch_end(SimpleNamespace())
    collector.on_train_end(SimpleNamespace())

    assert collector.step == -1
    assert not collector.directory.exists()


def test_trace_write_failure_disables_collection_without_raising(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    collector = TraceCollector(tmp_path / "trace")
    collector.step = 0
    collector.last_batch = {}
    monkeypatch.setattr(collector, "_write", lambda: (_ for _ in ()).throw(OSError("disk unavailable")))
    trainer = SimpleNamespace(epoch=0, loss_items={}, optimizer=None)

    collector.on_train_batch_end(trainer)

    assert collector.disabled
    assert collector.error == "OSError: disk unavailable"


def test_trace_write_uses_process_specific_temporary_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RANK", "0")
    collector = TraceCollector(tmp_path / "trace")
    collector.directory.mkdir()
    collector.samples.append({"epoch": 1, "batch": 0})

    collector._write()

    assert (collector.directory / "trace.json").is_file()
    assert list(collector.directory.glob("trace.json.*.tmp")) == []
