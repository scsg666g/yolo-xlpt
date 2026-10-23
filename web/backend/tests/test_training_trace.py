import json
import sys
from types import SimpleNamespace

import pytest

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


def fake_training(monkeypatch, directory, *, error=None, rank="-1"):
    from app import training_trace

    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("YOLO_FORGE_RUN_ID", "current-run")
    monkeypatch.setenv("YOLO_FORGE_TASK_ID", "task")
    monkeypatch.setattr(training_trace, "_value", lambda raw: int(raw) if raw.isdigit() else raw)
    callbacks = {}
    trainer = SimpleNamespace(save_dir=directory, args=SimpleNamespace(epochs=100, patience=35),
                              stopper=SimpleNamespace(best_epoch=21))

    class Model:
        def __init__(self, *_args, **_kwargs):
            self.trainer = trainer
            self.metrics = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.818})

        def add_callback(self, name, callback):
            callbacks[name] = callback

        def train(self, **_kwargs):
            assert not (directory / "training-result.json").exists()
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "results.csv").write_text("epoch,loss\n21,0.9\n56,0.8\n")
            # A trainer can invoke on_train_end before .train() itself returns.
            if "on_train_end" in callbacks:
                callbacks["on_train_end"](trainer)
            assert not (directory / "training-result.json").exists()
            if error:
                raise error

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    command = ["yolo", "detect", "train", "model=small.pt", "epochs=100", f"project={directory.parent}",
               f"name={directory.name}"]
    return training_trace, command, callbacks


def test_training_receipt_is_written_only_after_train_returns(tmp_path, monkeypatch):
    directory = tmp_path / "attempt-001"
    training_trace, command, _ = fake_training(monkeypatch, directory)

    training_trace.run(command, directory / "training-trace", 24, 10)

    record = json.loads((directory / "training-result.json").read_text())
    assert record["status"] == "completed"
    assert record["reason"] == "early_stopping"
    assert record["completed_epochs"] == 56
    assert record["best_epoch"] == 21
    assert record["patience"] == 35
    assert record["final_metrics"]["metrics/mAP50(B)"] == 0.818
    assert list(directory.glob("training-result.json.*.tmp")) == []


def test_validation_exception_after_train_end_writes_failure_and_reraises(tmp_path, monkeypatch):
    directory = tmp_path / "attempt-001"
    training_trace, command, _ = fake_training(monkeypatch, directory, error=RuntimeError("validation failed"))

    with pytest.raises(RuntimeError, match="validation failed"):
        training_trace.run(command, directory / "training-trace", 24, 10)

    record = json.loads((directory / "training-result.json").read_text())
    assert record["status"] == "failed"
    assert record["exit_code"] == 1
    assert record["reason"] == "exception"
    assert record["error_message"] == "RuntimeError: validation failed"


def test_non_primary_worker_does_not_publish_shared_training_result(tmp_path, monkeypatch):
    directory = tmp_path / "attempt-001"
    training_trace, command, _ = fake_training(monkeypatch, directory, rank="1")
    training_trace.run(command, directory / "training-trace", 24, 10)
    assert not (directory / "training-result.json").exists()


def test_non_detection_training_can_write_receipt_without_trace_hooks(tmp_path, monkeypatch):
    directory = tmp_path / "attempt-001"
    training_trace, command, callbacks = fake_training(monkeypatch, directory)
    command[1] = "classify"
    training_trace.run(command, directory / "training-trace", 24, 10, collect_trace=False)
    assert callbacks == {}
    assert (directory / "training-result.json").is_file()


@pytest.mark.parametrize("error,exit_code", [(SystemExit(0), 1), (SystemExit(2), 2), (KeyboardInterrupt(), 130)])
def test_early_interpreter_exit_is_failure_with_matching_wrapper_code(tmp_path, monkeypatch, error, exit_code):
    directory = tmp_path / "attempt-001"
    training_trace, command, _ = fake_training(monkeypatch, directory, error=error)
    with pytest.raises(SystemExit) as caught:
        training_trace.run(command, directory / "training-trace", 24, 10)
    assert caught.value.code == exit_code
    record = json.loads((directory / "training-result.json").read_text())
    assert record["status"] == "failed"
    assert record["reason"] == "exception"
    assert record["exit_code"] == exit_code


def test_result_disk_failure_preserves_successful_training_return(tmp_path, monkeypatch):
    directory = tmp_path / "attempt-001"
    training_trace, command, _ = fake_training(monkeypatch, directory)
    monkeypatch.setattr(training_trace, "_write_training_result", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    training_trace.run(command, directory / "training-trace", 24, 10)
    assert not (directory / "training-result.json").exists()


def test_receipt_uses_actual_resumed_epoch_target_over_command_target(tmp_path, monkeypatch):
    directory = tmp_path / "attempt-001"
    training_trace, command, _ = fake_training(monkeypatch, directory)
    command[4] = "epochs=200"
    training_trace.run(command, directory / "training-trace", 24, 10)
    record = json.loads((directory / "training-result.json").read_text())
    assert record["expected_epochs"] == 100
    assert record["reason"] == "early_stopping"
