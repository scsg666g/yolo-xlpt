import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from app.training_outcome import classify_training_exit


@pytest.fixture
def attempt(tmp_path):
    directory = tmp_path / "detect" / "task" / "attempt-001"
    directory.mkdir(parents=True)
    return {
        "run_dir": directory, "log_path": tmp_path / "attempt.log", "run_id": "current-run", "task_id": "task",
        "expected_epochs": 100, "completed_epochs": 56,
        "started_at": datetime.now(timezone.utc) - timedelta(hours=1),
    }


def write_record(attempt, **overrides):
    record = {
        "schema_version": 1, "run_id": "current-run", "task_id": "task", "status": "completed", "exit_code": 0,
        "reason": "early_stopping", "completed_epochs": 56, "expected_epochs": 100, "best_epoch": 21, "patience": 35,
        "finished_at": datetime.now(timezone.utc).isoformat(), "error_message": None,
        "final_metrics": {"metrics/mAP50(B)": 0.818},
    }
    record.update(overrides)
    (attempt["run_dir"] / "training-result.json").write_text(json.dumps(record))


def write_legacy(attempt, *, final_validation=True, error=None, directory=None):
    run_dir = attempt["run_dir"]
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").write_bytes(b"checkpoint")
    (run_dir / "results.csv").write_text("epoch,metrics/mAP50(B)\n21,0.818\n56,0.8016\n")
    text = (
        "[training-trace] collection disabled: OutOfMemoryError: CUDA out of memory\n"
        "EarlyStopping: Training stopped early as no improvement observed in last 35 epochs. "
        "Best results observed at epoch 21, best model saved as best.pt.\n"
        "56 epochs completed in 20.4 hours.\n"
    )
    if final_validation:
        text += f"Validating best.pt...\nResults saved to {directory or run_dir}\n"
    if error:
        text += f"{error}\n"
    attempt["log_path"].write_text(text)


def test_current_runner_receipt_recovers_early_stop_with_real_exit_code(attempt):
    write_record(attempt)
    outcome = classify_training_exit(**attempt)
    assert outcome["status"] == "completed"
    assert outcome["exit_code"] == 0
    assert outcome["completion_info"]["best_epoch"] == 21
    assert outcome["completion_info"]["final_metrics"]["metrics/mAP50(B)"] == 0.818


@pytest.mark.parametrize("overrides", [
    {"schema_version": 2}, {"schema_version": True}, {"run_id": "old-run"}, {"task_id": "different-task"},
    {"exit_code": True}, {"status": "completed", "exit_code": 1}, {"completed_epochs": "56"},
    {"finished_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()},
    {"finished_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()},
    {"finished_at": "2026-10-23T12:00:00"}, {"final_metrics": {"map": float("nan")}},
])
def test_invalid_or_other_attempt_receipt_never_completes_recovered_process(attempt, overrides):
    write_record(attempt, **overrides)
    outcome = classify_training_exit(**attempt)
    assert outcome["status"] == "interrupted"
    assert outcome["exit_code"] is None


def test_recorded_exception_preserves_failure_without_original_process(attempt):
    write_record(attempt, status="failed", exit_code=1, reason="exception", error_message="RuntimeError: validation failed")
    outcome = classify_training_exit(**attempt)
    assert outcome["status"] == "failed"
    assert outcome["exit_code"] == 1
    assert outcome["error_message"] == "RuntimeError: validation failed"


def test_nonzero_receipt_keeps_exception_metadata_with_known_process_exit(attempt):
    write_record(attempt, status="failed", exit_code=2, reason="exception", error_message="SystemExit: 2")
    outcome = classify_training_exit(**attempt, actual_exit_code=2)
    assert outcome["status"] == "failed"
    assert outcome["exit_code"] == 2
    assert outcome["completion_info"]["reason"] == "exception"
    assert outcome["completion_info"]["completed_epochs"] == 56


def test_real_nonzero_exit_takes_priority_over_success_receipt(attempt):
    write_record(attempt)
    outcome = classify_training_exit(**attempt, actual_exit_code=-9)
    assert outcome["status"] == "failed"
    assert outcome["exit_code"] == -9


def test_legacy_early_stop_requires_final_validation_and_fresh_artifacts(attempt):
    write_legacy(attempt)
    outcome = classify_training_exit(**attempt)
    assert outcome["status"] == "completed"
    assert outcome["exit_code"] is None
    assert outcome["completion_info"]["source"] == "legacy_log"
    assert outcome["completion_info"]["reason"] == "early_stopping"


def test_early_stop_and_epoch_summary_do_not_prove_validation_succeeded(attempt):
    write_legacy(attempt, final_validation=False)
    assert classify_training_exit(**attempt)["status"] == "interrupted"


def test_error_after_completion_markers_is_failure(attempt):
    write_legacy(attempt, error="RuntimeError: final validation failed")
    outcome = classify_training_exit(**attempt)
    assert outcome["status"] == "failed"
    assert outcome["error_message"] == "RuntimeError: final validation failed"


def test_legacy_result_from_other_directory_is_not_trusted(attempt):
    write_legacy(attempt, directory=attempt["run_dir"].parent / "attempt-000")
    assert classify_training_exit(**attempt)["status"] == "interrupted"


def test_previous_attempt_weights_do_not_prove_completion(attempt):
    write_legacy(attempt)
    old_time = (attempt["started_at"] - timedelta(hours=1)).timestamp()
    os.utime(attempt["run_dir"] / "weights" / "best.pt", (old_time, old_time))
    assert classify_training_exit(**attempt)["status"] == "interrupted"


def test_known_zero_exit_does_not_require_files_to_prove_success(attempt):
    outcome = classify_training_exit(**attempt, actual_exit_code=0)
    assert outcome["status"] == "completed"
    assert outcome["exit_code"] == 0
    assert outcome["completion_info"]["reason"] == "training_completed"


def test_collector_cuda_warning_alone_is_not_training_failure(attempt):
    attempt["log_path"].write_text("[training-trace] collection disabled: OutOfMemoryError: CUDA out of memory\n")
    assert classify_training_exit(**attempt)["status"] == "interrupted"
