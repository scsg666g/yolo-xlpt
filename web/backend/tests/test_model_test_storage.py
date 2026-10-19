from types import SimpleNamespace

import pytest

import app.model_test_storage as storage


def configure_storage(tmp_path, monkeypatch, *, history_quota=0, task_quota=0, min_free=0, image_reserve=10):
    monkeypatch.setattr(storage, "HISTORY_ROOT", tmp_path / "history")
    monkeypatch.setattr(storage, "MODEL_TEST_HISTORY_QUOTA_BYTES", history_quota)
    monkeypatch.setattr(storage, "MODEL_TEST_TASK_QUOTA_BYTES", task_quota)
    monkeypatch.setattr(storage, "MODEL_TEST_MIN_FREE_BYTES", min_free)
    monkeypatch.setattr(storage, "MODEL_TEST_IMAGE_RESERVE_BYTES", image_reserve)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=10_000_000))


def test_nonnegative_capacity_config(monkeypatch):
    from app.config import nonnegative_int_env

    monkeypatch.setenv("TEST_CAPACITY_BYTES", "123")
    assert nonnegative_int_env("TEST_CAPACITY_BYTES", 9) == 123
    monkeypatch.setenv("TEST_CAPACITY_BYTES", "-1")
    with pytest.raises(RuntimeError, match="非负整数"):
        nonnegative_int_env("TEST_CAPACITY_BYTES", 9)


def test_estimate_reserves_model_upload_normalized_results_and_zip(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch, image_reserve=10)
    estimate = storage.estimate_test_storage(model_bytes=100, declared_upload_bytes=20, image_count=3)

    assert estimate.model_bytes == 100
    assert estimate.declared_upload_bytes == 20
    assert estimate.image_bytes == 30
    assert estimate.fixed_bytes == 64 * 1024**2
    assert estimate.history_bytes == 30 * 3 + 64 * 1024**2
    assert estimate.peak_bytes == 100 + 20 + 30 * 3 + 64 * 1024**2


def test_prepare_reservation_is_counted_against_global_and_task_quota(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch, history_quota=1_000, task_quota=700)
    first = storage.ModelTestStorageEstimate(600, 600, 0, 0, 0, 0)
    second = storage.ModelTestStorageEstimate(250, 250, 0, 0, 0, 0)

    storage.prepare_test_history("task-one", "a" * 32, first)
    assert storage.accounted_history_bytes() >= 600
    with pytest.raises(storage.ModelTestCapacityError, match="当前训练任务") as task_error:
        storage.prepare_test_history("task-one", "b" * 32, second)
    assert task_error.value.code == "task_quota_exceeded"

    with pytest.raises(storage.ModelTestCapacityError, match="总配额") as global_error:
        storage.prepare_test_history("task-two", "c" * 32, storage.ModelTestStorageEstimate(500, 500, 0, 0, 0, 0))
    assert global_error.value.code == "history_quota_exceeded"


def test_low_free_space_includes_reservation_and_minimum(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch, min_free=100)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=149))

    with pytest.raises(storage.ModelTestCapacityError, match="可用") as error:
        storage.prepare_test_history("task-one", "a" * 32, storage.ModelTestStorageEstimate(50, 50, 0, 0, 0, 0))
    assert error.value.code == "low_disk_space"


def test_low_free_space_counts_other_pending_tests(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch, min_free=100)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=449))
    storage.prepare_test_history(
        "task-one", "a" * 32, storage.ModelTestStorageEstimate(100, 300, 0, 0, 0, 0),
    )

    with pytest.raises(storage.ModelTestCapacityError) as error:
        storage.prepare_test_history(
            "task-two", "b" * 32, storage.ModelTestStorageEstimate(50, 100, 0, 0, 0, 0),
        )
    assert error.value.code == "low_disk_space"


def test_actual_files_override_reservation_and_terminal_release(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch)
    directory = storage.prepare_test_history(
        "task-one", "a" * 32, storage.ModelTestStorageEstimate(100, 100, 0, 0, 0, 0),
    )
    (directory / "results.bin").write_bytes(b"x" * 250)
    assert storage.accounted_history_bytes("task-one") >= 250

    storage.release_test_reservation("task-one", "a" * 32)
    assert not storage.reservation_path("task-one", "a" * 32).exists()
    assert storage.accounted_history_bytes("task-one") >= 250


def test_malformed_reservation_never_reduces_real_usage(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch)
    directory = storage.test_history_root("task-one", "a" * 32)
    directory.mkdir(parents=True)
    (directory / ".storage-reservation.json").write_text("not-json", encoding="utf-8")
    (directory / "data.bin").write_bytes(b"x" * 42)

    assert storage.accounted_history_bytes("task-one") >= 42


def test_reconcile_keeps_active_and_releases_terminal_or_orphan_reservations(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch)
    estimate = storage.ModelTestStorageEstimate(100, 100, 0, 0, 0, 0)
    storage.prepare_test_history("task-one", "a" * 32, estimate)
    storage.prepare_test_history("task-one", "b" * 32, estimate)

    storage.reconcile_test_reservations({("task-one", "a" * 32)})

    assert storage.reservation_path("task-one", "a" * 32).is_file()
    assert not storage.reservation_path("task-one", "b" * 32).exists()


def test_prepare_never_deletes_an_existing_test_directory(tmp_path, monkeypatch):
    configure_storage(tmp_path, monkeypatch)
    directory = storage.test_history_root("task-one", "a" * 32)
    directory.mkdir(parents=True)
    sentinel = directory / "result.jpg"
    sentinel.write_bytes(b"existing-history")

    with pytest.raises(storage.ModelTestCapacityError) as error:
        storage.prepare_test_history(
            "task-one", "a" * 32, storage.ModelTestStorageEstimate(100, 100, 0, 0, 0, 0),
        )

    assert error.value.code == "reservation_write_failed"
    assert sentinel.read_bytes() == b"existing-history"
