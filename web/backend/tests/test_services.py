from pathlib import Path

import yaml
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Artifact, TrainingTask
from app.services import TrainingScheduler, failure_detail, path_is_within, prepare_dataset_config, resume_checkpoint, sync_artifacts, training_environment


def test_path_is_within(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    assert path_is_within(allowed / "model.pt", [allowed])
    assert not path_is_within(tmp_path / "outside.pt", [allowed])


def test_prepare_dataset_config_makes_dataset_root_absolute(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    source = dataset / "data.yaml"
    source.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")

    monkeypatch.setattr("app.services.DATASET_ROOTS", [tmp_path])
    destination = tmp_path / "runtime" / "dataset.yaml"
    prepare_dataset_config(str(source), destination)

    generated = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert generated["path"] == str(dataset.resolve())
    assert destination.parent.name == "runtime"


def test_training_environment_uses_gnu_mkl_threading_layer():
    assert training_environment()["MKL_THREADING_LAYER"] == "GNU"


def test_failure_detail_prefers_root_error(tmp_path: Path):
    log = tmp_path / "platform.log"
    log.write_text(
        "Error: mkl-service is incompatible with libgomp.so.1\n"
        "subprocess.CalledProcessError: worker failed\n",
        encoding="utf-8",
    )
    assert failure_detail(str(log)) == "Error: mkl-service is incompatible with libgomp.so.1"


def test_sync_artifacts_is_idempotent(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text("epoch,metrics\n", encoding="utf-8")
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    task = TrainingTask(
        id="task-idempotent",
        name="test",
        task_type="detect",
        model_path="model.pt",
        data_path="data.yaml",
        run_dir=str(run_dir),
    )
    session.add(task)
    session.commit()

    sync_artifacts(session, task)
    session.commit()
    sync_artifacts(session, task)
    session.commit()

    paths = session.scalars(select(Artifact.path).where(Artifact.task_id == task.id)).all()
    assert paths == [str((run_dir / "results.csv").resolve())]
    session.close()


def test_resume_checkpoint_and_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    task = TrainingTask(
        id="task-resume",
        name="test",
        task_type="detect",
        model_path="model.pt",
        data_path="data.yaml",
        run_dir=str(run_dir),
        config={},
    )
    assert resume_checkpoint(task) == checkpoint
    command = TrainingScheduler()._command(task, Path("generated.yaml"), checkpoint)
    assert f"resume={checkpoint}" in command
