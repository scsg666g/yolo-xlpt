from pathlib import Path

import yaml

from app.services import failure_detail, path_is_within, prepare_dataset_config, training_environment


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

