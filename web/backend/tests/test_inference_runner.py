from types import SimpleNamespace

import pytest

from app.inference_runner import ensure_detection_model


def test_rejects_non_detection_ultralytics_model():
    with pytest.raises(RuntimeError, match="仅支持 YOLO 目标检测"):
        ensure_detection_model(SimpleNamespace(task="classify"))


def test_accepts_detection_ultralytics_model():
    ensure_detection_model(SimpleNamespace(task="detect"))
