from types import SimpleNamespace

import pytest

from app.inference_runner import extract_predictions, ensure_supported_model_task


@pytest.mark.parametrize("task", ["detect", "segment", "classify", "pose", "obb"])
def test_accepts_supported_ultralytics_tasks(task):
    assert ensure_supported_model_task(SimpleNamespace(task=task)) == task


def test_rejects_unknown_and_mismatched_task():
    with pytest.raises(RuntimeError, match="目前支持 YOLO 任务"):
        ensure_supported_model_task(SimpleNamespace(task="track"))
    with pytest.raises(RuntimeError, match="训练任务类型为 segment"):
        ensure_supported_model_task(SimpleNamespace(task="detect"), expected_task="segment")


def test_extracts_classification_top_five_probabilities():
    result = SimpleNamespace(probs=SimpleNamespace(top5=[1, 0], top5conf=[0.8, 0.2]))

    predictions = extract_predictions(result, "classify", {0: "cat", 1: "dog"})

    assert predictions == [
        {"class_id": 1, "class_name": "dog", "confidence": 0.8},
        {"class_id": 0, "class_name": "cat", "confidence": 0.2},
    ]

    assert extract_predictions(result, "classify", {0: "cat", 1: "dog"}, 0.5) == [predictions[0]]
    assert extract_predictions(result, "classify", {0: "cat", 1: "dog"}, 0.9) == []


def test_extracts_detection_boxes_and_segment_presence():
    boxes = SimpleNamespace(cls=[0], conf=[0.9], xyxy=[[1, 2, 3, 4]])
    points = [[index, index + 1] for index in range(130)]
    result = SimpleNamespace(
        boxes=boxes,
        masks=SimpleNamespace(data=SimpleNamespace(shape=(1, 10, 10)), xy=[points]),
    )

    detected = extract_predictions(result, "detect", {0: "person"})
    segmented = extract_predictions(result, "segment", {0: "person"})

    assert detected[0]["bbox_xyxy"] == [1, 2, 3, 4]
    assert "mask_available" not in detected[0]
    assert segmented[0]["mask_available"] is True
    assert len(segmented[0]["mask_polygon_xy"]) == 128
    assert segmented[0]["mask_polygon_xy"][0] == [0, 1]
    assert segmented[0]["mask_polygon_xy"][-1] == [129, 130]


def test_extracts_pose_keypoints_and_obb_corners():
    box_values = {"cls": [0], "conf": [0.95], "xyxy": [[1, 2, 10, 20]]}
    pose_result = SimpleNamespace(
        boxes=SimpleNamespace(**box_values),
        keypoints=SimpleNamespace(data=[[[4, 5, 0.75], [8, 9, 0.5]]]),
    )
    obb_result = SimpleNamespace(obb=SimpleNamespace(
        **box_values,
        xyxyxyxy=[[[1, 2], [3, 2], [3, 4], [1, 4]]],
        xywhr=[[2, 3, 4, 5, 0.2]],
    ))

    pose = extract_predictions(pose_result, "pose", {0: "person"})[0]
    obb = extract_predictions(obb_result, "obb", {0: "vehicle"})[0]

    assert pose["keypoints"] == [
        {"index": 0, "x": 4.0, "y": 5.0, "confidence": 0.75},
        {"index": 1, "x": 8.0, "y": 9.0, "confidence": 0.5},
    ]
    assert obb["obb_points"] == [[1, 2], [3, 2], [3, 4], [1, 4]]
    assert obb["obb_xywhr"] == [2, 3, 4, 5, 0.2]


def test_warm_batches_reuse_model_reset_parameters_and_drop_image_references(tmp_path, monkeypatch):
    import json
    import sys
    from PIL import Image
    from app.inference_runner import run_job

    calls = []
    loads = []

    class Pixels:
        __array_interface__ = {"shape": (4, 4, 3), "typestr": "|u1", "strides": (12, 3, 1)}

        def __getitem__(self, key): return self
        def copy(self): return self
        def tobytes(self): return bytes(4 * 4 * 3)

    class Model:
        task = "detect"
        names = {0: "test"}

        def __init__(self, path):
            loads.append(path)
            self.predictor = SimpleNamespace(results="old", batch="old", dataset="old")

        def predict(self, **kwargs):
            calls.append(kwargs)
            yield SimpleNamespace(boxes=None, speed={}, plot=Pixels)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    source = tmp_path / "images"
    source.mkdir()
    Image.new("RGB", (4, 4)).save(source / "0.jpg")
    args = SimpleNamespace(model="pinned.pt", source=source, output=tmp_path / "output", summary=tmp_path / "summary.json",
                           confidence=0.25, iou=0.7, imgsz=640, expected_task="detect")
    model = run_job(args)
    assert json.loads(args.summary.read_text())["timings"]["model_reused"] is False
    args.confidence, args.iou = 0.8, 0.3
    assert run_job(args, model) is model
    summary = json.loads(args.summary.read_text())
    assert summary["timings"]["model_reused"] is True
    assert summary["timings"]["model_load"] == 0
    assert len(loads) == 1
    assert [item["conf"] for item in calls] == [0.25, 0.8]
    assert [item["iou"] for item in calls] == [0.7, 0.3]
    assert model.predictor.results is model.predictor.batch is model.predictor.dataset is None
    args.expected_task = "pose"
    with pytest.raises(RuntimeError, match="训练任务类型为 pose"):
        run_job(args, model)


def test_resident_protocol_stops_after_failed_batch_and_logs_error(tmp_path, monkeypatch):
    import json
    from io import StringIO
    import app.inference_runner as runner

    def fail(*args, **kwargs):
        raise RuntimeError("broken weight")

    monkeypatch.setattr(runner, "run_job", fail)
    request = {"request_id": "test", "model_key": "digest:10", "log": str(tmp_path / "job.log"),
               "response": str(tmp_path / "response.json"), "progress": str(tmp_path / "progress.json")}
    runner.serve(StringIO(json.dumps(request) + "\n" + "invalid next request\n"))
    assert json.loads((tmp_path / "response.json").read_text()) == {"request_id": "test", "ok": False}
    assert "broken weight" in (tmp_path / "job.log").read_text()
    assert not list(tmp_path.glob("*.tmp"))
