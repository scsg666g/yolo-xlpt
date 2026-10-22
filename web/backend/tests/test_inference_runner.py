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
