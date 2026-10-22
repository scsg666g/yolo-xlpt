"""Run one isolated Ultralytics image inference job and emit structured results."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SUPPORTED_TASKS = frozenset({"detect", "segment", "classify", "pose", "obb"})
MAX_MASK_POLYGON_POINTS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--ultralytics-root", required=True)
    parser.add_argument("--confidence", required=True, type=float)
    parser.add_argument("--iou", required=True, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--expected-task", choices=sorted(SUPPORTED_TASKS))
    return parser.parse_args()


def ensure_supported_model_task(model, expected_task: str | None = None) -> str:
    """Validate the task declared by an Ultralytics checkpoint."""
    task = getattr(model, "task", None)
    if task not in SUPPORTED_TASKS:
        supported = " / ".join(sorted(SUPPORTED_TASKS))
        raise RuntimeError(f"目前支持 YOLO 任务 {supported}，该权重声明的任务类型为 {task or '未知'}")
    if expected_task and task != expected_task:
        raise RuntimeError(
            f"训练任务类型为 {expected_task}，但权重声明为 {task}；请检查所选的模型权重"
        )
    return task


def _values(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _class_names(names) -> dict[int, str]:
    entries = names.items() if isinstance(names, dict) else enumerate(names or [])
    return {int(key): str(value) for key, value in entries}


def _bounded_polygon(points, maximum: int = MAX_MASK_POLYGON_POINTS) -> list[list[float]]:
    values = _values(points)
    if not values:
        return []
    if len(values) > maximum:
        indexes = [round(index * (len(values) - 1) / (maximum - 1)) for index in range(maximum)]
        values = [values[index] for index in indexes]
    return [[round(float(coordinate), 2) for coordinate in point[:2]] for point in values]


def extract_predictions(
    result,
    task: str,
    names: dict[int, str],
    confidence_threshold: float = 0.0,
) -> list[dict]:
    """Convert Ultralytics results into a task-tagged, JSON-safe structure."""
    if task == "classify":
        probabilities = getattr(result, "probs", None)
        if probabilities is None:
            return []
        class_ids = _values(getattr(probabilities, "top5", None))
        scores = _values(getattr(probabilities, "top5conf", None))
        raw = _values(getattr(probabilities, "data", None))
        if not class_ids:
            class_ids = sorted(range(len(raw)), key=lambda index: raw[index], reverse=True)[:5]
        if len(scores) < len(class_ids):
            scores = [raw[class_id] if class_id < len(raw) else 0 for class_id in class_ids]
        return [
            {
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), str(class_id)),
                "confidence": round(float(scores[index]), 6),
            }
            for index, class_id in enumerate(class_ids)
            if index < len(scores) and float(scores[index]) >= confidence_threshold
        ]

    container = getattr(result, "obb", None) if task == "obb" else getattr(result, "boxes", None)
    if container is None:
        return []
    classes = _values(getattr(container, "cls", None))
    confidences = _values(getattr(container, "conf", None))
    boxes = _values(getattr(container, "xyxy", None))
    predictions = []
    if task == "obb":
        corners = _values(getattr(container, "xyxyxyxy", None))
        xywhr = _values(getattr(container, "xywhr", None))
    elif task == "pose":
        keypoints = getattr(result, "keypoints", None)
        keypoint_rows = _values(getattr(keypoints, "data", None))
    elif task == "segment":
        masks = getattr(result, "masks", None)
        mask_data = getattr(masks, "data", None)
        mask_polygons = getattr(masks, "xy", None) if masks is not None else None
        mask_count = (
            int(mask_data.shape[0])
            if getattr(mask_data, "shape", None) is not None
            else len(mask_polygons) if mask_polygons is not None else 0
        )

    for index, (class_id, confidence) in enumerate(zip(classes, confidences)):
        item = {
            "class_id": int(class_id),
            "class_name": names.get(int(class_id), str(class_id)),
            "confidence": round(float(confidence), 6),
        }
        if index < len(boxes):
            item["bbox_xyxy"] = [round(float(value), 2) for value in boxes[index]]
        if task == "segment":
            item["mask_available"] = index < mask_count
            if mask_polygons is not None and index < len(mask_polygons):
                polygon = _bounded_polygon(mask_polygons[index])
                if polygon:
                    item["mask_polygon_xy"] = polygon
        elif task == "pose" and index < len(keypoint_rows):
            item["keypoints"] = [
                {
                    "index": point_index,
                    "x": round(float(point[0]), 2),
                    "y": round(float(point[1]), 2),
                    "confidence": round(float(point[2]), 6) if len(point) > 2 else None,
                }
                for point_index, point in enumerate(keypoint_rows[index])
            ]
        elif task == "obb":
            if index < len(corners):
                item["obb_points"] = [
                    [round(float(coordinate), 2) for coordinate in point]
                    for point in corners[index]
                ]
            if index < len(xywhr):
                item["obb_xywhr"] = [round(float(value), 4) for value in xywhr[index]]
        predictions.append(item)
    return predictions


def plot_predictions(result, task: str, predictions: list[dict]):
    """Keep the classification preview consistent with the filtered JSON results."""
    if task != "classify":
        return result.plot()
    from ultralytics.utils.plotting import Annotator

    annotator = Annotator(result.orig_img.copy(), pil=True, example=result.names)
    label = "\n".join(f"{item['class_name']} {item['confidence']:.3f}" for item in predictions)
    position = max(1, round(min(result.orig_shape) * 0.03))
    annotator.text(
        [position, position], label or "No class above confidence threshold",
        txt_color=(255, 255, 255), box_color=(64, 64, 64, 128),
    )
    return annotator.result()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.ultralytics_root).resolve()))

    from PIL import Image
    from ultralytics import YOLO

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    images = sorted(source.glob("*.jpg"), key=lambda path: int(path.stem))
    if not images:
        raise RuntimeError("没有可用于推理的图片")

    started = time.perf_counter()
    model_started = time.perf_counter()
    model = YOLO(args.model)
    task = ensure_supported_model_task(model, args.expected_task)
    model_load_ms = (time.perf_counter() - model_started) * 1000
    names = _class_names(model.names)
    results = model.predict(
        source=[str(path) for path in images],
        conf=args.confidence,
        iou=args.iou,
        imgsz=args.imgsz,
        device="cpu",
        batch=1,
        stream=True,
        verbose=False,
    )

    items = []
    for index, result in enumerate(results):
        if index >= len(images):
            raise RuntimeError("推理结果数量超出输入图片数量")
        objects = extract_predictions(result, task, names, args.confidence)
        plotted = plot_predictions(result, task, objects)
        Image.fromarray(plotted[..., ::-1].copy()).save(output / f"{index}.jpg", "JPEG", quality=95)
        speed = {key: round(float(value), 3) for key, value in result.speed.items()}
        items.append({"index": index, "prediction_count": len(objects), "objects": objects, "timings": speed})

    if len(items) != len(images):
        raise RuntimeError("推理没有生成完整结果")
    timings = {
        "model_load": round(model_load_ms, 3),
        "preprocess": round(sum(item["timings"].get("preprocess", 0) for item in items), 3),
        "inference": round(sum(item["timings"].get("inference", 0) for item in items), 3),
        "postprocess": round(sum(item["timings"].get("postprocess", 0) for item in items), 3),
        "total": round((time.perf_counter() - started) * 1000, 3),
    }
    payload = {
        "task_type": task,
        "class_names": names,
        "images": items,
        "timings": timings,
        "device": "CPU",
        "imgsz": args.imgsz,
    }
    Path(args.summary).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
