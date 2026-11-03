"""Isolated CPU inference, with optional single-model reuse over a private stdin pipe."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import logging
import os
from queue import Queue
import threading
import sys
import time
import traceback
from types import SimpleNamespace
from pathlib import Path

SUPPORTED_TASKS = frozenset({"detect", "segment", "classify", "pose", "obb"})
MAX_MASK_POLYGON_POINTS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--source")
    parser.add_argument("--output")
    parser.add_argument("--summary")
    parser.add_argument("--ultralytics-root", required=True)
    parser.add_argument("--confidence", default=0.25, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--expected-task", choices=sorted(SUPPORTED_TASKS))
    args = parser.parse_args()
    if not args.resident and not all((args.model, args.source, args.output, args.summary)):
        parser.error("单次推理必须提供 model、source、output 和 summary")
    return args


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


def run_job(args, model=None, *, on_loaded=None) -> object:
    """Run a batch and keep only the model/predictor, never the batch's images."""
    from PIL import Image
    from ultralytics import YOLO

    # Ultralytics creates its handler on first import; point it at the current
    # batch log again when the same interpreter handles another request.
    for handler in logging.getLogger("ultralytics").handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setStream(sys.stdout)

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    images = sorted(source.glob("*.jpg"), key=lambda path: int(path.stem))
    if not images:
        raise RuntimeError("没有可用于推理的图片")

    started = time.perf_counter()
    model_started = time.perf_counter()
    reused = model is not None
    if model is None:
        model = YOLO(args.model)
    task = ensure_supported_model_task(model, args.expected_task)
    model_load_ms = 0.0 if reused else (time.perf_counter() - model_started) * 1000
    if on_loaded:
        on_loaded()
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
        save=False, save_txt=False, save_crop=False, show=False,
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
        "model_reused": reused,
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
    # Ultralytics retains its last dataset, batch and results on the predictor.
    # Reuse the predictor but release images/tensors between submissions.
    predictor = getattr(model, "predictor", None)
    if predictor is not None:
        predictor.results = None
        predictor.batch = None
        predictor.dataset = None
    return model


def write_message(path: str, payload: dict) -> None:
    target = Path(path)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def serve(stream, *, exit_on_disconnect: bool = False) -> None:
    lines = stream
    if exit_on_disconnect:
        messages = Queue(maxsize=1)

        def read_requests():
            try:
                for line in stream:
                    messages.put(line)
            finally:
                # The pipe is private to the owning backend. EOF also stops a
                # busy predictor, preventing orphan memory and late file writes
                # after backend recovery has requeued the same batch.
                os._exit(0)

        threading.Thread(target=read_requests, name="inference-owner", daemon=True).start()
        lines = iter(messages.get, None)
    model = None
    model_key = None
    for line in lines:
        request = json.loads(line)
        request_id = request["request_id"]
        key = request["model_key"]
        # The parent replaces the process for a different weight version. Refuse
        # cross-version reuse even if a caller accidentally violates that rule.
        if model_key is not None and key != model_key:
            raise RuntimeError("驻留模型版本不匹配")
        try:
            with open(request["log"], "w", encoding="utf-8") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    try:
                        model = run_job(
                            SimpleNamespace(**request), model,
                            on_loaded=lambda: write_message(request["progress"], {
                                "request_id": request_id, "stage": "running",
                            }),
                        )
                        model_key = key
                    except Exception:
                        traceback.print_exc()
                        raise
                    finally:
                        for handler in logging.getLogger("ultralytics").handlers:
                            if isinstance(handler, logging.StreamHandler):
                                handler.setStream(sys.__stdout__)
            write_message(request["response"], {"request_id": request_id, "ok": True})
        except Exception:
            write_message(request["response"], {"request_id": request_id, "ok": False})
            # A partially failed predictor is never reused.
            return
    # EOF means the owning backend has exited. Do not leave an orphan cache.


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(args.ultralytics_root).resolve()))
    if args.resident:
        serve(sys.stdin, exit_on_disconnect=True)
    else:
        run_job(args)


if __name__ == "__main__":
    main()
