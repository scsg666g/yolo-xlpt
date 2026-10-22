"""Run one isolated Ultralytics image inference job and emit structured results."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


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
    return parser.parse_args()


def ensure_detection_model(model) -> None:
    """Reject valid Ultralytics checkpoints for unsupported task types."""
    if getattr(model, "task", None) != "detect":
        raise RuntimeError("模型测试目前仅支持 YOLO 目标检测（detect）模型")


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
    ensure_detection_model(model)
    model_load_ms = (time.perf_counter() - model_started) * 1000
    names = {int(key): str(value) for key, value in model.names.items()}
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
        plotted = result.plot()
        Image.fromarray(plotted[..., ::-1].copy()).save(output / f"{index}.jpg", "JPEG", quality=95)

        objects = []
        boxes = result.boxes
        if boxes is not None:
            classes = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()
            for class_id, confidence, bbox in zip(classes, confidences, coordinates):
                objects.append({
                    "class_id": class_id,
                    "class_name": names.get(class_id, str(class_id)),
                    "confidence": round(float(confidence), 6),
                    "bbox_xyxy": [round(float(value), 2) for value in bbox],
                })
        speed = {key: round(float(value), 3) for key, value in result.speed.items()}
        items.append({"index": index, "objects": objects, "timings": speed})

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
        "class_names": names,
        "images": items,
        "timings": timings,
        "device": "CPU",
        "imgsz": args.imgsz,
    }
    Path(args.summary).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
