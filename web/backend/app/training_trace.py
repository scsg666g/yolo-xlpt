"""Collect a small, real training trace for the platform's inspection view."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _value(value: str) -> Any:
    from ultralytics.cfg import smart_value

    return smart_value(value)


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(item) for item in shape] if shape is not None else None


def _tensor_stats(value: Any) -> dict[str, Any] | None:
    if not hasattr(value, "detach"):
        return None
    tensor = value.detach().float()
    if tensor.numel() == 0:
        return {"shape": _shape(tensor), "numel": 0}
    return {
        "shape": _shape(tensor),
        "numel": int(tensor.numel()),
        "mean": round(float(tensor.mean().cpu()), 7),
        "std": round(float(tensor.std().cpu()), 7) if tensor.numel() > 1 else 0.0,
        "min": round(float(tensor.min().cpu()), 7),
        "max": round(float(tensor.max().cpu()), 7),
    }


class TraceCollector:
    def __init__(self, directory: Path, maximum: int = 24, interval: int = 10) -> None:
        self.directory = directory
        self.maximum = maximum
        self.interval = max(1, interval)
        self.samples: list[dict[str, Any]] = []
        self.step = -1
        self.layer_stats: dict[str, dict[str, Any]] = {}
        self.before_update: dict[str, dict[str, float]] = {}
        self.update_stats: dict[str, dict[str, float]] = {}
        self.input_stats: dict[str, Any] | None = None
        self.last_batch: dict[str, Any] = {}
        self.hooks: list[Any] = []
        self.optimizer_hooks: list[Any] = []
        self.started_at = time.time()
        self.disabled = False
        self.error: str | None = None

    @staticmethod
    def _is_primary_rank() -> bool:
        """Only the main DDP rank may collect and write the shared trace."""
        return os.getenv("RANK", "-1") in {"-1", "0"}

    def _disable(self, exc: Exception) -> None:
        self.disabled = True
        self.error = f"{type(exc).__name__}: {exc}"
        print(f"[training-trace] collection disabled: {self.error}", file=sys.stderr, flush=True)

    def _write(self) -> None:
        payload = {
            "version": 1,
            "available": bool(self.samples),
            "source": "real training callback",
            "updated_at": time.time(),
            "sample_count": len(self.samples),
            "maximum": self.maximum,
            "interval": self.interval,
            "samples": self.samples,
        }
        temporary = self.directory / f"trace.json.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "trace.json")

    def _hook(self, name: str):
        def capture(_module, _inputs, output):
            if self.disabled or not self._is_primary_rank() or self.step < 0 or self.step % self.interval or len(self.samples) >= self.maximum:
                return
            try:
                value = output
                if isinstance(value, (list, tuple)):
                    value = next((item for item in value if hasattr(item, "detach")), None)
                stats = _tensor_stats(value)
                if stats:
                    self.layer_stats[name] = stats
            except Exception as exc:
                self._disable(exc)

        return capture

    def on_train_start(self, trainer) -> None:
        if not self._is_primary_rank():
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            original_preprocess = trainer.preprocess_batch

            def capture_batch(batch):
                self.last_batch = original_preprocess(batch)
                return self.last_batch

            trainer.preprocess_batch = capture_batch
            # Keep a representative layer trace instead of retaining full feature maps.
            for index, (name, module) in enumerate(trainer.model.named_modules()):
                if index > 40 or not name or list(module.children()):
                    continue
                self.hooks.append(module.register_forward_hook(self._hook(name)))

            optimizer = getattr(trainer, "optimizer", None)
            if optimizer is not None and hasattr(optimizer, "register_step_pre_hook"):
                self.optimizer_hooks.append(optimizer.register_step_pre_hook(self._guarded_before_optimizer))
                self.optimizer_hooks.append(optimizer.register_step_post_hook(self._guarded_after_optimizer))
        except Exception as exc:
            self._disable(exc)

    def _guarded_before_optimizer(self, optimizer, args, kwargs) -> None:
        try:
            self._before_optimizer(optimizer, args, kwargs)
        except Exception as exc:
            self._disable(exc)

    def _guarded_after_optimizer(self, optimizer, args, kwargs) -> None:
        try:
            self._after_optimizer(optimizer, args, kwargs)
        except Exception as exc:
            self._disable(exc)

    def _before_optimizer(self, optimizer, _args, _kwargs) -> None:
        if self.disabled or not self._is_primary_rank():
            return
        self.before_update = {}
        for index, parameter in enumerate(optimizer.param_groups[0].get("params", [])):
            if parameter.grad is None or index >= 12:
                continue
            gradient = parameter.grad.detach().float()
            values = parameter.detach().float()
            self.before_update[str(index)] = {
                "gradient_mean": round(float(gradient.abs().mean().cpu()), 8),
                "gradient_max": round(float(gradient.abs().max().cpu()), 8),
                "weight_mean_before": round(float(values.mean().cpu()), 8),
            }

    def _after_optimizer(self, _optimizer, _args, _kwargs) -> None:
        if self.disabled or not self._is_primary_rank():
            return
        self.update_stats = {}
        for index, parameter in enumerate(_optimizer.param_groups[0].get("params", [])):
            before = self.before_update.get(str(index))
            if before is None or index >= 12:
                continue
            values = parameter.detach().float()
            self.update_stats[str(index)] = {
                **before,
                "weight_mean_after": round(float(values.mean().cpu()), 8),
                "weight_delta_mean": round(abs(float(values.mean().cpu()) - before["weight_mean_before"]), 8),
            }

    def on_train_batch_start(self, _trainer) -> None:
        if self.disabled or not self._is_primary_rank():
            return
        self.step += 1

    def on_train_batch_end(self, trainer) -> None:
        if self.disabled or not self._is_primary_rank() or self.step % self.interval or len(self.samples) >= self.maximum:
            return
        try:
            batch = self.last_batch
            image = batch.get("img") if isinstance(batch, dict) else None
            self.input_stats = _tensor_stats(image)
            sample: dict[str, Any] = {
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "batch": self.step,
                "images": self.input_stats,
                "labels": int(batch.get("cls").shape[0]) if isinstance(batch, dict) and hasattr(batch.get("cls"), "shape") else None,
                "loss": {key: round(float(value), 7) for key, value in (getattr(trainer, "loss_items", {}) or {}).items()},
                "learning_rate": [round(float(group.get("lr", 0.0)), 10) for group in getattr(trainer, "optimizer", None).param_groups] if getattr(trainer, "optimizer", None) else [],
                "layers": dict(self.layer_stats),
                "updates": dict(self.update_stats),
            }
            if hasattr(image, "detach") and getattr(image, "ndim", 0) == 4:
                try:
                    from PIL import Image

                    pixels = image[0].detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
                    file_name = f"batch-{self.step:05d}.jpg"
                    Image.fromarray(pixels).save(self.directory / file_name, quality=88)
                    sample["image"] = file_name
                except (OSError, ValueError, RuntimeError):
                    pass
            self.samples.append(sample)
            self.input_stats = None
            self._write()
        except Exception as exc:
            # Observability must never terminate the actual training process.
            self._disable(exc)

    def on_train_end(self, _trainer) -> None:
        for hook in [*self.hooks, *self.optimizer_hooks]:
            try:
                hook.remove()
            except (AttributeError, RuntimeError):
                pass
        if self._is_primary_rank() and not self.disabled:
            try:
                self._write()
            except Exception as exc:
                self._disable(exc)


def run(command: list[str], trace_dir: Path, maximum: int, interval: int) -> None:
    # The runner is launched with the Ultralytics checkout as cwd, but an absolute
    # script path is used by the platform. Make the checkout importable explicitly.
    sys.path.insert(0, str(Path.cwd()))
    from ultralytics import YOLO

    values: dict[str, Any] = {}
    for item in command[3:]:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        values[key] = _value(raw)
    model_path = str(values.pop("model"))
    task_type = str(values.pop("task", command[1]))
    values.pop("mode", None)
    collector = TraceCollector(trace_dir, maximum=maximum, interval=interval)
    model = YOLO(model_path, task=task_type)
    model.add_callback("on_train_start", collector.on_train_start)
    model.add_callback("on_train_batch_start", collector.on_train_batch_start)
    model.add_callback("on_train_batch_end", collector.on_train_batch_end)
    model.add_callback("on_train_end", collector.on_train_end)
    model.train(**values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=24)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--command-json", required=True)
    args, _ = parser.parse_known_args()
    run(json.loads(args.command_json), args.trace_dir, args.maximum, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
