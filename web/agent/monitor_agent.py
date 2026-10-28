#!/usr/bin/env python3
"""YOLO Forge compute Agent with opt-in remote training. Python 3.9+."""
from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

VERSION = "2.5.3"
LIGHTWEIGHT_DATASET_PHASES = ('check', 'discover')
DEFAULT_DATASET_REFRESH_READ_RATE = 32 * 1024**2
DEFAULT_DATASET_REFRESH_MIN_FREE = 512 * 1024**2
DEFAULT_DATASET_REFRESH_IO_PRESSURE_PCT = 25
DELETE_RECEIPT_VERSION = 1
TRACE_VERSION = 1
TRACE_MAXIMUM = 24
TRACE_INTERVAL = 10
MODEL_TRANSFER_FINALIZE_TIMEOUT_SECONDS = 1800
LOG_SNAPSHOT_MAX_BYTES = 256 * 1024 * 1024
LOGGER = logging.getLogger("yolo-monitor-agent")
CONFIRM_RETRY_DELAYS = (1, 2, 4, 8)
ACTIVE_STATE_VERSION = 1
WORKSPACE_REGISTRY_VERSION = 2
_YOLO_VERSION_CACHE: dict[tuple[str, int | None], str | None] = {}
REPORT_BOOT_ID = uuid.uuid4().hex
_REPORT_SEQUENCE = 0
_REPORT_SEQUENCE_LOCK = threading.Lock()


REMOTE_TRACE_RUNNER = r'''#!/usr/bin/env python3
import argparse, csv, json, math, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

def stats(value):
    if not hasattr(value, "detach"):
        return None
    value = value.detach().float()
    shape = [int(x) for x in getattr(value, "shape", [])]
    if not value.numel(): return {"shape": shape, "numel": 0}
    return {"shape": shape, "numel": int(value.numel()), "mean": round(float(value.mean().cpu()), 7),
            "std": round(float(value.std().cpu()), 7) if value.numel() > 1 else 0.0,
            "min": round(float(value.min().cpu()), 7), "max": round(float(value.max().cpu()), 7)}

class Collector:
    def __init__(self, root, maximum, interval):
        self.root, self.maximum, self.interval = Path(root), maximum, max(1, interval)
        self.samples, self.step, self.layers, self.before, self.updates = [], -1, {}, {}, {}
        self.last_batch, self.disabled = {}, False
    def primary(self): return os.getenv("RANK", "-1") in {"-1", "0"}
    def write(self):
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "available": bool(self.samples), "source": "remote training callback",
                   "updated_at": time.time(), "sample_count": len(self.samples), "maximum": self.maximum,
                   "interval": self.interval, "samples": self.samples}
        tmp = self.root / ("trace.json.%s.tmp" % os.getpid())
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.root / "trace.json")
    def hook(self, name):
        def capture(_module, _inputs, output):
            if self.disabled or not self.primary() or self.step < 0 or self.step % self.interval or len(self.samples) >= self.maximum: return
            try:
                if isinstance(output, (list, tuple)): output = next((x for x in output if hasattr(x, "detach")), None)
                value = stats(output)
                if value: self.layers[name] = value
            except Exception: self.disabled = True
        return capture
    def start(self, trainer):
        if not self.primary(): return
        try:
            original = trainer.preprocess_batch
            def preprocess(batch):
                self.last_batch = original(batch); return self.last_batch
            trainer.preprocess_batch = preprocess
            for index, (name, module) in enumerate(trainer.model.named_modules()):
                if index > 40 or not name or list(module.children()): continue
                module.register_forward_hook(self.hook(name))
            optimizer = getattr(trainer, "optimizer", None)
            if optimizer is not None and hasattr(optimizer, "register_step_pre_hook"):
                optimizer.register_step_pre_hook(self.before_step); optimizer.register_step_post_hook(self.after_step)
        except Exception: self.disabled = True
    def before_step(self, optimizer, _args, _kwargs):
        self.before = {}
        for index, parameter in enumerate(optimizer.param_groups[0].get("params", [])):
            if index >= 12 or parameter.grad is None: continue
            gradient, value = parameter.grad.detach().float(), parameter.detach().float()
            self.before[str(index)] = {"gradient_mean": round(float(gradient.abs().mean().cpu()), 8), "gradient_max": round(float(gradient.abs().max().cpu()), 8), "weight_mean_before": round(float(value.mean().cpu()), 8)}
    def after_step(self, optimizer, _args, _kwargs):
        self.updates = {}
        for index, parameter in enumerate(optimizer.param_groups[0].get("params", [])):
            before = self.before.get(str(index))
            if index >= 12 or before is None: continue
            after = float(parameter.detach().float().mean().cpu())
            self.updates[str(index)] = {**before, "weight_mean_after": round(after, 8), "weight_delta_mean": round(abs(after - before["weight_mean_before"]), 8)}
    def batch_start(self, _trainer):
        if self.primary() and not self.disabled: self.step += 1
    def batch_end(self, trainer):
        if self.disabled or not self.primary() or self.step % self.interval or len(self.samples) >= self.maximum: return
        try:
            batch = self.last_batch; image = batch.get("img") if isinstance(batch, dict) else None
            sample = {"epoch": int(getattr(trainer, "epoch", 0)) + 1, "batch": self.step,
                      "images": stats(image), "labels": int(batch.get("cls").shape[0]) if isinstance(batch, dict) and hasattr(batch.get("cls"), "shape") else None,
                      "loss": {k: round(float(v), 7) for k, v in (getattr(trainer, "loss_items", {}) or {}).items()},
                      "learning_rate": [round(float(g.get("lr", 0)), 10) for g in getattr(trainer, "optimizer", None).param_groups] if getattr(trainer, "optimizer", None) else [],
                      "layers": dict(self.layers), "updates": dict(self.updates)}
            if hasattr(image, "detach") and getattr(image, "ndim", 0) == 4:
                try:
                    from PIL import Image
                    pixels = image[0].detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
                    name = "batch-%05d.jpg" % self.step
                    Image.fromarray(pixels).save(self.root / name, quality=88); sample["image"] = name
                except Exception: pass
            self.samples.append(sample); self.write()
        except Exception: self.disabled = True
    def end(self, _trainer):
        if self.primary() and not self.disabled:
            try: self.write()
            except Exception: pass

def publish_result(run_dir, model, values, error=None):
    if os.getenv("RANK", "-1") not in {"-1", "0"}: return
    def integer(value):
        try: return max(0, int(value or 0))
        except (TypeError, ValueError, OverflowError): return 0
    trainer = getattr(model, "trainer", None)
    expected = integer(getattr(getattr(trainer, "args", None), "epochs", None) or getattr(trainer, "epochs", None) or values.get("epochs"))
    completed = 0
    try:
        with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                raw = next((value for key, value in row.items() if key and key.strip() == "epoch"), "0")
                completed = max(completed, int(float(raw or 0)))
    except (OSError, ValueError, TypeError, csv.Error): pass
    stopper = getattr(trainer, "stopper", None)
    best = integer(getattr(stopper, "best_epoch", 0))
    patience = integer(getattr(stopper, "patience", 0))
    reason = "exception" if error is not None else "epochs_completed" if expected and completed >= expected else "training_completed"
    if error is None and completed < expected and best > 0 and patience > 0 and completed - best >= patience:
        reason = "early_stopping"
    metrics = getattr(getattr(model, "metrics", None), "results_dict", {}) or {}
    final_metrics = {}
    for key, value in metrics.items():
        try:
            value = float(value)
            if math.isfinite(value): final_metrics[str(key)] = value
        except (ValueError, TypeError): pass
    exit_code = ((int(error.code) % 256 or 1) if isinstance(error, SystemExit) and isinstance(error.code, int)
                 else 130 if isinstance(error, KeyboardInterrupt) else 1 if error is not None else 0)
    payload = {"schema_version": 1, "run_id": os.getenv("YOLO_FORGE_RUN_ID"),
               "task_id": os.getenv("YOLO_FORGE_TASK_ID"), "status": "failed" if error is not None else "completed",
               "exit_code": exit_code, "reason": reason, "completed_epochs": completed,
               "expected_epochs": expected, "finished_at": datetime.now(timezone.utc).isoformat(),
               "error_message": (type(error).__name__ + ": " + str(error))[:4000] if error is not None else None,
               "final_metrics": final_metrics}
    if best > 0: payload["best_epoch"] = best
    if patience > 0: payload["patience"] = patience
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        temporary = run_dir / (".training-result.%s.tmp" % os.getpid())
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
                handle.flush(); os.fsync(handle.fileno())
            temporary.replace(run_dir / "training-result.json")
        finally: temporary.unlink(missing_ok=True)
    except OSError as exc:
        print("[YOLO Forge] 无法保存训练结束记录: %s" % exc, file=sys.stderr, flush=True)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--trace-dir", required=True); parser.add_argument("--run-dir", required=True); parser.add_argument("--trace-enabled", action="store_true"); parser.add_argument("--maximum", type=int, default=24); parser.add_argument("--interval", type=int, default=10); parser.add_argument("--command-json", required=True)
    args, _ = parser.parse_known_args()
    command = json.loads(args.command_json); values = {}
    for item in command[3:]:
        if "=" in item:
            key, value = item.split("=", 1)
            try:
                from ultralytics.cfg import smart_value; value = smart_value(value)
            except Exception: pass
            values[key] = value
    model = None
    try:
        from ultralytics import YOLO
        model = YOLO(str(values.pop("model")), task=str(values.pop("task", command[1])))
        values.pop("mode", None)
        if args.trace_enabled:
            collector = Collector(args.trace_dir, args.maximum, args.interval)
            model.add_callback("on_train_start", collector.start); model.add_callback("on_train_batch_start", collector.batch_start); model.add_callback("on_train_batch_end", collector.batch_end); model.add_callback("on_train_end", collector.end)
        model.train(**values)
    except BaseException as exc:
        publish_result(Path(args.run_dir), model, values, exc)
        if isinstance(exc, KeyboardInterrupt): raise SystemExit(130) from exc
        if isinstance(exc, SystemExit) and (exc.code is None or isinstance(exc.code, int)):
            raise SystemExit((int(exc.code or 0) % 256) or 1) from exc
        raise
    publish_result(Path(args.run_dir), model, values)
if __name__ == "__main__": main()
'''


def number(value: str, integer: bool = False):
    try:
        parsed = float(value.strip())
        if not math.isfinite(parsed) or parsed < 0:
            return None
        return int(parsed) if integer else parsed
    except (ValueError, OverflowError):
        return None


def query(binary: str, arguments: list[str]) -> str:
    return subprocess.check_output([binary, *arguments], text=True, timeout=5, stderr=subprocess.STDOUT)


def collect_gpus(binary: str, config: dict | None = None) -> tuple[list[dict], str | None]:
    fields = ["index", "uuid", "name", "driver_version", "memory.total", "memory.used", "memory.free",
              "utilization.gpu", "utilization.memory", "temperature.gpu", "power.draw", "power.limit",
              "clocks.current.graphics", "clocks.current.memory"]
    try:
        output = query(binary, ["--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError):
        # Older drivers may not expose clock/power fields. Keep the other metrics.
        fields = fields[:10]
        try:
            output = query(binary, ["--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
        except (OSError, subprocess.SubprocessError):
            return [], "未检测到 NVIDIA GPU，或 nvidia-smi 不可用；请检查驱动与执行权限"
    processes: dict[str, list[dict]] = {}
    message = None
    try:
        rows = query(binary, ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
        for row in csv.reader(rows.splitlines()):
            if len(row) != 4:
                continue
            uuid, pid, name, memory = (value.strip() for value in row)
            parsed_pid = number(pid, True)
            if parsed_pid:
                processes.setdefault(uuid, []).append({"pid": parsed_pid, "name": Path(name).name[:255], "memory_used": number(memory, True)})
    except (OSError, subprocess.SubprocessError):
        message = "GPU 指标已采集，计算进程查询不可用"
    active_roots = active_attribution_roots(config) if config else {}
    attribution_cache: dict[int, dict] = {}
    for processes_for_gpu in processes.values():
        for process in processes_for_gpu:
            pid = process["pid"]
            if pid not in attribution_cache:
                attribution_cache[pid] = process_owner_hint(pid, active_roots)
            process["owner_hint"] = attribution_cache[pid]
    numeric = {"memory_total": 4, "memory_used": 5, "memory_free": 6, "utilization": 7,
               "memory_utilization": 8, "temperature": 9, "power_draw": 10, "power_limit": 11,
               "graphics_clock": 12, "memory_clock": 13}
    result = []
    for row in csv.reader(output.splitlines()):
        if len(row) != len(fields):
            continue
        row = [value.strip() for value in row]
        index = number(row[0], True)
        if index is None:
            continue
        gpu = {"index": index, "uuid": row[1], "name": row[2], "driver_version": row[3],
               "processes": processes.get(row[1], [])[:512]}
        gpu.update({key: number(row[position]) if position < len(row) else None for key, position in numeric.items()})
        result.append(gpu)
    return result[:128], message


def _probe_writable_directory(path: Path) -> None:
    """Verify a directory with an exclusive private file without clobbering user data."""
    name = f".platform-write-check-{os.getpid()}-{threading.get_ident()}"
    probe = path / name
    descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b"ok")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
        probe.unlink(missing_ok=True)


def _configured_yolo_version(binary: str) -> str | None:
    """Return a cached CLI version; inability to print a version does not disable training."""
    resolved = shutil.which(binary) if "/" not in binary else str(Path(binary))
    if not resolved:
        return None
    try:
        modified = Path(resolved).stat().st_mtime_ns
    except OSError:
        modified = None
    key = (resolved, modified)
    if key in _YOLO_VERSION_CACHE:
        return _YOLO_VERSION_CACHE[key]
    try:
        output = subprocess.check_output([resolved, "version"], text=True, timeout=5,
                                         stderr=subprocess.STDOUT).strip()
        clean = re.sub(r"\x1b\[[0-9;]*m", "", output)
        match = re.search(r"(?<!\d)(\d+\.\d+(?:\.\d+)?(?:[-+._a-zA-Z0-9]*)?)", clean)
        version = match.group(1)[:100] if match else clean.splitlines()[-1][:100] if clean else None
    except (OSError, subprocess.SubprocessError):
        version = None
    _YOLO_VERSION_CACHE[key] = version
    return version


def training_lock_occupied(config: dict) -> bool:
    """Inspect both the legacy configured lock and output-root intention locks."""
    lock_path = Path(config.get("training_lock_file") or Path(config["training_cwd"]) / ".yolo-forge-training.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
    root_locks = acquire_output_root_locks(config)
    if root_locks is None:
        return True
    release_training_lock(root_locks)
    return False


def training_capability(config: dict) -> dict:
    """Report whether this Agent can safely accept a remote training run."""
    enabled = bool(config.get("enable_training"))
    if not enabled:
        return {"enabled": False, "ready": False, "reason": "远程训练未启用",
                "supported_parameters": sorted(TRAINING_KEYS), "yolo_version": None,
                "python_version": platform.python_version(), "output_writable": False,
                "lock_occupied": False, "output_roots": [], "gpu_parallelism": True,
                "model_test_roots": list(config.get("model_test_roots", [])),
                "dataset_roots": list(config.get("dataset_roots", [])), "dataset_discovery": True, "dataset_progress": True,
                "dataset_concurrent_reads": True, "dataset_concurrent_refresh": True}
    reported_roots = []
    try:
        reported_roots = [str(root) for root in configured_output_roots(config)]
        cwd = Path(config["training_cwd"])
        roots = [Path(item) for item in config["allowed_training_roots"]]
        if not cwd.is_dir():
            raise ValueError("训练工作目录不存在")
        if not roots:
            raise ValueError("未配置允许的训练目录")
        missing = [str(root) for root in roots if not root.is_dir()]
        if missing:
            raise ValueError(f"允许的训练目录不存在：{missing[0]}")
        output_roots = [Path(item) for item in config.get("training_output_roots", [])] or roots
        output_writable = False
        # Legacy configs do not identify output roots, so inspect their allowed
        # roots from last to first. Every task root is checked again at launch.
        for root in reversed(output_roots):
            try:
                _probe_writable_directory(root)
                output_writable = True
                break
            except OSError:
                continue
        if not output_writable:
            raise ValueError("允许的训练目录均不可写")
        yolo = config["yolo_bin"]
        resolved_yolo = shutil.which(yolo) if "/" not in yolo else yolo
        if not resolved_yolo or not Path(resolved_yolo).is_file() or not os.access(resolved_yolo, os.X_OK):
            raise ValueError("YOLO 可执行文件不存在或不可执行")
        state_dir = Path(config["training_state_dir"])
        state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        _probe_writable_directory(state_dir)
        occupied = training_lock_occupied(config)
        return {"enabled": True, "ready": True, "reason": None,
                "supported_parameters": sorted(TRAINING_KEYS), "yolo_version": _configured_yolo_version(yolo),
                "python_version": platform.python_version(), "output_writable": True,
                "lock_occupied": occupied,
                "output_roots": reported_roots, "gpu_parallelism": True,
                "model_test_roots": list(config.get("model_test_roots", [])),
                "dataset_roots": list(config.get("dataset_roots", [])), "dataset_discovery": True, "dataset_progress": True,
                "dataset_concurrent_reads": True, "dataset_concurrent_refresh": True}
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {"enabled": True, "ready": False, "reason": str(exc)[:400],
                "supported_parameters": sorted(TRAINING_KEYS), "yolo_version": None,
                "python_version": platform.python_version(), "output_writable": False,
                "lock_occupied": False,
                "output_roots": reported_roots, "gpu_parallelism": True,
                "model_test_roots": list(config.get("model_test_roots", [])),
                "dataset_roots": list(config.get("dataset_roots", [])), "dataset_discovery": True, "dataset_progress": True,
                "dataset_concurrent_reads": True, "dataset_concurrent_refresh": True}


def next_report_sequence() -> int:
    global _REPORT_SEQUENCE
    with _REPORT_SEQUENCE_LOCK:
        _REPORT_SEQUENCE += 1
        return _REPORT_SEQUENCE


def collect(binary: str = "nvidia-smi", config: dict | None = None) -> dict:
    memory: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            memory[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        memory = {}
    total, available = memory.get("MemTotal"), memory.get("MemAvailable")
    try:
        load_average = [round(value, 2) for value in os.getloadavg()]
    except (AttributeError, OSError):
        load_average = []
    gpus, message = collect_gpus(binary, config)
    payload = {
        "agent_version": VERSION,
        "boot_id": REPORT_BOOT_ID,
        "report_seq": next_report_sequence(),
        "compute": {
            "available": bool(gpus), "provider": "NVIDIA" if gpus else None,
            "message": message, "sampled_at": datetime.now(timezone.utc).isoformat(),
            "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
                     "load_average": load_average, "memory_total": total,
                     "memory_used": total - available if total is not None and available is not None else None},
            "gpus": gpus,
        },
    }
    if config is not None:
        payload["training"] = training_capability(config)
    return payload


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the bearer token to a redirect destination.
        return None


def config_path(value, base: Path, label: str) -> Path:
    """Resolve deployment paths against an explicit base, never the shell cwd."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} 必须是非空路径")
    candidate = Path(value)
    try:
        return (candidate if candidate.is_absolute() else base / candidate).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} 路径无法解析：{value}") from exc


def normalized_training_paths(config: dict, path: Path) -> dict:
    """Share path semantics with live-Agent inspection so relative roots stay protected."""
    cwd = config_path(config.get("training_cwd", "."), path.resolve().parent, "training_cwd")
    result = {"training_cwd": str(cwd)}
    for key in ("allowed_training_roots", "training_output_roots"):
        values = config.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"{key} 必须是路径列表")
        result[key] = [str(config_path(value, cwd, key)) for value in values]
    result["training_lock_file"] = str(config_path(
        config.get("training_lock_file", ".yolo-forge-training.lock"), cwd, "training_lock_file"))
    model_roots = config.get("model_test_roots", ["pt"])
    if (not isinstance(model_roots, list) or len(model_roots) > 128 or
            any(not isinstance(value, str) or not value.strip() or "\x00" in value for value in model_roots)):
        raise ValueError("model_test_roots 必须是最多 128 个非空路径组成的列表")
    # Keep these paths lexical so a configured root that is itself a symlink is
    # still detectable when a model is requested.  Missing roots are allowed:
    # they must not make monitoring or remote training unavailable.
    parsed_model_roots = []
    for value in model_roots:
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else cwd / candidate
        try:
            normalized = Path(os.path.abspath(os.fspath(candidate)))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"model_test_roots 路径无法解析：{value}") from exc
        if str(normalized) not in parsed_model_roots:
            parsed_model_roots.append(str(normalized))
    result["model_test_roots"] = parsed_model_roots
    return result


def load_config(path: Path) -> dict:
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("配置包含 Token，请先执行 chmod 600 <配置文件>")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置必须是 JSON 对象")
    center = str(config.get("center_url", "")).rstrip("/")
    parsed = urlsplit(center)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("center_url 必须是有效的 HTTP(S) 平台地址")
    node_id = str(config.get("node_id", ""))
    if len(node_id) != 32 or any(c not in "0123456789abcdef" for c in node_id):
        raise ValueError("node_id 不合法，请使用平台下载的配置")
    token = config.get("token", "")
    if not isinstance(token, str) or not token.startswith("ylp_") or not 20 <= len(token) <= 200 or any(c.isspace() for c in token):
        raise ValueError("Token 不合法，请使用平台下载的配置")
    interval = config.get("report_interval_seconds", 5)
    if type(interval) is not int or not 5 <= interval <= 300:
        raise ValueError("report_interval_seconds 必须为 5～300 的整数")
    stop_timeout = config.get("training_stop_timeout_seconds", 15)
    if type(stop_timeout) is not int or not 1 <= stop_timeout <= 300:
        raise ValueError("training_stop_timeout_seconds 必须为 1～300 的整数")
    model_finalize_timeout = config.get(
        "model_transfer_finalize_timeout_seconds", MODEL_TRANSFER_FINALIZE_TIMEOUT_SECONDS,
    )
    if type(model_finalize_timeout) is not int or not 60 <= model_finalize_timeout <= 7200:
        raise ValueError("model_transfer_finalize_timeout_seconds 必须为 60～7200 的整数")
    binary = config.get("nvidia_smi_bin", "nvidia-smi")
    if not isinstance(binary, str) or not binary:
        raise ValueError("nvidia_smi_bin 必须是命令或可执行文件路径")
    enabled = config.get("enable_training", False)
    if type(enabled) is not bool:
        raise ValueError("enable_training 必须为布尔值")
    paths = normalized_training_paths(config, path)
    roots = paths["allowed_training_roots"]
    if enabled and not roots:
        raise ValueError("启用训练时必须配置 allowed_training_roots")
    output_roots = paths["training_output_roots"]
    resolved_roots = [Path(item).resolve() for item in roots]
    resolved_outputs = [Path(item).resolve() for item in output_roots]
    if any(not any(output == root or root in output.parents for root in resolved_roots)
           for output in resolved_outputs):
        raise ValueError("training_output_roots 必须位于 allowed_training_roots 内；"
                         f"输出目录：{'、'.join(output_roots)}；允许目录：{'、'.join(roots)}")
    trace_enabled = config.get("enable_training_trace", True)
    if type(trace_enabled) is not bool:
        raise ValueError("enable_training_trace 必须为布尔值")
    yolo_bin = config.get("yolo_bin", "yolo")
    training_cwd = paths["training_cwd"]
    if not isinstance(yolo_bin, str) or not yolo_bin.strip():
        raise ValueError("yolo_bin 必须是命令名或可执行文件路径")
    cwd_path = Path(training_cwd).resolve(strict=True) if enabled else Path(training_cwd).resolve()
    if "/" in yolo_bin:
        yolo_bin = str(config_path(yolo_bin, cwd_path, "yolo_bin"))
    dataset_roots = config.get('dataset_roots', [])
    if not isinstance(dataset_roots, list) or any(not isinstance(v, str) or not v.strip() for v in dataset_roots):
        raise ValueError('dataset_roots 必须为路径列表')
    dataset_roots = [str(config_path(value, cwd_path, 'dataset_roots')) for value in dataset_roots]
    if enabled and any(not any(Path(root) == allowed or allowed in Path(root).parents for allowed in resolved_roots) for root in dataset_roots):
        raise ValueError('dataset_roots 必须位于 allowed_training_roots 内，以便训练读取上传的数据')
    quota = config.get('dataset_staging_quota_bytes', 100 * 1024**3)
    if type(quota) is not int or quota <= 0:
        raise ValueError('dataset_staging_quota_bytes 必须为正整数字节数')
    refresh_rate = config.get('dataset_refresh_read_rate_bytes_per_second', DEFAULT_DATASET_REFRESH_READ_RATE)
    if type(refresh_rate) is not int or not 1024**2 <= refresh_rate <= 1024**3:
        raise ValueError('dataset_refresh_read_rate_bytes_per_second 必须为 1 MiB/s～1 GiB/s 的整数')
    refresh_min_free = config.get('dataset_refresh_min_free_bytes', DEFAULT_DATASET_REFRESH_MIN_FREE)
    if type(refresh_min_free) is not int or not 0 <= refresh_min_free <= 1024**4:
        raise ValueError('dataset_refresh_min_free_bytes 必须为 0～1 TiB 的整数字节数')
    refresh_pressure = config.get('dataset_refresh_io_pressure_pct', DEFAULT_DATASET_REFRESH_IO_PRESSURE_PCT)
    if type(refresh_pressure) is not int or not 0 <= refresh_pressure <= 100:
        raise ValueError('dataset_refresh_io_pressure_pct 必须为 0～100 的整数百分比')
    lock_path = Path(paths["training_lock_file"])
    if enabled and not lock_path.parent.is_dir():
        raise ValueError("training_lock_file 的父目录必须已存在")
    return {"center_url": center, "node_id": node_id, "token": token,
            "report_interval_seconds": interval, "nvidia_smi_bin": binary,
            "model_transfer_finalize_timeout_seconds": model_finalize_timeout,
            "training_stop_timeout_seconds": stop_timeout,
            "enable_training": enabled, "allowed_training_roots": [str(item) for item in resolved_roots],
            "training_output_roots": [str(item) for item in resolved_outputs],
            "model_test_roots": paths["model_test_roots"],
            "yolo_bin": yolo_bin, "training_cwd": str(cwd_path),
            "training_lock_file": str(lock_path),
            "training_state_dir": str(path.resolve().parent / ".training-state"),
            "enable_training_trace": trace_enabled, "dataset_roots": dataset_roots,
            "dataset_staging_quota_bytes": quota,
            "dataset_refresh_read_rate_bytes_per_second": refresh_rate,
            "dataset_refresh_min_free_bytes": refresh_min_free,
            "dataset_refresh_io_pressure_pct": refresh_pressure}


class LockBundle:
    """Keep a deterministic set of advisory locks alive as one ownership token."""

    def __init__(self, handles: list, primary=0):
        self.handles = handles
        self.primary = primary

    def fileno(self) -> int:
        return self.handles[self.primary].fileno()

    def filenos(self) -> tuple[int, ...]:
        return tuple(handle.fileno() for handle in self.handles)

    def close(self) -> None:
        # Closing without LOCK_UN preserves locks inherited by a training child.
        for handle in reversed(self.handles):
            handle.close()
        self.handles.clear()


class TrainingDetached(Exception):
    """The live training process retains ownership until the next Agent adopts it."""


def detach_training(config: dict, task: dict, command: list, identity: dict | None) -> None:
    # Never publish a terminal result or unlock a live child's inherited flock.
    # A previous launching/running record remains useful if this refresh fails.
    try:
        if not process_identity_matches(identity):
            raise ValueError("无法重新确认训练进程身份，保留原活动记录等待核验")
        save_active_run(config, task, "running", command=command, identity=identity)
    except (OSError, ValueError, TypeError):
        LOGGER.exception("训练交接记录刷新失败；保留原记录，不停止进程、不释放其资源锁")
    LOGGER.info("Agent 退出，训练 %s 继续运行；下次启动将核验并接管", task["run_id"])
    raise TrainingDetached()


def _release_lock_handles(handles: list) -> None:
    for handle in reversed(handles):
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def acquire_training_lock(config: dict, output_roots: list[Path] | None = None):
    """Reserve the configured workspace and every overlapping output-root scope."""
    lock_path = Path(config.get("training_lock_file") or Path(config["training_cwd"]) / ".yolo-forge-training.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    os.chmod(lock_path, 0o660)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    workspace_locks = acquire_output_root_locks(config, output_roots)
    if workspace_locks is None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    json.dump({"pid": os.getpid(), "node_id": config["node_id"],
               "acquired_at": datetime.now(timezone.utc).isoformat()}, handle)
    handle.flush()
    return LockBundle([handle, *workspace_locks.handles])


def release_training_lock(handle) -> None:
    if handle is None:
        return
    handles = handle.handles if isinstance(handle, LockBundle) else [handle]
    _release_lock_handles(handles)
    if isinstance(handle, LockBundle):
        handle.handles = []


def process_holds_training_lock(config: dict, pid: int) -> bool:
    """Return whether a live process has an FD for this workspace lock inode."""
    if type(pid) is not int or pid <= 0:
        return False
    lock_path = Path(config.get("training_lock_file") or Path(config["training_cwd"]) / ".yolo-forge-training.lock")
    try:
        lock_stat = lock_path.stat()
        descriptors = list(Path(f"/proc/{pid}/fd").iterdir())
    except OSError:
        return False
    for descriptor in descriptors:
        try:
            descriptor_stat = descriptor.stat()
        except OSError:
            continue
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
            return True
    return False


def process_holds_task_locks(config: dict, task: dict, pid: int) -> bool:
    """Check that a recovered process inherited every reservation for its Run."""
    if type(pid) is not int or pid <= 0:
        return False
    # Pre-2.0 tasks have no assignment payload and inherited only the shared
    # workspace lock. Keep recognizing those processes so restart recovery can
    # continue monitoring them instead of treating their live lock as a conflict.
    if "gpu_assignment" not in task:
        return process_holds_training_lock(config, pid)
    assignment = normalize_gpu_assignment(task)
    paths = [configured_training_lock(config)]
    roots = configured_output_roots(config)
    output_root = (task.get("parameters") or {}).get("remote_output_root")
    if output_root:
        roots = _normalized_output_roots([*roots, Path(output_root).resolve()])
    lock_directory = workspace_state_dir(config) / "locks"
    for value in _output_lock_requests(roots):
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        paths.append(lock_directory / f"{digest}.lock")
    if assignment:
        gpu_directory = workspace_state_dir(config) / "gpu-locks"
        for item in assignment:
            digest = hashlib.sha256(item["uuid"].encode("utf-8")).hexdigest()
            paths.append(gpu_directory / f"{digest}.lock")
    try:
        descriptors = list(Path(f"/proc/{pid}/fd").iterdir())
    except OSError:
        return False
    held = set()
    for descriptor in descriptors:
        try:
            info = descriptor.stat()
        except OSError:
            continue
        held.add((info.st_dev, info.st_ino))
    for path in paths:
        try:
            info = path.stat()
        except OSError:
            return False
        if (info.st_dev, info.st_ino) not in held:
            return False
    return True


def acquire_agent_instance_lock(config: dict):
    """Prevent two processes from polling with the same on-disk node config."""
    directory = Path(config["training_state_dir"])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / ".agent-instance.lock"
    handle = path.open("a+", encoding="utf-8")
    os.chmod(path, 0o600)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    json.dump({"pid": os.getpid(), "node_id": config["node_id"],
               "acquired_at": datetime.now(timezone.utc).isoformat()}, handle)
    handle.flush()
    return handle


def configured_training_lock(config: dict) -> Path:
    return Path(config.get("training_lock_file") or Path(config["training_cwd"]) / ".yolo-forge-training.lock").resolve()


def workspace_state_dir(config: dict) -> Path:
    override = config.get("_workspace_state_dir")
    if override is not None:
        directory = Path(override)
        if not directory.is_absolute():
            raise ValueError("内部 workspace 状态目录必须是绝对路径")
    else:
        configured = os.environ.get("XDG_STATE_HOME")
        base = Path(configured) if configured else Path.home() / ".local" / "state"
        if not base.is_absolute():
            raise ValueError("XDG_STATE_HOME 必须是绝对路径")
        directory = base / "yolo-forge-agent"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def workspace_registry_dir(config: dict) -> Path:
    return workspace_state_dir(config) / "workspaces"


def legacy_workspace_registry_dir(config: dict) -> Path:
    lock_path = configured_training_lock(config)
    return lock_path.with_name(lock_path.name + ".agents")


def workspace_owner_id(config: dict) -> str:
    return hashlib.sha256(config["node_id"].encode("ascii")).hexdigest()


def _normalized_output_roots(values) -> list[Path]:
    roots = sorted({Path(value).resolve() for value in values}, key=lambda item: (len(item.parts), str(item)))
    return [root for root in roots if not any(parent == root or parent in root.parents for parent in roots if parent != root)]


def configured_output_roots(config: dict) -> list[Path]:
    values = config.get("training_output_roots") or config.get("allowed_training_roots") or []
    return _normalized_output_roots(values)


def paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _output_lock_requests(roots: list[Path]) -> dict[str, int]:
    requests: dict[str, int] = {}
    for root in _normalized_output_roots(roots):
        current = Path(root.anchor)
        prefixes = [current]
        for part in root.parts[1:]:
            current /= part
            prefixes.append(current)
        for prefix in prefixes[:-1]:
            requests.setdefault(str(prefix), fcntl.LOCK_SH)
        requests[str(root)] = fcntl.LOCK_EX
    return requests


def acquire_output_root_locks(config: dict, additional_roots: list[Path] | None = None,
                              *, shared: bool = False) -> LockBundle | None:
    """Lock output scopes; shared holders are active 2.x runs and exclusive holders mutate files."""
    roots = configured_output_roots(config)
    if additional_roots:
        roots = _normalized_output_roots([*roots, *additional_roots])
    requests = ({path: fcntl.LOCK_SH for path in _output_lock_requests(roots)}
                if shared else _output_lock_requests(roots))
    directory = workspace_state_dir(config) / "locks"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    handles = []
    try:
        for path, mode in sorted(requests.items()):
            digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
            handle = (directory / f"{digest}.lock").open("a+", encoding="utf-8")
            os.chmod(handle.name, 0o600)
            try:
                fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                _release_lock_handles(handles)
                return None
            handles.append(handle)
    except Exception:
        _release_lock_handles(handles)
        raise
    return LockBundle(handles)


def _gpu_identity_rows(binary: str) -> list[dict]:
    """Read the launch-time GPU index/UUID identity map used to verify a center reservation."""
    try:
        output = query(binary, ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("无法核对平台分配的 GPU；nvidia-smi 不可用") from exc
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 2:
            continue
        index = number(row[0], True)
        uuid = row[1].strip()
        if index is None or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", uuid):
            continue
        rows.append({"index": index, "uuid": uuid})
    indexes = [row["index"] for row in rows]
    uuids = [row["uuid"] for row in rows]
    if not rows or len(indexes) != len(set(indexes)) or len(uuids) != len(set(uuids)):
        raise ValueError("无法建立唯一的 GPU 索引与 UUID 映射")
    return rows


def normalize_gpu_assignment(task: dict) -> list[dict] | None:
    """Validate the center payload structure and requested-device ordering without trusting it."""
    if "gpu_assignment" not in task:
        return None  # Old centers retain the legacy server-wide serialization path.
    raw = task.get("gpu_assignment")
    if not isinstance(raw, list) or len(raw) > 128:
        raise ValueError("平台下发的 GPU 分配格式无效")
    requested = str((task.get("parameters") or {}).get("device", "")).strip().lower()
    if requested in {"cpu", "mps"}:
        if raw:
            raise ValueError("CPU/MPS 任务不应包含 GPU 分配")
        return []
    if not re.fullmatch(r"\d+(?:,\d+)*", requested):
        raise ValueError("训练设备不是可验证的 GPU 索引列表")
    requested_indexes = [int(value) for value in requested.split(",")]
    if not raw or len(raw) != len(requested_indexes):
        raise ValueError("平台分配的 GPU 数量与训练设备不一致")
    normalized = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"uuid", "index"}:
            raise ValueError("平台下发的 GPU 分配项无效")
        index, uuid = item.get("index"), item.get("uuid")
        if type(index) is not int or index < 0 or index != requested_indexes[position]:
            raise ValueError("平台分配的 GPU 顺序与训练设备不一致")
        if not isinstance(uuid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", uuid):
            raise ValueError("平台下发的 GPU UUID 无效")
        normalized.append({"uuid": uuid, "index": index})
    if (len({item["uuid"] for item in normalized}) != len(normalized)
            or len({item["index"] for item in normalized}) != len(normalized)):
        raise ValueError("平台下发的 GPU 分配包含重复设备")
    return normalized


def validate_gpu_assignment(config: dict, task: dict) -> list[dict] | None:
    assignment = normalize_gpu_assignment(task)
    if assignment is None or not assignment:
        return None
    inventory = _gpu_identity_rows(config.get("nvidia_smi_bin", "nvidia-smi"))
    by_index = {item["index"]: item["uuid"] for item in inventory}
    for item in assignment:
        if by_index.get(item["index"]) != item["uuid"]:
            raise ValueError(
                f"GPU {item['index']} 的当前 UUID 与平台分配不一致，已拒绝启动训练"
            )
    return assignment


def _acquire_gpu_uuid_locks(config: dict, task: dict, assignment: list[dict]) -> LockBundle | None:
    directory = workspace_state_dir(config) / "gpu-locks"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    handles = []
    try:
        for item in sorted(assignment, key=lambda value: value["uuid"]):
            digest = hashlib.sha256(item["uuid"].encode("utf-8")).hexdigest()
            handle = (directory / f"{digest}.lock").open("a+", encoding="utf-8")
            os.chmod(handle.name, 0o600)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                _release_lock_handles(handles)
                return None
            handle.seek(0)
            handle.truncate()
            json.dump({"node_id": config["node_id"], "run_id": task.get("run_id"),
                       "gpu_uuid": item["uuid"], "acquired_at": datetime.now(timezone.utc).isoformat()}, handle)
            handle.flush()
            handles.append(handle)
    except Exception:
        _release_lock_handles(handles)
        raise
    return LockBundle(handles)


def acquire_parallel_training_lock(config: dict, task: dict) -> LockBundle | None:
    """Reserve assigned GPUs while sharing the legacy workspace gate with other 2.x runs."""
    assignment = validate_gpu_assignment(config, task)
    if not assignment:
        raise ValueError("并行训练缺少可验证的 GPU 分配")
    lock_path = configured_training_lock(config)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    legacy = lock_path.open("a+", encoding="utf-8")
    os.chmod(lock_path, 0o660)
    try:
        fcntl.flock(legacy.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        legacy.close()
        return None
    output_root = Path(task["parameters"]["remote_output_root"]).resolve()
    root_locks = acquire_output_root_locks(config, [output_root], shared=True)
    if root_locks is None:
        _release_lock_handles([legacy])
        return None
    gpu_locks = _acquire_gpu_uuid_locks(config, task, assignment)
    if gpu_locks is None:
        release_training_lock(root_locks)
        _release_lock_handles([legacy])
        return None
    return LockBundle([legacy, *root_locks.handles, *gpu_locks.handles])


def acquire_task_training_lock(config: dict, task: dict):
    """Use GPU-scoped 2.x reservations, falling back to the legacy exclusive protocol."""
    assignment = normalize_gpu_assignment(task)
    if assignment:
        return acquire_parallel_training_lock(config, task)
    return acquire_training_lock(config)


def training_launch_environment(config: dict, task: dict) -> dict[str, str]:
    """Map a verified UUID reservation into a compact CUDA device namespace."""
    environment = training_process_environment(config, task)
    assignment = normalize_gpu_assignment(task)
    if not assignment:
        return environment
    assignment = validate_gpu_assignment(config, task) or []
    if not assignment:
        raise ValueError("平台分配的 GPU 在启动时不可用")
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(item["uuid"] for item in assignment)
    return environment


def training_command_for_assignment(command: list[str], task: dict) -> list[str]:
    """Rewrite requested physical indices to logical CUDA indices after masking visibility."""
    assignment = normalize_gpu_assignment(task)
    if not assignment:
        return command
    logical = ",".join(str(index) for index in range(len(assignment)))
    rewritten = list(command)
    for index, value in enumerate(rewritten):
        if value.startswith("device="):
            rewritten[index] = "device=" + logical
            break
    else:
        raise ValueError("训练命令中缺少设备参数")
    return rewritten


def _workspace_registry_lock(config: dict):
    directory = workspace_registry_dir(config)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = workspace_state_dir(config) / ".registry.lock"
    handle = path.open("a+", encoding="utf-8")
    os.chmod(path, 0o600)
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _live_agent_configs():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        pid = int(entry.name)
        argv = _process_argv(pid)
        config_path = _agent_config_path(pid, argv)
        if config_path is None:
            continue
        try:
            if config_path.stat().st_size > 128 * 1024:
                yield pid, None
                continue
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = {**payload, **normalized_training_paths(payload, config_path)}
            yield pid, payload if isinstance(payload, dict) else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            yield pid, None


def _cleanup_legacy_workspace_markers(config: dict) -> None:
    directory = legacy_workspace_registry_dir(config)
    if not directory.is_dir():
        return
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("node_id") == config["node_id"]:
                path.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue


def register_workspace(config: dict) -> Path:
    """Atomically register this live node without persisting credentials."""
    current_identity = process_identity(os.getpid())
    if current_identity is None:
        raise ValueError("无法确认当前 Agent 进程身份")
    for _pid, other in _live_agent_configs():
        if other is not None and other.get("node_id") == config["node_id"]:
            raise ValueError("同一 node_id 已有另一个 Agent 进程在运行")
    directory = workspace_registry_dir(config)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    owner = workspace_owner_id(config)
    path = directory / f"{owner}.json"
    payload = {
        "version": WORKSPACE_REGISTRY_VERSION,
        "owner_id": owner,
        "node_id": config["node_id"],
        "training_output_roots": [str(path) for path in configured_output_roots(config)],
        "process": current_identity,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    guard = _workspace_registry_lock(config)
    try:
        for candidate in directory.glob("*.json"):
            try:
                existing = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                if candidate == path:
                    candidate.unlink(missing_ok=True)
                    continue
                raise ValueError("共享 workspace 登记文件损坏，已拒绝继续") from None
            if not isinstance(existing, dict) or existing.get("node_id") != config["node_id"]:
                continue
            identity = existing.get("process")
            if process_identity_matches(identity) and identity.get("pid") != os.getpid():
                raise ValueError("同一 node_id 已有另一个 Agent 进程在运行")
            if candidate != path or not process_identity_matches(identity):
                candidate.unlink(missing_ok=True)
        _atomic_private_json(path, payload)
        _cleanup_legacy_workspace_markers(config)
    finally:
        release_training_lock(guard)
    return path


def _agent_config_path(pid: int, argv: list[str]) -> Path | None:
    if not any(Path(argument).name == "monitor_agent.py" for argument in argv):
        return None
    raw = None
    for index, argument in enumerate(argv):
        if argument == "--config" and index + 1 < len(argv):
            raw = argv[index + 1]
            break
        if argument.startswith("--config="):
            raw = argument.split("=", 1)[1]
            break
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    try:
        return (Path(f"/proc/{pid}/cwd").resolve(strict=True) / path).resolve()
    except OSError:
        return None


def _other_live_agent_conflicts(config: dict, target_roots: list[Path]) -> bool:
    """Detect older Agents that cannot publish a shared workspace record."""
    for _pid, other in _live_agent_configs():
        try:
            if other is None:
                return True
            if not other.get("enable_training", False):
                continue
            values = other.get("training_output_roots") or other.get("allowed_training_roots") or []
            other_roots = _normalized_output_roots(
                value for value in values if isinstance(value, str) and Path(value).is_absolute()
            )
            if not other_roots or any(paths_overlap(target, root) for target in target_roots for root in other_roots):
                return True
        except (OSError, ValueError, KeyError, TypeError):
            # A visible Agent with an unreadable configuration is an unknown
            # owner. Refusing deletion is safer than guessing its path scope.
            return True
    return False


def workspace_delete_conflicts(config: dict, target_roots: list[Path]) -> bool:
    """Reject deletion when another center owns an overlapping output root."""
    register_workspace(config)
    owner = workspace_owner_id(config)
    directory = workspace_registry_dir(config)
    guard = _workspace_registry_lock(config)
    try:
        for path in directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (not isinstance(payload, dict) or payload.get("version") != WORKSPACE_REGISTRY_VERSION or
                        not isinstance(payload.get("owner_id"), str)):
                    return True
                if payload["owner_id"] == owner:
                    continue
                if not process_identity_matches(payload.get("process")):
                    path.unlink(missing_ok=True)
                    continue
                values = payload.get("training_output_roots")
                if not isinstance(values, list) or any(
                        not isinstance(value, str) or not Path(value).is_absolute() for value in values):
                    return True
                other_roots = _normalized_output_roots(values)
                if any(paths_overlap(target, root) for target in target_roots for root in other_roots):
                    return True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return True
    finally:
        release_training_lock(guard)
    return _other_live_agent_conflicts(config, target_roots)


def _atomic_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def process_identity(pid: int) -> dict | None:
    """Read immutable Linux process identity fields, rejecting zombies."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        fields = raw[closing + 2:].split()
        if closing < 0 or len(fields) <= 19 or fields[0] == "Z":
            return None
        command_line = Path(f"/proc/{pid}/cmdline").read_bytes()
        if not command_line:
            return None
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        return {"pid": pid, "start_ticks": int(fields[19]), "boot_id": boot_id,
                "cmdline_sha256": hashlib.sha256(command_line).hexdigest()}
    except (OSError, ValueError, IndexError):
        return None


def process_identity_matches(identity: dict | None) -> bool:
    if not isinstance(identity, dict) or type(identity.get("pid")) is not int:
        return False
    current = process_identity(identity["pid"])
    return bool(current and current == identity)


def process_parent(pid: int) -> int | None:
    """Read a PID's parent from procfs, returning None when it is not inspectable."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        closing = raw.rfind(")")
        fields = raw[closing + 2:].split()
        if closing < 0 or not fields or fields[0] == "Z":
            return None
        return int(fields[1])
    except (OSError, ValueError, IndexError):
        return None


def process_environment(pid: int) -> dict[str, str] | None:
    try:
        entries = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    result = {}
    for entry in entries:
        key, separator, value = entry.partition(b"=")
        if separator:
            result[os.fsdecode(key)] = os.fsdecode(value)
    return result


def active_attribution_roots(config: dict | None) -> dict[int, dict]:
    """Load only live root identities from this Agent's durable active state."""
    roots: dict[int, dict] = {}
    if not config:
        return roots
    directory = Path(config.get("training_state_dir", "")) / "active"
    try:
        paths = list(directory.glob("*.json"))
    except OSError:
        return roots
    for path in paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                continue
            task = state.get("task") if isinstance(state, dict) else None
            identity = state.get("process") if isinstance(state, dict) else None
            if (state.get("version") != ACTIVE_STATE_VERSION or state.get("node_id") != config.get("node_id")
                    or state.get("phase") != "running" or not isinstance(task, dict)
                    or not process_identity_matches(identity)):
                continue
            roots[identity["pid"]] = {"instance_id": task.get("platform_instance_id"),
                                       "node_id": config["node_id"], "run_id": task.get("run_id"),
                                       "task_id": task.get("task_id")}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return roots


def process_owner_hint(pid: int, active_roots: dict[int, dict], max_depth: int = 32) -> dict:
    """Identify the run marker or verified active launcher above a GPU process."""
    current = pid
    chain: list[int] = []
    markers = []
    leaf_environment_accessible = False
    complete = False
    for _ in range(max_depth):
        if current <= 0 or current in chain:
            break
        parent = process_parent(current)
        if parent is None:
            break
        chain.append(current)
        environment = process_environment(current)
        if len(chain) == 1:
            leaf_environment_accessible = environment is not None
        if environment:
            marker = {
                "instance_id": environment.get("YOLO_FORGE_INSTANCE_ID"),
                "node_id": environment.get("YOLO_FORGE_NODE_ID"),
                "run_id": environment.get("YOLO_FORGE_RUN_ID"),
                "task_id": environment.get("YOLO_FORGE_TASK_ID"),
            }
            if (re.fullmatch(r"[0-9a-f]{32}", marker["instance_id"] or "")
                    and re.fullmatch(r"[0-9a-f]{32}", marker["run_id"] or "")
                    and re.fullmatch(r"[A-Za-z0-9_-]{1,160}", marker["task_id"] or "")
                    and (marker["node_id"] == "local" or
                         re.fullmatch(r"[0-9a-f]{32}", marker["node_id"] or ""))):
                if marker not in markers:
                    markers.append(marker)
        if current == 1 or parent == 0:
            complete = True
            break
        current = parent

    root_candidates = [active_roots[ancestor] for ancestor in chain if ancestor in active_roots]
    keys = {(item.get("instance_id"), item.get("node_id"), item.get("run_id"), item.get("task_id"))
            for item in root_candidates + markers}
    if len(keys) > 1:
        return {"match_method": "ambiguous", "scan_complete": complete}
    if markers:
        return {**markers[0], "match_method": "runtime_marker", "scan_complete": complete}
    if root_candidates:
        return {**root_candidates[0], "match_method": "ancestor", "scan_complete": complete}
    return {"match_method": "none", "scan_complete": complete and leaf_environment_accessible}


def training_process_environment(config: dict, task: dict) -> dict[str, str]:
    """Add non-secret run identifiers inherited by torchrun and DDP workers."""
    environment = os.environ.copy()
    environment["MKL_THREADING_LAYER"] = "GNU"
    environment["PYTHONUNBUFFERED"] = "1"
    markers = {
        "YOLO_FORGE_INSTANCE_ID": task.get("platform_instance_id"),
        "YOLO_FORGE_NODE_ID": config.get("node_id"),
        "YOLO_FORGE_RUN_ID": task.get("run_id"),
        "YOLO_FORGE_TASK_ID": task.get("task_id"),
    }
    # Result identity is also required with older centers that lack attribution.
    if (re.fullmatch(r"[0-9a-f]{32}", markers["YOLO_FORGE_RUN_ID"] or "")
            and re.fullmatch(r"[A-Za-z0-9_-]{1,160}", markers["YOLO_FORGE_TASK_ID"] or "")):
        environment.update({key: str(markers[key]) for key in ("YOLO_FORGE_RUN_ID", "YOLO_FORGE_TASK_ID")})
    if (re.fullmatch(r"[0-9a-f]{32}", markers["YOLO_FORGE_INSTANCE_ID"] or "")
            and re.fullmatch(r"[0-9a-f]{32}", markers["YOLO_FORGE_RUN_ID"] or "")
            and re.fullmatch(r"[A-Za-z0-9_-]{1,160}", markers["YOLO_FORGE_TASK_ID"] or "")
            and re.fullmatch(r"[0-9a-f]{32}", markers["YOLO_FORGE_NODE_ID"] or "")):
        environment.update({key: str(value) for key, value in markers.items()})
    return environment


def _process_argv(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [os.fsdecode(item) for item in raw.rstrip(b"\0").split(b"\0") if item]
    except OSError:
        return []


def _argv_matches_command(argv: list[str], command: list[str]) -> bool:
    suffix = command[1:]
    return argv == command or bool(suffix and len(argv) >= len(suffix) and argv[-len(suffix):] == suffix)


def find_process_for_command(command: list[str]) -> list[dict]:
    """Find exact command arguments after a possible shebang interpreter prefix."""
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        argv = _process_argv(int(entry.name))
        if _argv_matches_command(argv, command):
            identity = process_identity(int(entry.name))
            if identity:
                matches.append(identity)
    return matches


def active_state_path(config: dict, run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("运行 ID 无效")
    return Path(config["training_state_dir"]) / "active" / f"{run_id}.json"


def save_active_run(config: dict, task: dict, phase: str, *, command: list[str] | None = None,
                    identity: dict | None = None) -> Path:
    if phase not in {"claimed", "confirmed", "launching", "running"}:
        raise ValueError("活动训练阶段无效")
    run_id = task.get("run_id", "")
    path = active_state_path(config, run_id)
    payload = {"version": ACTIVE_STATE_VERSION, "node_id": config["node_id"],
               "run_id": run_id, "phase": phase, "task": task,
               "command": command, "process": identity,
               "updated_at": datetime.now(timezone.utc).isoformat()}
    _atomic_private_json(path, payload)
    return path


def clear_active_run(config: dict, run_id: str) -> None:
    active_state_path(config, run_id).unlink(missing_ok=True)


def load_active_runs(config: dict) -> list[dict]:
    """Load every durable Run independently, pruning acknowledged terminal markers."""
    directory = Path(config["training_state_dir"]) / "active"
    if not directory.exists():
        return []
    paths = sorted(directory.glob("*.json"))
    states = []
    for path in paths:
        state = json.loads(path.read_text(encoding="utf-8"))
        run_id = path.stem
        if (state.get("version") != ACTIVE_STATE_VERSION or state.get("node_id") != config["node_id"] or
                state.get("run_id") != run_id or state.get("phase") not in {"claimed", "confirmed", "launching", "running"} or
                not isinstance(state.get("task"), dict) or state["task"].get("run_id") != run_id):
            raise ValueError("活动训练状态文件无效，已阻止继续领取任务")
        # A terminal payload is authoritative if the Agent stopped between writing
        # it and removing the active marker.
        if (Path(config["training_state_dir"]) / f"{run_id}.json").is_file():
            path.unlink(missing_ok=True)
            continue
        states.append(state)
    return states


def load_active_run(config: dict) -> dict | None:
    """Backward-compatible single-Run accessor used by older Agent integrations."""
    states = load_active_runs(config)
    if len(states) > 1:
        raise ValueError("检测到多个活动训练状态；请使用 load_active_runs()")
    return states[0] if states else None


def api_json(config: dict, path: str, method: str = "GET", payload: dict | None = None,
             extra_headers: dict[str, str] | None = None, *, timeout_seconds: int = 20) -> dict:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if payload is not None else None
    headers = {"Authorization": "Bearer " + config["token"], "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    request = Request(config["center_url"] + path, data=data, method=method, headers=headers)
    with build_opener(NoRedirect()).open(request, timeout=timeout_seconds) as response:
        body = response.read(2 * 1024 * 1024)
    return json.loads(body or b"{}")


TRAINING_KEYS = {"patience", "lr0", "pretrained", "cache", "amp", "close_mosaic", "cos_lr", "weight_decay", "momentum", "warmup_epochs", "freeze", "multi_scale", "rect", "seed", "deterministic", "fraction", "save_period", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "box", "cls", "dfl", "save", "verbose", "single_cls", "classes", "profile", "compile", "channels_last", "overlap_mask", "mask_ratio", "dropout", "val", "split", "save_json", "conf", "iou", "max_det", "plots", "lrf", "warmup_momentum", "warmup_bias_lr", "nbs", "bgr", "cutmix", "copy_paste_mode", "auto_augment", "erasing", "cls_pw", "pose", "kobj", "rle", "angle", "dlog", "dgrad", "dlam", "cls_remap", "dis", "distill_model", "dnn", "nms", "quantize", "time"}
RESERVED_KEYS = {"model", "data", "project", "name", "device", "epochs", "batch", "imgsz", "workers", "optimizer", "task", "mode", "exist_ok", "save_dir", "resume", "source"}


def within_allowed(path: str, roots: list[str], *, file_required: bool = True,
                   label: str = "路径") -> Path:
    try:
        candidate = Path(path).resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError):
        raise ValueError(
            f"远程服务器上的{label}不存在：{path}。"
            "请填写所选远程服务器上的路径，不是中心平台路径"
        ) from None
    if file_required and not candidate.is_file():
        raise ValueError(f"远程服务器上的{label}不是文件：{candidate}")
    if not file_required and not candidate.is_dir():
        raise ValueError(f"远程服务器上的{label}不是目录：{candidate}")
    if not any(candidate == Path(root) or Path(root) in candidate.parents for root in roots):
        raise ValueError(f"远程服务器上的{label}不在 Agent 允许目录内：{candidate}；"
                         f"allowed_training_roots：{'、'.join(roots)}")
    return candidate


def scalar_parameter(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)) and math.isfinite(value):
        return str(value)
    if isinstance(value, str) and len(value) <= 500 and not any(ord(char) < 32 for char in value):
        return value
    raise ValueError("不支持的参数类型")


def training_data_path(task_type: str, raw: str, roots: list[str]) -> Path:
    """Validate the different data contract used by classification training."""
    if task_type == "classify":
        dataset = within_allowed(raw, roots, file_required=False, label="分类数据集目录")
        train = dataset / "train"
        if not train.is_dir():
            raise ValueError(f"远程分类数据集缺少 train 目录：{train}")
        try:
            has_class = any(child.is_dir() and not child.name.startswith(".") for child in train.iterdir())
        except OSError as exc:
            raise ValueError(f"远程分类数据集无法读取：{dataset}") from exc
        if not has_class:
            raise ValueError(f"远程分类数据集 train 目录中没有类别子目录：{train}")
        return dataset
    dataset = within_allowed(raw, roots, label="数据集 YAML 文件")
    if dataset.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"非分类任务的数据集必须是 YAML 文件：{dataset}")
    return dataset


def build_training_command(config: dict, task: dict) -> list[str]:
    params = task["parameters"]
    roots = config["allowed_training_roots"]
    model = within_allowed(task["model_path"], roots, label="基础模型文件")
    dataset = training_data_path(task["task_type"], task["data_path"], roots)
    run_dir = Path(task["run_dir"]).resolve()
    output_root = within_allowed(params["remote_output_root"], roots, file_required=False,
                                 label="训练输出根目录")
    configured_outputs = [Path(item) for item in config.get("training_output_roots", [])]
    if configured_outputs and not any(output_root == root or root in output_root.parents
                                      for root in configured_outputs):
        raise ValueError(f"训练输出根目录不在 training_output_roots 内：{output_root}；"
                         f"可用输出根目录：{'、'.join(str(root) for root in configured_outputs)}；"
                         "请修改任务输出目录，或核对远端实际加载的 Agent 配置后重新启动 Agent")
    try:
        _probe_writable_directory(output_root)
    except OSError as exc:
        raise ValueError(f"训练输出根目录不可写：{output_root}") from exc
    if output_root not in run_dir.parents:
        raise ValueError("训练运行目录必须位于配置的输出根目录内")
    task_id = task["task_id"]
    attempt_name = f"attempt-{int(task['attempt_number']):03d}"
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", task_id):
        raise ValueError("任务 ID 无效")
    if task["task_type"] not in {"detect", "segment", "classify", "pose", "obb"}:
        raise ValueError("训练任务类型无效")
    if run_dir != output_root / task["task_type"] / task_id / attempt_name:
        raise ValueError("训练运行目录必须对应当前任务和 Attempt")
    log_path = Path(task["log_path"])
    try:
        datetime.strptime(log_path.parent.parent.name, "%Y%m%d")
    except ValueError:
        raise ValueError("日志目录必须包含年月日") from None
    expected_log = output_root / ".platform-logs" / log_path.parent.parent.name / task_id / f"{attempt_name}.log"
    if log_path != expected_log or log_path.resolve() != expected_log:
        raise ValueError("日志路径必须位于输出根目录的独立 .platform-logs 目录内")
    if log_path.exists():
        raise ValueError("训练日志已存在，禁止覆盖旧日志")
    if run_dir.exists():
        raise ValueError("训练 Attempt 目录已存在，禁止覆盖旧训练产物")
    yolo_bin = str(Path(config["yolo_bin"]).resolve(strict=True)) if "/" in config["yolo_bin"] else config["yolo_bin"]
    if "/" in yolo_bin and not os.access(yolo_bin, os.X_OK):
        raise ValueError("配置的 yolo_bin 不存在或不可执行")
    command = [yolo_bin, task["task_type"], "train", f"model={model}", f"data={dataset}",
               f"epochs={int(params['epochs'])}", f"batch={int(params['batch'])}", f"imgsz={int(params['imgsz'])}",
               f"device={scalar_parameter(params['device'])}", f"workers={int(params['workers'])}",
               f"optimizer={scalar_parameter(params['optimizer'])}", f"project={run_dir.parent}", f"name={run_dir.name}", "exist_ok=True"]
    checkpoint = task.get("checkpoint_path")
    if checkpoint:
        checkpoint_path = within_allowed(checkpoint, roots, label="续训检查点")
        if (checkpoint_path.name != "last.pt" or checkpoint_path.parent.name != "weights" or
                checkpoint_path.parent.parent.parent.name != task_id or
                checkpoint_path.parent.parent.parent.parent.name != task["task_type"] or
                not re.fullmatch(r"attempt-\d{3,}", checkpoint_path.parent.parent.name) or
                int(checkpoint_path.parent.parent.name[8:]) >= int(task["attempt_number"])):
            raise ValueError("续训检查点必须属于该任务的历史 Attempt")
        # Ultralytics reloads saved args on resume; explicit save_dir preserves history.
        command.extend([f"resume={checkpoint_path}", f"save_dir={run_dir}"])
    extras = params.get("config", {})
    for key, value in extras.items():
        if key in TRAINING_KEYS and value is not None:
            command.append(f"{key}={scalar_parameter(value)}")
    for key, value in (extras.get("custom_args") or {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) or key in RESERVED_KEYS or key in TRAINING_KEYS:
            raise ValueError(f"不允许的自定义训练参数：{key}")
        command.append(f"{key}={scalar_parameter(value)}")
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return command


def operation_paths(
    config: dict,
    payload: dict,
    *,
    allow_missing_root: bool = False,
    allow_legacy_flat: bool = False,
) -> tuple[Path, Path | None]:
    """Validate all deletion/scan/fetch targets against an exact task/Attempt layout."""
    task_id, task_type = payload["task_id"], payload["task_type"]
    attempt = payload["attempt_number"]
    if (not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", task_id) or
            task_type not in {"detect", "segment", "classify", "pose", "obb"} or
            type(attempt) is not int or attempt < 1):
        raise ValueError("远程文件操作任务信息无效")
    root = Path(payload["output_root"])
    root_missing = not root.exists()
    resolved_root = root.resolve()
    if (not root.is_absolute() or root.is_symlink() or resolved_root != root or
            (not root_missing and not root.is_dir())):
        raise ValueError("输出根目录无效或包含符号链接")
    if root_missing:
        raise ValueError("输出根目录不存在，无法验证或执行远程文件操作")
    allowed_roots = [Path(item).resolve() for item in config["allowed_training_roots"]]
    enclosing_roots = [allowed for allowed in allowed_roots
                       if root == allowed or allowed in root.parents]
    if not enclosing_roots:
        raise ValueError("输出根目录不在 Agent 允许目录内")
    run_dir = Path(payload["run_dir"])
    layout = payload.get("layout", "canonical")
    if layout == "legacy_flat":
        if not allow_legacy_flat or attempt != 1:
            raise ValueError("旧版平铺 Attempt 仅允许在删除操作中清理")
        expected = root / task_type / task_id
    elif layout == "canonical":
        expected = root / task_type / task_id / f"attempt-{attempt:03d}"
    else:
        raise ValueError("Attempt 目录布局无效")
    if run_dir != expected or run_dir.resolve() != expected or run_dir.is_symlink():
        raise ValueError("只能操作该任务的受管 Attempt 目录")
    log = Path(payload["log_path"]) if payload.get("log_path") else None
    if log is not None:
        legacy = run_dir / "platform.log"
        date = log.parent.parent.name
        expected_log = root / ".platform-logs" / date / task_id / f"attempt-{attempt:03d}.log"
        if log != legacy:
            datetime.strptime(date, "%Y%m%d")
            if log != expected_log:
                raise ValueError("只能操作该任务的独立日志")
        if log.resolve() != log or log.is_symlink():
            raise ValueError("日志路径包含符号链接")
    return run_dir, log


def _delete_plan_run(run: dict) -> dict:
    return {
        key: run.get(key)
        for key in (
            "task_id", "task_type", "run_id", "attempt_number", "run_dir",
            "log_path", "output_root", "layout",
        )
    }


def deletion_plan_digest(runs: list[dict]) -> str:
    document = {
        "version": DELETE_RECEIPT_VERSION,
        "scope": "task",
        "runs": [_delete_plan_run(run) for run in runs],
    }
    encoded = json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deletion_plan_targets(runs: list[dict]) -> list[str]:
    targets = []
    seen = set()
    for run in runs:
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id or len(run_id) > 160 or run_id in seen:
            raise ValueError("远程删除计划包含无效或重复的 Attempt 标识")
        seen.add(run_id)
        targets.append(f"{run_id}:output")
        if run.get("log_path"):
            targets.append(f"{run_id}:log")
    return targets


def report_manifest(config: dict, payload: dict) -> None:
    paths = operation_paths(config, payload)
    assert paths is not None
    run_dir, _ = paths
    items = []
    if run_dir.is_dir():
        for path in run_dir.rglob("*"):
            relative = path.relative_to(run_dir)
            if relative.parts and relative.parts[0] == "training-trace":
                continue
            if any(part.startswith(".") for part in relative.parts) or path.is_symlink() or not path.is_file():
                continue
            if path.resolve() != path or run_dir not in path.resolve().parents:
                continue
            info = path.stat()
            items.append({"name": relative.as_posix(), "size": info.st_size,
                          "modified_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat()})
            if len(items) > 1024:
                raise ValueError("产物数量超过单次同步上限 1024")
    api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/runs/{payload['run_id']}/artifacts", "POST", {"items": items})


def delete_remote_files(config: dict, payload: dict) -> dict | None:
    # Serialize deletion with every training process using this workspace. This
    # also prevents a second Agent from claiming new work during validation.
    runs = payload["runs"]
    receipt_required = payload.get("receipt_required") is True
    expected_targets = deletion_plan_targets(runs) if receipt_required else []
    expected_digest = deletion_plan_digest(runs) if receipt_required else None
    if receipt_required and (
        payload.get("scope") != "task"
        or payload.get("receipt_version") != DELETE_RECEIPT_VERSION
        or payload.get("plan_digest") != expected_digest
        or payload.get("plan_targets") != expected_targets
    ):
        raise ValueError("远程删除计划摘要或回执要求无效")
    requested_roots = [Path(run["output_root"]).resolve() for run in runs]
    training_lock = acquire_training_lock(config, requested_roots)
    if training_lock is None:
        raise ValueError("训练工作区正在使用，已拒绝远程删除；请等待训练结束后重试")
    try:
        # Validate the complete plan before deleting the first file. rmtree never
        # follows directory symlinks inside an owned Attempt.
        plans = [
            (*operation_paths(config, run, allow_missing_root=True, allow_legacy_flat=True),
             run.get("layout", "canonical"))
            for run in runs
        ]
        target_roots = _normalized_output_roots(requested_roots)
        if workspace_delete_conflicts(config, target_roots):
            raise ValueError(
                "检测到另一个 Agent 也登记或正在使用相同训练输出目录，"
                "为防止跨平台误删已拒绝清理；请为每个平台配置独立的 training_output_roots"
            )
        required_files = payload.get("required_files", [])
        if not isinstance(required_files, list):
            raise ValueError("续训权重校验清单无效")
        required_paths = []
        for item in required_files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or type(item.get("size")) is not int or item["size"] <= 0:
                raise ValueError("续训权重校验清单无效")
            candidate = Path(item["path"])
            resolved = candidate.resolve()
            if (candidate != resolved or not any(root in resolved.parents for root in requested_roots)
                    or not candidate.is_file() or candidate.stat().st_size != item["size"]):
                raise ValueError("后续 Attempt 的独立权重缺失、变更或路径不可验证，已拒绝清理旧检查点")
            required_paths.append(resolved)
        protected = [Path(path).resolve() for path in payload.get("protected_paths", [])]
        protected.extend(required_paths)
        protected.extend([Path(config["training_cwd"]).resolve(), Path(config["yolo_bin"]).resolve()])
        deletion_targets = []
        for run, (run_dir, log, layout) in zip(runs, plans):
            if (run_dir.exists() and not run_dir.is_dir()) or (log and log.exists() and not log.is_file()):
                raise ValueError("Attempt 或日志的文件类型不正确，已拒绝清理")
            if layout == "legacy_flat":
                # The old first Attempt wrote directly into the task container.
                # Preserve every Attempt-shaped child, including orphaned rows,
                # and remove only root-level legacy output.
                owned = [
                    child for child in run_dir.iterdir()
                    if not re.fullmatch(r"attempt-\d{3,}", child.name)
                ] if run_dir.is_dir() else []
            else:
                owned = [run_dir]
            for target in owned:
                if target.parent.resolve() != run_dir.resolve() and layout == "legacy_flat":
                    raise ValueError("旧版 Attempt 清理目标超出任务目录")
                for path in protected:
                    if path == target or target in path.parents:
                        raise ValueError("任务目录包含仍被引用的模型、数据或训练程序，已拒绝清理")
            for path in protected:
                if log and path == log:
                    raise ValueError("任务目录包含仍被引用的模型、数据或训练程序，已拒绝清理")
            deletion_targets.append((run, run_dir, log, layout, owned))
        removed_targets = []
        absent_targets = []
        for run, run_dir, log, layout, owned in deletion_targets:
            output_key = f"{run['run_id']}:output"
            output_existed = bool(owned) if layout == "legacy_flat" else (run_dir.exists() or run_dir.is_symlink())
            log_existed = bool(log and (log.exists() or log.is_symlink()))
            for target in owned:
                if target.is_symlink() or target.is_file():
                    target.unlink(missing_ok=True)
                elif target.exists():
                    if not target.is_dir():
                        raise ValueError("Attempt 清理目标类型无效")
                    shutil.rmtree(target)
            if log:
                log.unlink(missing_ok=True)
            if layout == "legacy_flat":
                remaining_owned = [
                    child for child in run_dir.iterdir()
                    if not re.fullmatch(r"attempt-\d{3,}", child.name)
                ] if run_dir.is_dir() else []
                if remaining_owned:
                    raise OSError("旧版 Attempt 文件未完全清理")
            elif run_dir.exists() or run_dir.is_symlink():
                raise OSError("Attempt 目录未完全清理")
            if log and (log.exists() or log.is_symlink()):
                raise OSError("Attempt 日志未完全清理")
            (removed_targets if output_existed else absent_targets).append(output_key)
            if log:
                log_key = f"{run['run_id']}:log"
                (removed_targets if log_existed else absent_targets).append(log_key)
            cleanup_parents = {run_dir.parent, log.parent if log else run_dir.parent}
            if layout == "legacy_flat":
                # Remove the old task container only when it is now empty.  A
                # canonical attempt-NNN child makes rmdir fail and is retained.
                cleanup_parents.add(run_dir)
            for parent in cleanup_parents:
                try:
                    parent.rmdir()  # remove empty task directory only
                except OSError:
                    pass
        if receipt_required:
            covered = removed_targets + absent_targets
            if len(covered) != len(set(covered)) or set(covered) != set(expected_targets):
                raise OSError("远程删除结果未完整覆盖冻结计划")
            return {
                "version": DELETE_RECEIPT_VERSION,
                "plan_digest": expected_digest,
                "targets": expected_targets,
                "removed": removed_targets,
                "already_absent": absent_targets,
                "verified_absent": True,
            }
        return None
    finally:
        release_training_lock(training_lock)


def remote_model_path(config: dict, payload: dict) -> Path:
    """Validate a standalone model without granting access to training roots."""
    model_asset_id = payload.get("model_asset_id")
    if not isinstance(model_asset_id, str) or not re.fullmatch(r"[0-9a-f]{32}", model_asset_id):
        raise ValueError("远程模型资产标识无效")
    raw = payload.get("path")
    if (not isinstance(raw, str) or not raw or len(raw) > 4096 or "\x00" in raw or
            any(ord(character) < 32 for character in raw)):
        raise ValueError("远程模型路径无效")
    path = Path(raw)
    if not path.is_absolute() or path.suffix != ".pt":
        raise ValueError("远程模型必须是 model_test_roots 内的绝对 .pt 文件路径")
    configured = config.get("model_test_roots")
    if configured is None:
        configured = [str(Path(config["training_cwd"]) / "pt")]
    if not isinstance(configured, list) or not configured:
        raise ValueError("Agent 未配置可用的 model_test_roots")
    roots = []
    invalid_roots = []
    for value in configured:
        try:
            root = Path(value)
            resolved = root.resolve(strict=True)
        except (OSError, RuntimeError, TypeError, ValueError):
            invalid_roots.append(str(value))
            continue
        if (not root.is_absolute() or root.is_symlink() or resolved != root or
                not stat.S_ISDIR(os.stat(root, follow_symlinks=False).st_mode)):
            invalid_roots.append(str(value))
            continue
        roots.append(root)
    if not any(root in path.parents for root in roots):
        configured_text = "、".join(str(value) for value in configured)
        if invalid_roots:
            raise ValueError(
                f"远程模型路径不在可用的 model_test_roots 内：{path}；"
                f"当前配置：{configured_text}；请检查目录是否存在且不含符号链接"
            )
        raise ValueError(
            f"远程模型路径不在 Agent model_test_roots 内：{path}；"
            f"允许目录：{configured_text}"
        )
    try:
        resolved_path = path.resolve(strict=True)
        info = os.stat(path, follow_symlinks=False)
    except (FileNotFoundError, NotADirectoryError):
        raise ValueError(f"远程服务器上的模型文件不存在：{path}") from None
    except OSError as exc:
        raise ValueError(f"无法读取远程模型文件：{path}") from exc
    if path.is_symlink() or resolved_path != path:
        raise ValueError("远程模型路径或其父目录包含符号链接，已拒绝读取")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("远程模型路径不是普通文件")
    if info.st_size <= 0:
        raise ValueError("远程模型文件为空")
    return path


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns


def submit_operation_chunk(
    config: dict,
    endpoint: str,
    chunk: dict,
    stop: threading.Event,
) -> dict | None:
    """Submit one block, waiting safely when another center worker finalizes it."""
    final = bool(chunk.get("final"))
    timeout = (
        config.get(
            "model_transfer_finalize_timeout_seconds",
            MODEL_TRANSFER_FINALIZE_TIMEOUT_SECONDS,
        )
        if final else 20
    )
    while not stop.is_set():
        for retry in range(3):
            try:
                if final:
                    response = api_json(
                        config,
                        endpoint + "/chunk",
                        "POST",
                        chunk,
                        timeout_seconds=timeout,
                    )
                else:
                    response = api_json(config, endpoint + "/chunk", "POST", chunk)
                break
            except (OSError, URLError):
                if retry == 2 or stop.wait(2):
                    raise
        if not final or response.get("status") == "succeeded":
            return response
        if response.get("status") not in {"running", "finalizing"}:
            raise ValueError("中心未确认远程文件最终分块")
        if stop.wait(2):
            return None
    return None


def fetch_remote_model(config: dict, operation: dict, stop: threading.Event, endpoint: str) -> None:
    """Stream one immutable standalone checkpoint through the operation channel."""
    path = remote_model_path(config, operation["payload"])
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ValueError("远程模型不是非空普通文件")
        identity = _file_identity(info)
        try:
            current_path = os.stat(path, follow_symlinks=False)
            if path.resolve(strict=True) != path or _file_identity(current_path) != identity:
                raise ValueError("打开模型时路径发生变化，请重试")
        except OSError as exc:
            raise ValueError("打开模型时路径发生变化，请重试") from exc
        metadata = {
            "size": info.st_size,
            "modified_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
            "mtime_ns": info.st_mtime_ns,
        }
        digest, offset = hashlib.sha256(), 0
        while not stop.is_set():
            data = handle.read(512 * 1024)
            digest.update(data)
            final = offset + len(data) == info.st_size
            if _file_identity(os.fstat(handle.fileno())) != identity:
                raise ValueError("远程模型在传输期间发生变化，请重试")
            if final:
                try:
                    current_path = os.stat(path, follow_symlinks=False)
                    if path.resolve(strict=True) != path or _file_identity(current_path) != identity:
                        raise ValueError("远程模型在传输期间发生变化，请重试")
                except OSError as exc:
                    raise ValueError("远程模型在传输期间发生变化，请重试") from exc
            chunk = {
                "lease": operation["lease"], "offset": offset,
                "data": base64.b64encode(data).decode("ascii"), "final": final,
                "sha256": digest.hexdigest() if final else None,
                **metadata,
            }
            response = submit_operation_chunk(config, endpoint, chunk, stop)
            if response is None:
                return
            if response.get("status") == "succeeded":
                return
            offset += len(data)
            if not data:
                raise ValueError("远程模型读取不完整")
        # The lease will expire; a restarted Agent can safely retry from byte 0.


def fetch_remote_log(config: dict, operation: dict, stop: threading.Event, endpoint: str) -> None:
    """Stream the bytes present when a managed log is opened, allowing later appends."""
    payload = operation["payload"]
    _run_dir, path = operation_paths(config, payload)
    if path is None:
        raise ValueError("远程训练日志路径缺失")
    maximum = payload.get("max_bytes", LOG_SNAPSHOT_MAX_BYTES)
    if type(maximum) is not int or maximum < 0:
        raise ValueError("远程日志下载大小限制无效")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"远程训练日志无法打开：{path}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("远程训练日志不是普通文件")
        snapshot_size = opened.st_size
        if maximum and snapshot_size > maximum:
            raise ValueError(f"远程训练日志超过下载大小限制：{snapshot_size} > {maximum}")
        identity = (opened.st_dev, opened.st_ino, opened.st_mode)

        def verify_snapshot() -> None:
            current = os.fstat(handle.fileno())
            if ((current.st_dev, current.st_ino, current.st_mode) != identity
                    or not stat.S_ISREG(current.st_mode) or current.st_size < snapshot_size):
                raise ValueError("远程训练日志在传输期间被截断或替换，请重试")
            try:
                current_path = os.stat(path, follow_symlinks=False)
                if (path.resolve(strict=True) != path or path.is_symlink()
                        or (current_path.st_dev, current_path.st_ino, current_path.st_mode) != identity):
                    raise ValueError("远程训练日志在传输期间被替换，请重试")
            except OSError as exc:
                raise ValueError("远程训练日志在传输期间被替换，请重试") from exc

        verify_snapshot()
        metadata = {
            "size": snapshot_size,
            "modified_at": datetime.fromtimestamp(opened.st_mtime, timezone.utc).isoformat(),
            "mtime_ns": opened.st_mtime_ns,
        }
        digest, offset = hashlib.sha256(), 0
        while not stop.is_set():
            remaining = snapshot_size - offset
            data = handle.read(min(512 * 1024, remaining)) if remaining else b""
            if remaining and not data:
                raise ValueError("远程训练日志在传输期间被截断，请重试")
            digest.update(data)
            final = offset + len(data) == snapshot_size
            verify_snapshot()
            chunk = {
                "lease": operation["lease"], "offset": offset,
                "data": base64.b64encode(data).decode("ascii"), "final": final,
                "sha256": digest.hexdigest() if final else None,
                **metadata,
            }
            response = submit_operation_chunk(config, endpoint, chunk, stop)
            if response is None or response.get("status") == "succeeded":
                return
            offset += len(data)
        # The lease will expire; a restarted Agent can safely take a fresh snapshot.


def execute_operation(config: dict, operation: dict, stop: threading.Event) -> None:
    endpoint = f"/api/compute/nodes/{config['node_id']}/agent/operations/{operation['id']}"
    result = {"lease": operation["lease"], "success": True}
    try:
        payload = operation["payload"]
        if operation["kind"] == "delete":
            receipt = delete_remote_files(config, payload)
            if receipt is not None:
                result["receipt"] = receipt
        elif operation["kind"] == "scan":
            report_manifest(config, payload)
        elif operation["kind"] == "fetch":
            paths = operation_paths(config, payload)
            assert paths is not None
            run_dir, _ = paths
            relative = Path(payload["name"])
            if (relative.is_absolute() or relative.as_posix() != payload["name"] or
                    any(p in {".", ".."} or p.startswith(".") for p in relative.parts)):
                raise ValueError("产物路径无效")
            path = run_dir / relative
            if path.resolve() != path or not path.is_file():
                raise ValueError("产物不存在或包含符号链接")
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "rb") as handle:
                info = os.fstat(handle.fileno())
                expected_time = datetime.fromisoformat(payload["modified_at"]).timestamp()
                if info.st_size != payload["size"] or abs(info.st_mtime - expected_time) > 0.00001:
                    raise ValueError("产物已更新，请刷新列表后重试下载")
                digest, offset = hashlib.sha256(), 0
                while not stop.is_set():
                    data = handle.read(512 * 1024)
                    digest.update(data)
                    final = offset + len(data) == info.st_size
                    current = path.stat()
                    if final and (current.st_mtime_ns != info.st_mtime_ns or current.st_size != info.st_size or current.st_ino != info.st_ino):
                        raise ValueError("传输期间产物发生变化，请重试下载")
                    chunk = {"lease": operation["lease"], "offset": offset,
                             "data": base64.b64encode(data).decode("ascii"), "final": final,
                             "sha256": digest.hexdigest() if final else None}
                    # Repeat exactly the same block after a timeout; center verifies retries.
                    response = submit_operation_chunk(config, endpoint, chunk, stop)
                    if response is None:
                        return
                    if response.get("status") == "succeeded":
                        return
                    offset += len(data)
                    if not data:
                        raise ValueError("文件读取不完整")
                return  # lease expires and a restarted Agent can retry from zero
        elif operation["kind"] == "fetch_model":
            fetch_remote_model(config, operation, stop, endpoint)
            return
        elif operation["kind"] == "fetch_log":
            fetch_remote_log(config, operation, stop, endpoint)
            return
        elif operation["kind"] == "batch_history":
            _run_dir, log_path = operation_paths(config, payload, allow_legacy_flat=True)
            if log_path is None:
                raise ValueError("远程训练日志路径缺失")
            configured = payload.get("batch")
            if type(configured) is not int or not 1 <= configured <= 1024:
                raise ValueError("Batch 初始值无效")
            result["batch_history"] = read_batch_history(log_path, configured, full=True)
        else:
            raise ValueError("未知远程操作")
    except Exception as exc:
        result.update(success=False, error=str(exc)[:2000])
        LOGGER.warning("远程文件操作失败 %s：%s", operation["kind"], type(exc).__name__)
    # Lost ACKs are recovered by the lease timeout; deletion is idempotent.
    try:
        api_json(config, endpoint + "/result", "POST", result)
    except (OSError, ValueError, HTTPError, URLError):
        LOGGER.warning("远程操作结果尚未确认，联网后平台将重新下发：%s", operation["id"])


def run_operations(config: dict, operation: dict, stop: threading.Event) -> None:
    # Drain small scans/previews without adding one heartbeat interval per file.
    for index in range(32):
        if stop.is_set():
            return
        execute_operation(config, operation, stop)
        if index == 31:
            return
        try:
            operation = api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/operations/poll", "POST", {}).get("operation")
        except (OSError, ValueError, HTTPError, URLError):
            return
        if not operation:
            return


# BEGIN PORTABLE METRIC CORE (generated from backend/app/metric_files.py)
METRIC_FILES_SOURCE = '"""Incremental training CSV reads; standard library only for standalone Agents."""\nfrom __future__ import annotations\n\nimport csv\nimport math\nimport threading\nfrom collections import OrderedDict\nfrom pathlib import Path\n\n\nclass MetricCsvReader:\n    """Keep bounded per-file caches; incomplete trailing records wait for a newline."""\n\n    def __init__(self, maximum_files=128):\n        self.maximum_files = max(1, maximum_files)\n        self._files = OrderedDict()\n        self._lock = threading.RLock()\n\n    def read(self, path, *, final=False, incremental=False):\n        path = Path(path)\n        key = str(path.absolute())\n        with self._lock:\n            try:\n                info = path.stat()\n                signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)\n                state = self._files.pop(key, None)\n                with path.open(\'rb\') as stream:\n                    reset = state is None or state[\'signature\'][:2] != signature[:2]\n                    reset = reset or info.st_size < state[\'offset\']\n                    if state is not None and not reset:\n                        # An append must retain the already consumed suffix. Same-size\n                        # rewrites and resumed/truncated files start a fresh generation.\n                        reset = info.st_size <= state[\'signature\'][2] and signature != state[\'signature\']\n                        if not reset and state[\'anchor\']:\n                            stream.seek(state[\'offset\'] - len(state[\'anchor\']))\n                            reset = stream.read(len(state[\'anchor\'])) != state[\'anchor\']\n                        if not reset and state[\'signature\'] == signature and (not final or not state[\'partial\']):\n                            state[\'delta\'] = []\n                            self._files[key] = state\n                            return self._snapshot(state, incremental=incremental)\n                    if reset:\n                        state = {\'offset\': 0, \'header\': None, \'metrics\': [], \'completed\': 0,\n                                 \'latest\': {}, \'anchor\': b\'\', \'partial\': False, \'delta\': [],\n                                 \'generation\': (state[\'generation\'] + 1) if state else 1}\n                    state[\'delta\'] = []\n                    stream.seek(state[\'offset\'])\n                    state[\'partial\'] = False\n                    while True:\n                        start = stream.tell()\n                        line = stream.readline()\n                        if not line:\n                            break\n                        if not line.endswith(b\'\\n\') and not final:\n                            stream.seek(start)\n                            state[\'partial\'] = True\n                            break\n                        try:\n                            values = next(csv.reader([line.decode(\'utf-8-sig\', errors=\'replace\')]))\n                        except (csv.Error, StopIteration):\n                            continue\n                        if state[\'header\'] is None:\n                            state[\'header\'] = [value.strip() for value in values]\n                            continue\n                        # CSV output contains one numeric record per physical line.\n                        # Reject truncated/extra fields rather than publishing half an epoch.\n                        if len(values) != len(state[\'header\']):\n                            continue\n                        metric = self._metric(dict(zip(state[\'header\'], values)))\n                        if metric is not None:\n                            state[\'metrics\'].append(metric)\n                            state[\'delta\'].append(metric)\n                            state[\'completed\'] = max(state[\'completed\'], metric[\'epoch\'])\n                            state[\'latest\'] = metric[\'values\']\n                    state[\'offset\'] = stream.tell()\n                    stream.seek(max(0, state[\'offset\'] - 256))\n                    state[\'anchor\'] = stream.read(min(256, state[\'offset\']))\n                state[\'signature\'] = signature\n                self._files[key] = state\n                while len(self._files) > self.maximum_files:\n                    self._files.popitem(last=False)\n                return self._snapshot(state, incremental=incremental)\n            except (OSError, csv.Error):\n                # Missing files can be normal during trainer startup/rotation.\n                return {\'completed\': 0, \'latest\': {}, \'metrics\': [], \'generation\': 0}\n\n    @staticmethod\n    def _snapshot(state, *, incremental=False):\n        return {\'completed\': state[\'completed\'], \'latest\': dict(state[\'latest\']),\n                \'metrics\': list(state[\'delta\'] if incremental else state[\'metrics\']),\n                \'generation\': state[\'generation\']}\n\n    @staticmethod\n    def _metric(row):\n        try:\n            epoch_number = float(row.get(\'epoch\', \'\'))\n            if not math.isfinite(epoch_number):\n                return None\n            epoch = int(epoch_number)\n        except (ValueError, TypeError, OverflowError):\n            return None\n        values = {}\n        for key, raw in row.items():\n            if not key or key in {\'epoch\', \'time\'}:\n                continue\n            if raw is None or raw == \'\':\n                values[key] = None\n                continue\n            try:\n                number = float(raw)\n                values[key] = number if math.isfinite(number) else None\n            except (ValueError, TypeError, OverflowError):\n                values[key] = str(raw)[:200]\n        try:\n            elapsed = float(row.get(\'time\') or 0)\n            if not math.isfinite(elapsed) or elapsed < 0:\n                elapsed = 0.0\n        except (ValueError, TypeError, OverflowError):\n            elapsed = 0.0\n        return {\'epoch\': epoch, \'elapsed\': elapsed, \'values\': values}\n'
METRIC_FILES = {}
exec(compile(METRIC_FILES_SOURCE, "<metric_files>", "exec"), METRIC_FILES)
_METRIC_CSV_READER = METRIC_FILES["MetricCsvReader"](maximum_files=4)
_REMOTE_CSV_GENERATIONS = {}
_REMOTE_CSV_GENERATION_LOCK = threading.Lock()
# END PORTABLE METRIC CORE


_BATCH_REDUCTION = re.compile(r"\bwith batch=(\d+)\.\s+Reducing to batch=(\d+) and retrying\b", re.I)
_BATCH_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_BATCH_LOG_CACHE = {}
_BATCH_LOG_LOCK = threading.Lock()


def _append_batch_reductions(history, lines, configured):
    result = list(history or [])
    for line in lines:
        match = _BATCH_REDUCTION.search(_BATCH_ANSI.sub("", line))
        if not match:
            continue
        old, new = map(int, match.groups())
        if not 1 <= new < old <= 1024:
            continue
        if not result:
            if configured > 0 and old != configured:
                continue
            result = [old]
        if result[-1] == new:
            continue
        if result[-1] == old and len(result) < 8:
            result.append(new)
    return result if len(result) > 1 else []


def read_batch_history(path: Path, configured: int, *, full: bool = False) -> list[int]:
    """Read appended lines during training, or scan historical log once in place."""
    if full:
        # Historical scans can be large. Never hold the live-report cache lock.
        history = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                history = _append_batch_reductions(history, (line,), configured)
        return history
    key = str(path)
    with _BATCH_LOG_LOCK:
        try:
            with path.open("rb") as handle:
                info = os.fstat(handle.fileno())
                state = _BATCH_LOG_CACHE.get(key)
                if not state or state[0] != info.st_dev or state[1] != info.st_ino or info.st_size < state[2]:
                    state = (info.st_dev, info.st_ino, 0, [])
                handle.seek(state[2])
                history = list(state[3])
                while True:
                    start = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    if not raw.endswith(b"\n"):
                        handle.seek(start)
                        break
                    history = _append_batch_reductions(history, [raw.decode("utf-8", errors="replace")], configured)
                if len(_BATCH_LOG_CACHE) >= 256 and key not in _BATCH_LOG_CACHE:
                    _BATCH_LOG_CACHE.pop(next(iter(_BATCH_LOG_CACHE)))
                _BATCH_LOG_CACHE[key] = (info.st_dev, info.st_ino, handle.tell(), history)
                return history
        except OSError:
            return list(_BATCH_LOG_CACHE.get(key, (None, None, None, []))[3])


def read_remote_progress(run_dir: Path, *, final: bool = False,
                         sent_epochs: set[int] | None = None,
                         pending_metrics: dict[int, dict] | None = None) -> tuple[int, dict, list[dict]]:
    # Monitoring consumes deltas; outcome/recovery readers need a non-consuming
    # full snapshot even when progress was read earlier in this process.
    snapshot = _METRIC_CSV_READER.read(run_dir / "results.csv", final=final,
                                     incremental=sent_epochs is not None)
    key = str((run_dir / "results.csv").absolute())
    with _REMOTE_CSV_GENERATION_LOCK:
        previous = _REMOTE_CSV_GENERATIONS.get(key)
        _REMOTE_CSV_GENERATIONS[key] = snapshot['generation']
    if previous is not None and previous != snapshot['generation'] and sent_epochs is not None:
        sent_epochs.clear()
        if pending_metrics is not None:
            pending_metrics.clear()
    if sent_epochs is not None:
        # Appended records may correct an acknowledged epoch without replacing
        # the file; each new physical record must be eligible for delivery.
        sent_epochs.difference_update(item['epoch'] for item in snapshot['metrics'])
    return snapshot["completed"], snapshot["latest"], snapshot["metrics"]


def recent_log_lines(path: Path) -> list[str]:
    # Bound disk reads as well as payload size for verbose, long-running jobs.
    text = recent_log_text(path, maximum_bytes=1024 * 1024)
    return [line[:4000] for line in text.splitlines()[-100:]]


def recent_log_text(path: Path, maximum_bytes: int = 2 * 1024 * 1024) -> str:
    """Read a bounded binary tail so durable completion markers survive verbose final validation."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - maximum_bytes))
            return handle.read(maximum_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def save_pending_final(config: dict, run_id: str, payload: dict) -> Path:
    """Keep terminal results on disk until the center acknowledges them; no credentials."""
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("运行 ID 无效")
    directory = Path(config["training_state_dir"])
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{run_id}.json"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return path


def deliver_pending_final(config: dict, path: Path) -> bool:
    run_id = path.stem
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("运行 ID 无效")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") not in {"completed", "stopped", "failed", "interrupted"}:
        raise ValueError("本地最终状态无效")
    metrics = payload.get("metrics", [])
    batch = dict(payload, metrics=metrics[:256])
    if len(metrics) > 256:
        batch["status"] = "stopping" if payload["status"] == "stopped" else "running"
        for key in ("error_message", "exit_code", "stop_reason"):
            batch.pop(key, None)
    endpoint = f"/api/compute/nodes/{config['node_id']}/agent/runs/{run_id}/update"
    lease = payload.get("_claim_lease")
    batch.pop("_claim_lease", None)
    response = (api_json(config, endpoint, "POST", batch, {"X-Training-Lease": lease})
                if lease else api_json(config, endpoint, "POST", batch))
    if len(metrics) > 256 and response.get("status") not in {"completed", "stopped", "failed", "interrupted"}:
        save_pending_final(config, run_id, dict(payload, metrics=metrics[256:]))
        return False
    clear_active_run(config, run_id)
    path.unlink()
    return True


def retry_pending_finals(config: dict) -> bool:
    """Recover finished runs after an Agent restart before accepting more work."""
    ready = True
    for path in sorted(Path(config["training_state_dir"]).glob("*.json")):
        try:
            ready = deliver_pending_final(config, path) and ready
        except (OSError, ValueError, HTTPError, URLError):
            LOGGER.warning("最终训练结果仍待补传：%s；请检查网络、节点配置及本地状态文件", path.stem)
            ready = False
    return ready


def prepare_trace_command(config: dict, task: dict, command: list[str]) -> list[str]:
    """Persist the real training outcome, with optional detection trace collection."""
    if len(command) < 3 or command[1] not in {"detect", "segment", "classify", "pose", "obb"} or command[2] != "train":
        return command
    trace_dir = Path(task["run_dir"]) / "training-trace"
    state_dir = Path(config["training_state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    runner = state_dir / "training_trace_runner.py"
    if not runner.is_file() or runner.read_text(encoding="utf-8") != REMOTE_TRACE_RUNNER:
        temporary = runner.with_name(f".{runner.name}.{threading.get_ident()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(REMOTE_TRACE_RUNNER)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o700)
            temporary.replace(runner)
        finally:
            temporary.unlink(missing_ok=True)
    yolo_bin = str(config.get("yolo_bin", ""))
    python_bin = str(Path(yolo_bin).parent / "python") if "/" in yolo_bin else sys.executable
    if not Path(python_bin).is_file():
        python_bin = sys.executable
    trace_flags = ["--trace-enabled"] if config.get("enable_training_trace", False) and task.get("task_type") == "detect" else []
    return [python_bin, str(runner), "--trace-dir", str(trace_dir), "--run-dir", str(task["run_dir"]), *trace_flags,
            "--maximum", str(TRACE_MAXIMUM), "--interval", str(TRACE_INTERVAL),
            "--command-json", json.dumps(command, ensure_ascii=False), *command]


def report_training_trace(config: dict, task: dict, sent_images: set[str]) -> None:
    trace_path = Path(task["run_dir"]) / "training-trace" / "trace.json"
    if not trace_path.is_file():
        return
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    payload["samples"] = payload.get("samples", [])[-TRACE_MAXIMUM:]
    image_name = next((sample.get("image") for sample in payload["samples"] if sample.get("image") and sample.get("image") not in sent_images), None)
    image_base64 = None
    if image_name:
        image_path = trace_path.parent / Path(image_name).name
        if image_path.is_file() and image_path.stat().st_size <= 512 * 1024:
            image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        else:
            image_name = None
    api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/runs/{task['run_id']}/trace", "POST",
             {"trace": payload, "image_name": image_name, "image_base64": image_base64})
    if image_name:
        sent_images.add(image_name)


class ClaimRejected(RuntimeError):
    pass


class ClaimConfirmationPending(RuntimeError):
    pass


def confirm_remote_claim(config: dict, task: dict, stop: threading.Event) -> bool:
    """Confirm a lease idempotently; retry only outcomes that may hide a committed ACK."""
    lease = task.get("claim_lease")
    if not lease:
        return False
    endpoint = f"/api/compute/nodes/{config['node_id']}/agent/runs/{task['run_id']}/confirm"
    attempts = len(CONFIRM_RETRY_DELAYS) + 1
    last_error = None
    for attempt in range(attempts):
        try:
            response = api_json(config, endpoint, "POST", {"lease": lease})
            if response.get("confirmed") is not True:
                raise ValueError("平台未返回有效的确认结果")
            return bool(response.get("stop"))
        except HTTPError as exc:
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise ClaimRejected(f"平台拒绝训练领取（HTTP {exc.code}）") from None
            last_error = exc
        except (OSError, URLError, ValueError) as exc:
            last_error = exc
        if attempt == attempts - 1:
            break
        # A stop during an ambiguous ACK must not prevent idempotent retry: the
        # center may already have committed the confirmation. Retry immediately
        # on shutdown so a confirmed claim can be closed as stopped.
        if not stop.is_set():
            stop.wait(CONFIRM_RETRY_DELAYS[attempt])
    raise ClaimConfirmationPending("训练领取确认尚未获得可靠响应") from last_error


def _save_terminal_for_retry(config: dict, task: dict, payload: dict) -> Path:
    payload = training_report_payload(task, payload)
    path = save_pending_final(config, task["run_id"], {"_claim_lease": task.get("claim_lease"), **payload})
    clear_active_run(config, task["run_id"])
    return path


def training_report_payload(task: dict, payload: dict) -> dict:
    """Old centers have strict schemas; only advertise negotiated additions."""
    if task.get('batch_history_version') == 1:
        payload = {**payload, 'batch_history': read_batch_history(
            Path(task['log_path']), int(task['parameters']['batch']))}
    if task.get('dataset_provenance_version') == 1 and task.get('dataset_snapshot'):
        return {**payload, 'dataset_snapshot': task['dataset_snapshot']}
    return payload


def _deliver_terminal_once(config: dict, path: Path) -> None:
    try:
        deliver_pending_final(config, path)
    except (OSError, ValueError, HTTPError, URLError):
        LOGGER.warning("最终训练状态已落盘，将在联网后重试：%s", path.stem)


def _terminate_recovered_process(identity: dict, *, force: bool = False) -> None:
    if not process_identity_matches(identity):
        return
    try:
        os.killpg(identity["pid"], signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def reap_finished_training(identity: dict | None) -> None:
    """Exec retains parenthood, so reap only this verified, already-dead child."""
    if not isinstance(identity, dict) or type(identity.get("pid")) is not int:
        return
    try:
        fields = Path(f"/proc/{identity['pid']}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
        if (fields[0] != "Z" or int(fields[1]) != os.getpid()
                or int(fields[19]) != identity.get("start_ticks")
                or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != identity.get("boot_id")):
            return
        os.waitpid(identity["pid"], os.WNOHANG)
    except (OSError, ValueError, IndexError, ChildProcessError):
        pass


def _recovery_identity(state: dict) -> dict | None:
    identity = state.get("process")
    if process_identity_matches(identity):
        return identity
    if isinstance(identity, dict) and type(identity.get("pid")) is int:
        current = process_identity(identity["pid"])
        if current and all(current.get(key) == identity.get(key) for key in ("pid", "start_ticks", "boot_id")):
            command = state.get("command")
            if isinstance(command, list) and _argv_matches_command(_process_argv(identity["pid"]), command):
                return current
            raise ValueError("原 PID 仍存活但命令行身份已变化，已保留活动记录并阻止新任务")
    if state.get("phase") != "launching" or not isinstance(state.get("command"), list):
        return None
    matches = find_process_for_command(state["command"])
    if len(matches) > 1:
        raise ValueError("发现多个与未完成启动记录匹配的进程，已阻止自动接管")
    return matches[0] if matches else None


def read_training_result(task: dict) -> dict | None:
    """Accept only the terminal record belonging to this exact Run."""
    path = Path(task["run_dir"]) / "training-result.json"
    try:
        if path.is_symlink() or path.stat().st_size > 256 * 1024:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or payload.get("run_id") != task.get("run_id")
                or payload.get("task_id") != task.get("task_id")
                or payload.get("status") not in {"completed", "failed"}
                or type(payload.get("exit_code")) is not int
                or type(payload.get("completed_epochs")) is not int
                or payload["completed_epochs"] < 0
                or type(payload.get("expected_epochs")) is not int
                or payload["expected_epochs"] <= 0
                or payload.get("reason") not in {"early_stopping", "epochs_completed", "training_completed", "exception"}):
            return None
        finished = datetime.fromisoformat(payload["finished_at"].replace("Z", "+00:00"))
        if finished.tzinfo is None or finished > datetime.now(timezone.utc) + timedelta(seconds=10):
            return None
        for key in ("best_epoch", "patience"):
            if key in payload and (type(payload[key]) is not int or payload[key] <= 0):
                return None
        if payload.get("best_epoch", 0) > payload["completed_epochs"]:
            return None
        if payload.get("error_message") is not None and not isinstance(payload["error_message"], str):
            return None
        if task.get("started_at"):
            started = datetime.fromisoformat(str(task["started_at"]).replace("Z", "+00:00"))
            started = started.replace(tzinfo=timezone.utc) if started.tzinfo is None else started
            if finished < started or path.stat().st_mtime < started.timestamp() - 1:
                return None
        if ((payload["status"] == "completed") != (payload["exit_code"] == 0)
                or (payload["status"] == "failed") != (payload["reason"] == "exception")):
            return None
        return payload
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _record_completion_info(record: dict) -> dict:
    info = {"source": "training_result", "reason": record["reason"],
            "completed_epochs": record["completed_epochs"], "expected_epochs": record["expected_epochs"],
            "finished_at": record["finished_at"]}
    for key in ("best_epoch", "patience"):
        if type(record.get(key)) is int and record[key] > 0:
            info[key] = record[key]
    final_metrics = record.get("final_metrics")
    if isinstance(final_metrics, dict):
        info["final_metrics"] = {str(key)[:200]: value for key, value in final_metrics.items()
                                 if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)}
    return info


def _legacy_completion_info(task: dict, completed: int, log_lines: list[str]) -> dict:
    """Require final validation/save evidence, rather than an early-stop announcement."""
    try:
        expected = max(1, int(task.get("parameters", {}).get("epochs") or 1))
    except (TypeError, ValueError):
        expected = 1
    run_dir = Path(task["run_dir"])
    weights = run_dir / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    text = recent_log_text(Path(task["log_path"])) or "\n".join(log_lines)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "\n")
    summaries = list(re.finditer(r"(?im)^\s*(\d+)\s+epochs? completed in\b", text))
    if not summaries or completed <= 0:
        return {}
    summary = summaries[-1]
    suffix = text[summary.end():]
    # A failure during final validation must dominate the earlier early-stop notice.
    if re.search(r"(?im)Traceback \(most recent call last\):|^\s*(?:[\w.]*Error|[\w.]*Exception):|CUDA error:|CUDA out of memory|^\s*Killed\s*$", suffix):
        return {}
    saves = list(re.finditer(r"(?im)^\s*Results saved to\s+([^\n]+)", suffix))
    if not saves:
        return {}
    try:
        saved_path = Path(saves[-1].group(1).strip().strip("'\""))
        actual_completed, _, rows = read_remote_progress(run_dir)
        first_epoch = min((row["epoch"] for row in rows), default=0)
        summary_epochs = int(summary.group(1))
        if (saved_path.resolve() != run_dir.resolve() or actual_completed != completed
                or summary_epochs not in {completed, completed - first_epoch + 1}
                or not best.is_file() or best.stat().st_size <= 0
                or not last.is_file() or last.stat().st_size <= 0):
            return {}
    except (OSError, ValueError):
        return {}
    early = list(re.finditer(r"(?im)EarlyStopping:.*?(?:\n[^\n]*){0,3}", text[:summary.start()]))
    info = {"source": "legacy_log", "reason": "early_stopping" if early and completed < expected else
            "epochs_completed" if completed >= expected else "training_completed",
            "completed_epochs": completed, "expected_epochs": expected}
    if early:
        notice = early[-1].group(0)
        for key, pattern in (("best_epoch", r"best results observed at epoch\s+(\d+)"),
                             ("patience", r"(?:last\s+(\d+)\s+epochs|patience\s*=\s*(\d+))")):
            match = re.search(pattern, notice, re.I)
            if match:
                info[key] = int(next(group for group in match.groups() if group is not None))
    return info


def training_completion_info(task: dict, completed: int, log_lines: list[str]) -> dict:
    record = read_training_result(task)
    return _record_completion_info(record) if record else _legacy_completion_info(task, completed, log_lines)


def recovered_natural_exit(task: dict, completed: int, log_lines: list[str]) -> tuple[str, str | None, int | None]:
    """Recover the durable outcome without inventing an orphan's OS exit code."""
    record = read_training_result(task)
    if record:
        return record["status"], str(record.get("error_message") or "")[:4000] or None, record["exit_code"]
    if _legacy_completion_info(task, completed, log_lines):
        return "completed", None, None
    return ("interrupted",
            "Agent 重启后接管的训练进程已结束，但无法取得原进程退出码；已保留产物供核对",
            None)


def recover_remote_task(config: dict, state: dict, stop: threading.Event, training_lock=None,
                        process_holds_lock: bool = False) -> None:
    """Monitor an orphaned, identity-verified process without starting a duplicate."""
    lock = training_lock
    release_lock = lock is not None
    task = state["task"]
    if lock is None and not process_holds_lock:
        try:
            lock = acquire_task_training_lock(config, task)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            identity = _recovery_identity(state)
            if identity is not None:
                LOGGER.error("无法验证恢复任务 %s 的本地资源锁，保留进程与活动记录：%s",
                             task.get("run_id"), exc)
                return
            completed, latest, metrics = read_remote_progress(Path(task["run_dir"]))
            latest = latest or task.get("latest_metrics", {})
            path = _save_terminal_for_retry(config, task, {
                "status": "failed", "error_message": f"恢复训练时无法验证 GPU/工作区资源分配：{str(exc)[:3500]}",
                "exit_code": 1, "completed_epochs": max(completed, task.get("completed_epochs", 0)),
                "progress": task.get("progress", 0), "latest_metrics": latest, "metrics": metrics,
                "log_tail": recent_log_lines(Path(task["log_path"])),
                "command": state.get("command") or [], "run_dir": task.get("run_dir")})
            _deliver_terminal_once(config, path)
            return
        release_lock = lock is not None
    if lock is None and not process_holds_lock:
        identity = _recovery_identity(state)
        if identity is not None:
            LOGGER.error("Run %s 的进程仍在运行，但 Agent 无法确认其完整资源锁；保留活动状态并拒绝重复启动",
                         task.get("run_id"))
            return
        completed, latest, metrics = read_remote_progress(Path(task["run_dir"]), final=True)
        latest = latest or task.get("latest_metrics", {})
        path = _save_terminal_for_retry(config, task, {
            "status": "failed", "error_message": "恢复训练时 GPU 或工作区资源锁已被占用；为避免重复训练，未重新启动",
            "exit_code": 1, "completed_epochs": max(completed, task.get("completed_epochs", 0)),
            "progress": task.get("progress", 0), "latest_metrics": latest, "metrics": metrics,
            "log_tail": recent_log_lines(Path(task["log_path"])),
            "command": state.get("command") or [], "run_dir": task.get("run_dir")})
        _deliver_terminal_once(config, path)
        return
    run_id = task["run_id"]
    endpoint = f"/api/compute/nodes/{config['node_id']}/agent/runs/{run_id}"
    try:
        identity = _recovery_identity(state)
        holds_expected_locks = False
        if identity:
            try:
                holds_expected_locks = (process_holds_task_locks(config, task, identity["pid"])
                                        if "gpu_assignment" in task else
                                        process_holds_training_lock(config, identity["pid"]))
            except (OSError, ValueError, TypeError):
                holds_expected_locks = False
        if process_holds_lock and (identity is None or not holds_expected_locks):
            process_holds_lock = False
            lock = acquire_task_training_lock(config, task)
            release_lock = lock is not None
            if lock is None:
                LOGGER.warning("活动训练进程已不再持有共享锁，且锁已被其他 Agent 占用，暂不接管")
                return
        identity = _recovery_identity(state)
        try:
            control = api_json(config, endpoint + "/control")
        except (OSError, ValueError, HTTPError, URLError):
            control = {}
        if control.get("status") in {"completed", "failed", "stopped", "interrupted"}:
            if identity:
                _terminate_recovered_process(identity)
            clear_active_run(config, run_id)
            return
        if identity is None:
            completed, latest, metrics = read_remote_progress(Path(task["run_dir"]))
            completed = max(completed, task.get("completed_epochs", 0))
            latest = latest or task.get("latest_metrics", {})
            log_lines = recent_log_lines(Path(task["log_path"]))
            requested_stop = bool(control.get("stop"))
            if requested_stop:
                status, error, exit_code = "stopped", None, None
            else:
                status, error, exit_code = recovered_natural_exit(task, completed, log_lines)
            try:
                report_manifest(config, {**task, "output_root": task["parameters"]["remote_output_root"]})
                report_training_trace(config, task, set())
            except (OSError, ValueError, HTTPError, URLError):
                LOGGER.warning("离线期间结束的任务产物暂未同步，可由平台重新请求扫描")
            path = _save_terminal_for_retry(config, task, {
                "status": status, "error_message": error, "exit_code": exit_code,
                "completion_info": {} if requested_stop else training_completion_info(task, completed, log_lines),
                "stop_reason": "platform_stop" if requested_stop else None,
                "completed_epochs": completed,
                "progress": (100 if status == "completed" else
                             min(99.99, completed / max(1, int(task["parameters"]["epochs"])) * 100)),
                "latest_metrics": latest, "metrics": metrics, "log_tail": log_lines,
                "command": state.get("command") or [], "run_dir": task.get("run_dir")})
            _deliver_terminal_once(config, path)
            return
        command = state.get("command") or []
        save_active_run(config, task, "running", command=command, identity=identity)
        LOGGER.info("已安全接管 Agent 重启前的训练进程 %s（PID %s）", run_id, identity["pid"])
        requested_stop = bool(control.get("stop"))
        stop_reason = "platform_stop" if requested_stop else None
        sent_epochs: set[int] = set()
        pending_metrics: dict[int, dict] = {}
        sent_trace_images: set[str] = set()
        last_artifacts_at = 0.0
        while process_identity_matches(identity) and not stop.is_set():
            if requested_stop:
                _terminate_recovered_process(identity)
            try:
                control = api_json(config, endpoint + "/control")
                if control.get("stop"):
                    requested_stop, stop_reason = True, "platform_stop"
                    _terminate_recovered_process(identity)
            except (OSError, ValueError, HTTPError, URLError):
                pass
            completed, latest, metrics = read_remote_progress(
                Path(task["run_dir"]), sent_epochs=sent_epochs, pending_metrics=pending_metrics,
            )
            completed = max(completed, task.get("completed_epochs", 0))
            latest = latest or task.get("latest_metrics", {})
            pending_metrics.update((item["epoch"], item) for item in metrics)
            new_metrics = [pending_metrics[epoch] for epoch in sorted(pending_metrics)
                           if epoch not in sent_epochs][:256]
            if time.monotonic() - last_artifacts_at >= 30:
                try:
                    report_manifest(config, {**task, "output_root": task["parameters"]["remote_output_root"]})
                    report_training_trace(config, task, sent_trace_images)
                except (OSError, ValueError, HTTPError, URLError):
                    pass
                last_artifacts_at = time.monotonic()
            payload = {"status": "stopping" if requested_stop else "running", "pid": identity["pid"],
                       "completed_epochs": completed,
                       "progress": min(99.99, completed / max(1, int(task["parameters"]["epochs"])) * 100),
                       "latest_metrics": latest, "metrics": new_metrics,
                       "log_tail": recent_log_lines(Path(task["log_path"])), "run_dir": task["run_dir"]}
            if command:
                payload["command"] = command
            payload = training_report_payload(task, payload)
            try:
                response = api_json(config, endpoint + "/update", "POST", payload,
                                    {"X-Training-Lease": task["claim_lease"]}) if task.get("claim_lease") else api_json(
                                        config, endpoint + "/update", "POST", payload)
                sent_epochs.update(item["epoch"] for item in new_metrics)
                for item in new_metrics:
                    pending_metrics.pop(item["epoch"], None)
                if response.get("stop_requested"):
                    requested_stop, stop_reason = True, "platform_stop"
            except (OSError, ValueError, HTTPError, URLError):
                pass
            if requested_stop:
                break
            stop.wait(2)
        if stop.is_set() and not requested_stop and getattr(stop, "preserve_training", False) and process_identity_matches(identity):
            detach_training(config, task, command, identity)
        if stop.is_set() and not requested_stop and process_identity_matches(identity):
            requested_stop, stop_reason = True, "agent_shutdown"
            _terminate_recovered_process(identity)
        if requested_stop:
            _terminate_recovered_process(identity)
        deadline = time.monotonic() + config.get("training_stop_timeout_seconds", 15)
        while process_identity_matches(identity) and time.monotonic() < deadline:
            time.sleep(0.2)
        if process_identity_matches(identity):
            _terminate_recovered_process(identity, force=True)
            kill_deadline = time.monotonic() + 5
            while process_identity_matches(identity) and time.monotonic() < kill_deadline:
                time.sleep(0.1)
            if process_identity_matches(identity):
                raise RuntimeError("强制停止后进程仍存活，保留活动记录以便重试")
        completed, latest, metrics = read_remote_progress(
            Path(task["run_dir"]), final=True, sent_epochs=sent_epochs, pending_metrics=pending_metrics,
        )
        pending_metrics.update((item["epoch"], item) for item in metrics)
        completed = max(completed, task.get("completed_epochs", 0))
        latest = latest or task.get("latest_metrics", {})
        log_lines = recent_log_lines(Path(task["log_path"]))
        if requested_stop:
            status, error, exit_code = "stopped", None, None
        else:
            status, error, exit_code = recovered_natural_exit(task, completed, log_lines)
        try:
            report_manifest(config, {**task, "output_root": task["parameters"]["remote_output_root"]})
            report_training_trace(config, task, sent_trace_images)
        except (OSError, ValueError, HTTPError, URLError):
            LOGGER.warning("接管任务产物暂未同步，可由平台重新请求扫描")
        path = _save_terminal_for_retry(config, task, {
            "status": status, "error_message": error, "exit_code": exit_code,
            "completion_info": {} if requested_stop else training_completion_info(task, completed, log_lines),
            "stop_reason": stop_reason, "completed_epochs": completed,
            "progress": 100 if status == "completed" else min(99.99, completed / max(1, int(task["parameters"]["epochs"])) * 100),
            "latest_metrics": latest,
            "metrics": [pending_metrics[epoch] for epoch in sorted(pending_metrics) if epoch not in sent_epochs],
            "log_tail": log_lines, "command": command,
            "run_dir": task["run_dir"]})
        _deliver_terminal_once(config, path)
    except TrainingDetached:
        if lock is not None:
            lock.close()  # Close our copy only; the training child keeps its locks.
        release_lock = False
    except Exception:
        LOGGER.exception("恢复远程训练监控失败，活动状态已保留")
    finally:
        reap_finished_training(state.get("process"))
        if release_lock:
            release_training_lock(lock)


def _run_remote_task(config: dict, task: dict, stop: threading.Event, training_lock) -> None:
    run_id = task["run_id"]
    lease = task.get("claim_lease")
    manifest_payload = {**task, "output_root": task["parameters"]["remote_output_root"]}
    last_manifest_at = 0.0
    endpoint = f"/api/compute/nodes/{config['node_id']}/agent/runs/{run_id}"
    log_lines: list[str] = []
    process = None
    requested_stop = False
    stop_reason = None
    last_warning = 0.0
    last_metrics: set[int] = set()
    pending_metrics: dict[int, dict] = {}
    command_reported = False
    sent_trace_images: set[str] = set()
    last_trace_at = 0.0

    def request_stop(reason: str) -> None:
        nonlocal requested_stop, stop_reason
        if not requested_stop:
            requested_stop = True
            stop_reason = reason
            LOGGER.info("停止远程训练 %s，原因：%s", run_id, reason)
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def report_progress(payload: dict) -> None:
        nonlocal command_reported, last_warning
        pending_metrics.update((item["epoch"], item) for item in payload.get("metrics", []))
        payload = dict(payload, metrics=[pending_metrics[epoch] for epoch in sorted(pending_metrics)
                                         if epoch not in last_metrics][:256])
        payload = training_report_payload(task, payload)
        if not command_reported:
            payload["command"] = command
        try:
            response = (api_json(config, endpoint + "/update", "POST", payload,
                                 {"X-Training-Lease": lease})
                        if lease else api_json(config, endpoint + "/update", "POST", payload))
        except (OSError, ValueError, HTTPError, URLError):
            if time.monotonic() - last_warning >= 30 or not last_warning:
                LOGGER.warning("训练 %s 状态上报失败；训练继续运行，日志与指标将在连接恢复后补传", run_id)
                last_warning = time.monotonic()
            return
        command_reported = True
        last_metrics.update(item["epoch"] for item in payload.get("metrics", []))
        for item in payload.get("metrics", []):
            pending_metrics.pop(item["epoch"], None)
        if response.get("stop_requested"):
            request_stop("platform_stop")

    try:
        command = build_training_command(config, task)
        command = training_command_for_assignment(command, task)
        command = prepare_trace_command(config, task, command)
        if task.get('dataset_provenance_version') == 1:
            task['dataset_snapshot'] = DATASET_PROVENANCE['capture'](
                task['data_path'], task_type=task['task_type'], node_id=config['node_id'],
            )
        save_active_run(config, task, "launching", command=command)
        log_path = Path(task["log_path"])
        with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] REMOTE COMMAND {shlex.join(command)}\n")
            environment = training_launch_environment(config, task)
            inherited_locks = (training_lock.filenos() if isinstance(training_lock, LockBundle)
                               else (training_lock.fileno(),))
            process = subprocess.Popen(command, cwd=config["training_cwd"], env=environment, stdout=log_handle,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True,
                                       pass_fds=inherited_locks)
            identity = process_identity(process.pid)
            if identity:
                save_active_run(config, task, "running", command=command, identity=identity)
            report_progress({"status": "running", "pid": process.pid, "run_dir": task["run_dir"],
                             "completed_epochs": task.get("completed_epochs", 0),
                             "progress": task.get("progress", 0), "latest_metrics": task.get("latest_metrics", {})})
            while process.poll() is None and not stop.is_set() and not requested_stop:
                if stop.wait(2):
                    break
                try:
                    control = api_json(config, endpoint + "/control")
                    if control.get("stop"):
                        request_stop("platform_stop")
                except (OSError, ValueError, HTTPError, URLError):
                    pass
                if requested_stop:
                    break
                completed, latest, metrics = read_remote_progress(
                    Path(task["run_dir"]), sent_epochs=last_metrics, pending_metrics=pending_metrics,
                )
                completed = max(completed, task.get("completed_epochs", 0))
                latest = latest or task.get("latest_metrics", {})
                if time.monotonic() - last_manifest_at >= 30:
                    try:
                        report_manifest(config, manifest_payload)
                        last_manifest_at = time.monotonic()
                    except (OSError, ValueError, HTTPError, URLError):
                        last_manifest_at = time.monotonic()
                pending_metrics.update((item["epoch"], item) for item in metrics)
                new_metrics = [pending_metrics[epoch] for epoch in sorted(pending_metrics)
                               if epoch not in last_metrics][:256]
                if time.monotonic() - last_trace_at >= 10:
                    try:
                        report_training_trace(config, task, sent_trace_images)
                    except (OSError, ValueError, HTTPError, URLError, json.JSONDecodeError):
                        pass
                    last_trace_at = time.monotonic()
                log_lines = recent_log_lines(log_path)
                status = "stopping" if requested_stop else "running"
                report_progress({"status": status, "pid": process.pid,
                             "completed_epochs": completed, "progress": min(99.99, completed / max(1, int(task['parameters']['epochs'])) * 100),
                             "latest_metrics": latest, "metrics": new_metrics, "log_tail": log_lines,
                             "run_dir": task["run_dir"]})
            if stop.is_set() and not requested_stop and getattr(stop, "preserve_training", False) and process.poll() is None:
                detach_training(config, task, command, identity)
            if stop.is_set() and process.poll() is None:
                request_stop("agent_shutdown")
            try:
                exit_code = process.wait(timeout=config.get("training_stop_timeout_seconds", 15))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                exit_code = process.wait()
        completed, latest, metrics = read_remote_progress(
            Path(task["run_dir"]), final=True, sent_epochs=last_metrics, pending_metrics=pending_metrics,
        )
        pending_metrics.update((item["epoch"], item) for item in metrics)
        completed = max(completed, task.get("completed_epochs", 0))
        latest = latest or task.get("latest_metrics", {})
        try:
            report_manifest(config, manifest_payload)
        except (OSError, ValueError, HTTPError, URLError):
            LOGGER.warning("产物清单暂未同步，平台下次查询时将请求补传")
        try:
            report_training_trace(config, task, sent_trace_images)
        except (OSError, ValueError, HTTPError, URLError, json.JSONDecodeError):
            LOGGER.warning("训练过程采样暂未同步")
        log_lines = recent_log_lines(Path(task["log_path"]))
        final_status = "stopped" if requested_stop else "completed" if exit_code == 0 else "failed"
        error = None if final_status != "failed" else f"训练进程异常退出，退出码 {exit_code}"
        completion_info = {} if requested_stop else training_completion_info(task, completed, log_lines)
        record = read_training_result(task)
        if final_status == "failed" and record and record.get("status") == "failed" and record.get("error_message"):
            error = str(record["error_message"])[:4000]
        if final_status == "failed" and (not record or record.get("status") != "failed"):
            completion_info = {"source": "process_exit", "reason": "exception", "completed_epochs": completed,
                               "expected_epochs": int(task["parameters"]["epochs"])}
        if not completion_info and final_status == "completed":
            completion_info = {"source": "process_exit", "reason": "epochs_completed" if completed >= int(task["parameters"]["epochs"]) else "training_completed",
                               "completed_epochs": completed, "expected_epochs": int(task["parameters"]["epochs"])}
        final_payload = {"_claim_lease": lease, "status": final_status, "completed_epochs": completed,
                         "completion_info": completion_info,
                         "progress": 100 if final_status == "completed" else min(99.99, completed / max(1, int(task['parameters']['epochs'])) * 100),
                         "latest_metrics": latest,
                         "metrics": [pending_metrics[epoch] for epoch in sorted(pending_metrics)
                                     if epoch not in last_metrics],
                         "log_tail": log_lines, "exit_code": exit_code, "command": command,
                         "error_message": error, "stop_reason": stop_reason, "run_dir": task["run_dir"]}
        pending_path = _save_terminal_for_retry(config, task, final_payload)
        while True:
            try:
                if deliver_pending_final(config, pending_path):
                    break
            except HTTPError as exc:
                if exc.code in {401, 403, 404, 422}:
                    LOGGER.error("最终训练状态无法上报，HTTP %s；本地退出码 %s；已保存 %s", exc.code, exit_code, pending_path)
                    break
                LOGGER.warning("最终训练状态上报失败，稍后重试：HTTP %s", exc.code)
            except (OSError, ValueError, URLError) as exc:
                LOGGER.warning("最终训练状态上报失败，稍后重试：%s", exc)
            if stop.is_set():
                break
            stop.wait(10)
    except TrainingDetached:
        raise
    except Exception as exc:
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            except OSError:
                pass
        try:
            pending_path = _save_terminal_for_retry(config, task, {
                "status": "failed", "error_message": str(exc)[:4000], "exit_code": 1,
                "completed_epochs": task.get("completed_epochs", 0),
                "progress": task.get("progress", 0), "latest_metrics": task.get("latest_metrics", {}),
                "log_tail": recent_log_lines(Path(task["log_path"]))})
            deliver_pending_final(config, pending_path)
        except Exception:
            LOGGER.exception("远程训练错误状态上报失败")
        LOGGER.error("远程训练任务失败：%s", exc)


def run_remote_task(config: dict, task: dict, stop: threading.Event, training_lock=None) -> None:
    """Run one task while holding its GPU/workspace reservations through final reporting."""
    try:
        lock = training_lock if training_lock is not None else acquire_task_training_lock(config, task)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        lock = None
        lock_error = str(exc)
    else:
        lock_error = "远程 Agent 未能取得该 Run 的 GPU 或输出目录锁，训练未启动"
    if lock is None:
        LOGGER.error("Run %s 的本地资源锁不可用，训练未启动", task.get("run_id"))
        try:
            save_active_run(config, task, "claimed")
            stop_requested = confirm_remote_claim(config, task, stop)
            status = "stopped" if stop_requested else "failed"
            error = None if stop_requested else lock_error[:4000]
            path = _save_terminal_for_retry(config, task, {
                "status": status, "error_message": error, "stop_reason": "platform_stop" if stop_requested else None,
                "exit_code": None if stop_requested else 1,
                "completed_epochs": task.get("completed_epochs", 0),
                "progress": task.get("progress", 0), "latest_metrics": task.get("latest_metrics", {}),
                "metrics": [], "log_tail": recent_log_lines(Path(task["log_path"])),
                "run_dir": task.get("run_dir")})
            _deliver_terminal_once(config, path)
        except ClaimConfirmationPending as exc:
            LOGGER.warning("%s；稍后重新确认资源冲突 Run", exc)
        except ClaimRejected as exc:
            clear_active_run(config, task["run_id"])
            LOGGER.warning("%s；资源冲突 Run 未启动", exc)
        except Exception:
            LOGGER.exception("无法收尾未取得资源锁的 Run %s", task.get("run_id"))
        return
    try:
        save_active_run(config, task, "claimed")
        try:
            stop_requested = confirm_remote_claim(config, task, stop)
        except ClaimConfirmationPending as exc:
            LOGGER.warning("%s；未启动训练，下次循环会使用同一租约继续确认", exc)
            return
        except ClaimRejected as exc:
            clear_active_run(config, task["run_id"])
            LOGGER.warning("%s；未启动训练进程", exc)
            return
        save_active_run(config, task, "confirmed")
        if stop_requested:
            path = _save_terminal_for_retry(config, task, {
                "status": "stopped", "stop_reason": "platform_stop", "exit_code": None,
                "completed_epochs": task.get("completed_epochs", 0), "progress": task.get("progress", 0),
                "latest_metrics": task.get("latest_metrics", {}), "metrics": [],
                "log_tail": recent_log_lines(Path(task["log_path"])), "run_dir": task.get("run_dir")})
            _deliver_terminal_once(config, path)
            return
        if stop.is_set() and getattr(stop, "preserve_training", False):
            return  # Keep the confirmed lease; do not launch new work during exit.
        _run_remote_task(config, task, stop, lock)
    except TrainingDetached:
        lock.close()  # No LOCK_UN while the child is still training.
        lock = None
    finally:
        release_training_lock(lock)


def send(config: dict, payload: dict) -> tuple[int, str | None]:
    """Send an ordered sample; a new center may request one immediate resample."""
    request = Request(config["center_url"] + "/api/compute/nodes/" + config["node_id"] + "/report",
                      data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), method="POST",
                      headers={"Authorization": "Bearer " + config["token"], "Content-Type": "application/json"})
    with build_opener(NoRedirect()).open(request, timeout=15) as response:
        result = json.loads(response.read(65536))
    interval = result.get("report_interval_seconds", config["report_interval_seconds"])
    if type(interval) is not int or not 5 <= interval <= 300:
        interval = config["report_interval_seconds"]
    request_id = result.get("refresh_request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
        request_id = None  # Older centers omit the command; malformed commands are ignored.
    return interval, request_id


def fulfill_refresh_request(config: dict, request_id: str, stop: threading.Event) -> dict | None:
    """Publish a distinct sample after the command, then resume training polls."""
    if stop.is_set():
        return None
    try:
        payload = collect(config["nvidia_smi_bin"], config)
        payload["refresh_request_id"] = request_id
        send(config, payload)
        return payload
    except (HTTPError, URLError, OSError, ValueError, TypeError):
        LOGGER.warning("按需算力采样未完成；保留常规上报与训练轮询，下次心跳重试")
        return None


# BEGIN PORTABLE DATASET CORE (generated from backend/app/dataset_files.py)
DATASET_FILES_SOURCE = '"""Portable dataset file operations; embedded unchanged into the standalone Agent."""\nfrom __future__ import annotations\nimport hashlib\nimport json\nimport os\nimport re\nfrom pathlib import Path, PurePosixPath\nimport shutil\nimport tarfile\n\nMAX_BYTES = 50 * 1024**3\nMAX_FILES = 200_010\nMARKER = ".yolo-platform-dataset.json"\n\n\ndef report_progress(check, stage, processed=0, total=None, unit=\'files\', current_file=\'\'):\n    """Optional progress hook; legacy callers keep the same cancellation callback."""\n    report = getattr(check, \'progress\', None)\n    if report:\n        report({\'stage\': stage, \'processed\': processed, \'total\': total, \'unit\': unit,\n                \'current_file\': str(current_file)[-400:]})\n\n\ndef throttle_read(check, size):\n    """Optional caller-owned budget; ordinary and legacy file work is unchanged."""\n    throttle = getattr(check, \'throttle_bytes\', None)\n    if throttle and size:\n        throttle(size)\n\n\ndef digest(path, check=None, stage=None):\n    value = hashlib.sha256()\n    processed = 0\n    total = Path(path).stat().st_size if stage else None\n    if stage:\n        report_progress(check, stage, 0, total, \'bytes\')\n    with Path(path).open("rb") as stream:\n        for chunk in iter(lambda: stream.read(1024 * 1024), b""):\n            if check:\n                check()\n            throttle_read(check, len(chunk))\n            value.update(chunk)\n            processed += len(chunk)\n            if stage:\n                report_progress(check, stage, processed, total, \'bytes\')\n    return value.hexdigest()\n\n\ndef inside(path, roots, create=False):\n    raw = Path(path).expanduser()\n    resolved = raw.resolve()\n    if not raw.is_absolute() or raw != resolved:\n        raise ValueError("目录必须为可解析的绝对路径，不能包含符号链接")\n    allowed = [Path(root).resolve() for root in roots]\n    if not any(root == resolved or root in resolved.parents for root in allowed):\n        raise ValueError("目录不在数据集允许范围内，请核对服务器配置")\n    if create:\n        resolved.mkdir(parents=True, exist_ok=True)\n    return resolved\n\n\ndef check_root(path, roots, create=False):\n    root = inside(path, roots, create)\n    if not root.is_dir():\n        raise ValueError("目录不存在或不是文件夹")\n    # Exercise actual filesystem permissions, rather than only reporting os.access.\n    import tempfile\n    with tempfile.TemporaryFile(dir=root):\n        pass\n    usage = shutil.disk_usage(root)\n    return {"resolved_path": str(root), "free_bytes": usage.free, "total_bytes": usage.total,\n            "status": "ready"}\n\n\ndef scan(root, check=None, stage="scan"):\n    root = Path(root)\n    if root.is_symlink() or not root.is_dir():\n        raise ValueError("数据集目录不存在或为符号链接")\n    files, total = [], 0\n    report_progress(check, stage)\n    for directory, dirs, names in os.walk(root, followlinks=False):\n        for name in dirs:\n            if (Path(directory) / name).is_symlink():\n                raise ValueError("数据集不能包含符号链接目录")\n        for name in sorted(names):\n            if name.endswith(\'.cache\') or name in {MARKER, \'.dataset-content.json\'}:\n                continue\n            if name == \'.dataset-storage.lock\' and Path(directory) == root:\n                # An Agent can be asked to export the allow-listed root itself.\n                # Its advisory lock is platform metadata, not dataset content.\n                continue\n            path = Path(directory) / name\n            # Give optional background readers a small metadata-I/O budget as\n            # well as byte budgets. Ordinary callers have no throttle hook.\n            throttle_read(check, 4096)\n            if path.is_symlink() or not path.is_file():\n                raise ValueError("数据集不能包含符号链接或特殊文件")\n            relative = path.relative_to(root).as_posix()\n            safe_member(relative)\n            before = path.stat()\n            size = before.st_size\n            total += size\n            if total > MAX_BYTES or len(files) >= MAX_FILES:\n                raise ValueError("数据集超过 50 GB 或 20 万文件限制")\n            if check:\n                check()\n            report_progress(check, stage, len(files), current_file=relative)\n            sha = digest(path, check)\n            after = path.stat()\n            if (before.st_dev, before.st_ino, before.st_mode, before.st_size,\n                    before.st_mtime_ns, before.st_ctime_ns) != (\n                    after.st_dev, after.st_ino, after.st_mode, after.st_size,\n                    after.st_mtime_ns, after.st_ctime_ns):\n                raise ValueError("数据集在校验中发生修改，请停止修改后重试")\n            files.append({"path": relative, "size": size, "sha256": sha})\n    report_progress(check, stage, len(files), len(files))\n    files.sort(key=lambda x: x[\'path\'])\n    signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()\n    return {"files": files, "total_bytes": total, "fingerprint": signature}\n\n\ndef safe_member(name):\n    p = PurePosixPath(name)\n    if not name or p.is_absolute() or \'\\\\\' in name or any(part in {\'\', \'.\', \'..\'} or part.startswith(\'.\') for part in name.split(\'/\')) or any(ord(c) < 32 for c in name):\n        raise ValueError("数据文件相对路径无效")\n    return p\n\n\ndef pack(root, archive, check=None, *, manifest=None):\n    # Reuse quota/archive planning while retaining full source verification.\n    manifest = manifest if manifest is not None else scan(root, check)\n    archive = Path(archive)\n    archive.parent.mkdir(parents=True, exist_ok=True)\n    temporary = archive.with_suffix(\'.tmp\')\n    processed = 0\n    report_progress(check, \'pack\', 0, manifest[\'total_bytes\'], \'bytes\')\n    class Reader:\n        def __init__(self, stream, name):\n            self.stream, self.name = stream, name\n            self.hash = hashlib.sha256()\n        def read(self, size=-1):\n            nonlocal processed\n            data = self.stream.read(size)\n            processed += len(data)\n            if check:\n                check()\n            throttle_read(check, len(data))\n            self.hash.update(data)\n            report_progress(check, \'pack\', processed, manifest[\'total_bytes\'], \'bytes\', self.name)\n            return data\n    try:\n        with tarfile.open(temporary, \'w\') as tar:\n            for file in manifest[\'files\']:\n                if check:\n                    check()\n                throttle_read(check, 4096)\n                path = Path(root) / file[\'path\']\n                info = tar.gettarinfo(str(path), arcname=file[\'path\'])\n                if not info.isfile() or info.size != file[\'size\']:\n                    raise ValueError(\'源数据在导出期间发生修改\')\n                with path.open(\'rb\') as stream:\n                    reader = Reader(stream, file[\'path\'])\n                    tar.addfile(info, reader)\n                    if reader.hash.hexdigest() != file[\'sha256\']:\n                        raise ValueError(\'源数据在导出期间发生修改\')\n        # A second scan catches changes to files already copied and new files.\n        if scan(root, check, stage=\'source_verify\')[\'fingerprint\'] != manifest[\'fingerprint\']:\n            raise ValueError("源数据在导出期间发生修改")\n        temporary.replace(archive)\n    except BaseException:\n        temporary.unlink(missing_ok=True)\n        raise\n    return {**manifest, \'archive_size\': archive.stat().st_size, \'archive_sha256\': digest(archive, check, stage=\'archive_hash\')}\n\n\ndef unpack(archive, target, manifest, check=None):\n    target = Path(target)\n    target.mkdir(parents=True, exist_ok=True)\n    entries = {item[\'path\']: item for item in manifest[\'files\']}\n    if len(entries) != len(manifest[\'files\']) or len(entries) > MAX_FILES or sum(i[\'size\'] for i in entries.values()) > MAX_BYTES:\n        raise ValueError("数据清单大小或路径无效")\n    seen = set()\n    processed = 0\n    total = sum(i[\'size\'] for i in entries.values())\n    report_progress(check, \'unpack\', 0, total, \'bytes\')\n    with tarfile.open(archive, \'r:\') as tar:\n        for item in tar:\n            if check:\n                check()\n            safe_member(item.name)\n            spec = entries.get(item.name)\n            if not item.isfile() or item.name in seen or not spec or item.size != spec[\'size\']:\n                raise ValueError("归档文件类型、大小或清单不匹配")\n            seen.add(item.name)\n            path = target / item.name\n            path.parent.mkdir(parents=True, exist_ok=True)\n            source = tar.extractfile(item)\n            report_progress(check, \'unpack\', processed, total, \'bytes\', item.name)\n            with path.open(\'xb\') as output:\n                for chunk in iter(lambda: source.read(1024 * 1024), b\'\'):\n                    throttle_read(check, len(chunk))\n                    output.write(chunk)\n                    if check:\n                        check()\n            if digest(path, check) != spec[\'sha256\']:\n                raise ValueError("数据文件 SHA-256 校验失败")\n            processed += item.size\n            report_progress(check, \'unpack\', processed, total, \'bytes\', item.name)\n    if seen != set(entries):\n        raise ValueError("归档缺少数据文件")\n\n\ndef publish(archive, root, name, manifest, identity, config_text, check=None):\n    if not name or name.startswith(\'.\') or \'/\' in name or \'\\\\\' in name:\n        raise ValueError("数据集名称无效")\n    if not re.fullmatch(r\'[0-9a-f]{32}\', identity):\n        raise ValueError(\'操作编号无效\')\n    root = Path(root)\n    if root.is_symlink() or not root.is_dir():\n        raise ValueError(\'目标根目录不可用\')\n    destination = root / name\n    if destination.exists() or destination.is_symlink():\n        try:\n            metadata = json.loads((destination / MARKER).read_text())\n        except (OSError, ValueError):\n            metadata = {}\n        if metadata.get(\'operation_id\') == identity:\n            actual = scan(destination, check)\n            if actual[\'fingerprint\'] != metadata.get(\'fingerprint\'):\n                raise ValueError(\'已发布的目录内容发生变化，请人工核对\')\n            return {\'path\': str(destination), \'fingerprint\': actual[\'fingerprint\'], \'total_bytes\': actual[\'total_bytes\'], \'config_sha256\': digest(destination / \'data.yaml\'), \'already_published\': True}\n        raise ValueError("目标存在同名目录，不允许覆盖")\n    work = root / (\'.dataset-publish-\' + identity)\n    if work.is_symlink():\n        raise ValueError(\'临时目录不能是符号链接\')\n    if work.exists():\n        shutil.rmtree(work)\n    required = manifest[\'total_bytes\'] + 100 * 1024**2\n    if shutil.disk_usage(root).free < required:\n        raise ValueError("目标磁盘空间不足")\n    try:\n        unpack(archive, work, manifest, check)\n        if config_text is not None:\n            (work / \'data.yaml\').write_text(config_text, encoding=\'utf-8\')\n        actual = scan(work, check, stage=\'target_verify\')\n        (work / MARKER).write_text(json.dumps({\'version\': 1, \'name\': name, \'operation_id\': identity,\n                                               \'fingerprint\': actual[\'fingerprint\']}, ensure_ascii=False), encoding=\'utf-8\')\n        if check:\n            check(\'committing\')\n        if destination.exists() or destination.is_symlink():\n            raise ValueError("目标目录已出现，拒绝覆盖")\n        work.rename(destination)\n        return {\'path\': str(destination), \'fingerprint\': actual[\'fingerprint\'], \'total_bytes\': actual[\'total_bytes\'], \'config_sha256\': digest(destination / \'data.yaml\')}\n    finally:\n        if work.exists():\n            shutil.rmtree(work)\n\n\ndef remove_dataset(path, roots, expected_fingerprint, check=None):\n    path = inside(path, roots)\n    allowed = [Path(root).resolve() for root in roots]\n    if any(path == root or path in root.parents for root in allowed) or not path.parent.is_dir():\n        raise ValueError("不能删除根目录或离线磁盘上的数据")\n    if not expected_fingerprint:\n        raise ValueError("缺少内容指纹，拒绝删除")\n    quarantine = path.parent / (\'.dataset-delete-\' + hashlib.sha256(\n        (str(path) + expected_fingerprint).encode()).hexdigest()[:32])\n    if quarantine.is_symlink():\n        raise ValueError("删除暂存目录不能为符号链接")\n    if quarantine.exists():\n        # A crash may have interrupted rmtree. Only this path+snapshot\'s private\n        # quarantine may be resumed; a newly created source is never removed.\n        if check:\n            check(\'committing\')\n        shutil.rmtree(quarantine)\n        return {\'deleted\': True, \'recovered\': True}\n    if not path.exists():\n        return {\'already_missing\': True}\n    if scan(path, check, stage=\'delete_verify\')[\'fingerprint\'] != expected_fingerprint:\n        raise ValueError("数据集内容已变化，拒绝删除，请重新校验")\n    if check:\n        check(\'committing\')\n    report_progress(check, \'remove\')\n    path.rename(quarantine)\n    shutil.rmtree(quarantine)\n    return {\'deleted\': True}\n'
DATASET_FILES = {}
exec(compile(DATASET_FILES_SOURCE, "<dataset_files>", "exec"), DATASET_FILES)
# END PORTABLE DATASET CORE

# BEGIN PORTABLE DISCOVERY CORE (generated from backend/app/dataset_discovery_files.py)
DATASET_DISCOVERY_SOURCE = '"""Read-only, bounded discovery. Standard library only; also embedded in the Agent."""\nfrom __future__ import annotations\n\nimport hashlib\nimport os\nfrom pathlib import Path\nimport stat\nimport time\nimport uuid\n\nCONFIG_LIMIT = 64 * 1024\nSKIP = {\'images\', \'labels\', \'train\', \'val\', \'test\', \'runs\', \'weights\', \'node_modules\', \'__pycache__\'}\n\n\ndef discover(path, roots, limits, report, check):\n    """Publish small idempotent chunks; never read image/label contents or write sources.\n\n    A restart begins a new generation, so old partial candidates cannot be mistaken\n    for the completed result. The centre retains the last progress across restarts.\n    """\n    root = Path(path)\n    if not root.is_absolute() or root != root.resolve() or not root.is_dir():\n        raise ValueError(\'扫描目录不存在或包含符号链接\')\n    if not any(root == Path(r).resolve() or Path(r).resolve() in root.parents for r in roots):\n        raise ValueError(\'扫描目录不在允许范围内\')\n    depth = int(limits.get(\'depth\', 3))\n    maximum = int(limits.get(\'max_candidates\', 500))\n    directories = int(limits.get(\'max_directories\', 10000))\n    seconds = int(limits.get(\'max_seconds\', 120))\n    if not (0 <= depth <= 12 and 1 <= maximum <= 2000 and 1 <= directories <= 50000 and 5 <= seconds <= 600):\n        raise ValueError(\'扫描限制无效\')\n    generation = uuid.uuid4().hex\n    progress = {\'generation\': generation, \'visited\': 0, \'found\': 0, \'skipped_links\': 0,\n                \'skipped_depth\': 0, \'limited\': False, \'limit_reason\': \'\', \'current_path\': str(root)}\n    pending, stack = [], [(root, 0)]\n    started = last_report = time.monotonic()\n    entries_seen = 0\n\n    def flush(force=False):\n        nonlocal pending, last_report\n        if force or len(pending) >= 20 or time.monotonic() - last_report >= 2:\n            report({**progress, \'candidates\': pending})\n            pending = []\n            last_report = time.monotonic()\n\n    def limit(reason):\n        progress.update(limited=True, limit_reason=reason)\n\n    flush(True)\n    while stack:\n        check()\n        if progress[\'visited\'] >= directories:\n            limit(\'达到目录数量上限，请缩小扫描位置或调高限制\'); break\n        if time.monotonic() - started >= seconds:\n            limit(\'达到扫描时间上限，请缩小扫描位置或调高限制\'); break\n        directory, level = stack.pop()\n        progress.update(visited=progress[\'visited\'] + 1, current_path=str(directory))\n        item = None\n        try:\n            # Recheck every parent, rather than following a link swapped in mid-scan.\n            if directory != directory.resolve() or not directory.is_dir():\n                progress[\'skipped_links\'] += 1\n                continue\n            children, names = [], set()\n            with os.scandir(directory) as iterator:\n                for entry in iterator:\n                    entries_seen += 1\n                    if entries_seen % 128 == 0:\n                        check(); flush()\n                    if time.monotonic() - started >= seconds or entries_seen > directories * 100:\n                        limit(\'达到扫描时间或目录项数量上限，请缩小扫描范围\'); break\n                    if entry.name.startswith(\'.\'):\n                        continue\n                    if entry.name in {\'data.yaml\', \'data.yml\', \'images\', \'train\', \'val\'}:\n                        names.add(entry.name)\n                    if entry.is_symlink():\n                        progress[\'skipped_links\'] += 1\n                    elif entry.name not in SKIP and entry.is_dir(follow_symlinks=False):\n                        children.append(Path(entry.path))\n                        if len(children) + len(stack) > directories:\n                            limit(\'达到待检查目录数量上限，请缩小扫描范围\'); break\n            likely = \'data.yaml\' in names or \'data.yml\' in names or \'images\' in names or (\'train\' in names and \'val\' in names)\n            if likely:\n                item = {\'relative_path\': directory.relative_to(root).as_posix(), \'state\': \'needs_attention\',\n                        \'reason\': \'\', \'config_text\': \'\', \'config_sha256\': \'\',\n                        \'structure\': {n: (directory / n).is_dir() and not (directory / n).is_symlink()\n                                      for n in (\'images/train\', \'images/val\', \'labels\', \'train\', \'val\')}}\n                config = directory / \'data.yaml\'\n                if \'data.yaml\' not in names:\n                    item[\'reason\'] = \'缺少 data.yaml；请整理目录或使用上传导入\'\n                    if \'train\' in names and \'val\' in names and \'images\' not in names:\n                        item.update(state=\'unsupported\', reason=\'疑似分类数据集，当前登记校验器仅支持目标检测\')\n                else:\n                    flags = os.O_RDONLY | getattr(os, \'O_NOFOLLOW\', 0) | getattr(os, \'O_NONBLOCK\', 0)\n                    with os.fdopen(os.open(config, flags), \'rb\') as handle:\n                        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):\n                            raise ValueError(\'data.yaml 必须是普通文件\')\n                        raw = handle.read(CONFIG_LIMIT + 1)\n                    if len(raw) > CONFIG_LIMIT:\n                        item[\'reason\'] = \'data.yaml 超过扫描配置大小上限（64 KiB）\'\n                    else:\n                        item.update(state=\'candidate\', config_text=raw.decode(\'utf-8-sig\'),\n                                    config_sha256=hashlib.sha256(raw).hexdigest())\n            # Dataset internals are excluded, but nested sibling versions remain discoverable.\n            if level < depth:\n                stack.extend((child, level + 1) for child in sorted(children, reverse=True))\n            else:\n                progress[\'skipped_depth\'] += len(children)\n        except (OSError, UnicodeError, ValueError) as exc:\n            item = {\'relative_path\': directory.relative_to(root).as_posix(), \'state\': \'inaccessible\',\n                    \'reason\': (\'无法检查目录或配置：\' + str(exc))[:600], \'config_text\': \'\', \'config_sha256\': \'\', \'structure\': {}}\n        if item:\n            pending.append(item); progress[\'found\'] += 1\n        flush()\n        if progress[\'limited\']:\n            break\n        if progress[\'found\'] >= maximum:\n            if stack:\n                limit(\'达到候选数量上限，请缩小扫描位置或调高限制\')\n            break\n    flush(True)\n    return progress\n'
DATASET_DISCOVERY = {}
exec(compile(DATASET_DISCOVERY_SOURCE, '<dataset_discovery_files>', 'exec'), DATASET_DISCOVERY)
# END PORTABLE DISCOVERY CORE

def dataset_api(config, path, method='GET', payload=None, timeout=120):
    """Larger bounded response for a 200k-file manifest; redirects stay disabled."""
    headers = {'Authorization': 'Bearer ' + config['token'], 'Content-Type': 'application/json'}
    request = Request(config['center_url'] + path, method=method, headers=headers,
                      data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None)
    with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        data = response.read(256 * 1024**2 + 1)
    if len(data) > 256 * 1024**2:
        raise ValueError('数据集协议响应超出限制')
    return json.loads(data or b'{}')


def dataset_atomic_json(path, payload):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def dataset_staging_owner(config):
    """Identify the authenticated owner without persisting credentials."""
    center = config.get('center_url')
    node_id = config.get('node_id')
    if not isinstance(center, str) or not center or not isinstance(node_id, str):
        return None
    return {'version': 1, 'node_id': node_id,
            'center_sha256': hashlib.sha256(center.rstrip('/').encode('utf-8')).hexdigest()}


def dataset_staging_owned_by(state, owner):
    """Legacy/unknown receipts remain replayable but are never auto-reclaimed."""
    if owner is None:
        return False
    path = state / 'owner.json'
    try:
        if path.is_symlink():
            return False
        with path.open('rb') as stream:
            data = stream.read(4097)
        return len(data) <= 4096 and json.loads(data) == owner
    except (OSError, ValueError, TypeError):
        return False


def reclaim_dataset_staging(config, *, now=None):
    """Reclaim only center-approved receipts; offline/unknown work stays resumable."""
    now = time.time() if now is None else now
    previous = config.get('_dataset_reconcile_at', 0)
    if previous and now - previous < 300:
        return 0
    config['_dataset_reconcile_at'] = now
    owner = dataset_staging_owner(config)
    if owner is None:
        return 0
    root = Path(config['training_state_dir']).parent / '.dataset-state'
    if not root.is_dir():
        return 0
    candidates = []
    # Age is only a scheduling hint. The authenticated center decides safety.
    for identity in list(root.iterdir()):
        if not identity.is_dir() or identity.is_symlink() or not re.fullmatch(r'[0-9a-f]{32}', identity.name):
            continue
        for state in identity.iterdir():
            if not state.is_dir() or state.is_symlink() or not re.fullmatch(r'[0-9a-f]{32}', state.name):
                continue
            try:
                if now - state.stat().st_mtime < 3600 or not dataset_staging_owned_by(state, owner):
                    continue
            except OSError:
                continue
            candidates.append(state)
    candidates.sort(key=lambda path: str(path))
    # Protected receipts must not permanently hide later completed work.
    cursor = config.get('_dataset_reconcile_cursor', '')
    start = next((index for index, state in enumerate(candidates) if str(state) > cursor), 0)
    candidates = (candidates[start:] + candidates[:start])[:1024]
    removed = 0
    # Keep each background pass bounded; a large historical backlog is drained
    # gradually without monopolizing network requests or blocking shutdown.
    for index in range(0, len(candidates), 128):
        batch = candidates[index:index + 128]
        receipts = [{'id': state.parent.name, 'token': state.name} for state in batch]
        try:
            response = dataset_api(config, f"/api/dataset-storage/agent/{config['node_id']}/reconcile",
                                   'POST', {'receipts': receipts}, timeout=20)
            approved = {(item['id'], item['token']) for item in response.get('reclaimable', [])}
        except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError):
            # Old centers do not implement reconciliation. Never infer terminal
            # state from a failed request or expire unfinished transfers by age.
            break
        config['_dataset_reconcile_cursor'] = str(batch[-1])
        for state in batch:
            if (state.parent.name, state.name) not in approved:
                continue
            try:
                with (state / '.lock').open('a') as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if not dataset_staging_owned_by(state, owner):
                        continue
                    shutil.rmtree(state)
                    removed += 1
            except OSError:
                continue
    return removed


def run_dataset_reclamation(config):
    try:
        reclaim_dataset_staging(config)
    except (OSError, ValueError, TypeError):
        LOGGER.warning('数据集中转清理暂未完成，现有断点已保留')


def poll_dataset_operation(config, busy):
    """Old centres may ignore the filter; never execute their heavy work while busy."""
    path = f"/api/dataset-storage/agent/{config['node_id']}/poll"
    if busy:
        path += '?lightweight_only=true'
    work = dataset_api(config, path, 'POST', {}).get('operation')
    if work and busy and work.get('phase') not in LIGHTWEIGHT_DATASET_PHASES and not concurrent_refresh_work(work):
        return None
    return work


def concurrent_refresh_work(work):
    """Only a centre-negotiated refresh export may bypass the training gate."""
    return (work.get('kind') == 'refresh' and work.get('phase') == 'export'
            and work.get('read_only_during_training') is True)


def dataset_blocks_training(thread, phase, read_only_refresh=False):
    return (thread is not None and thread.is_alive()
            and phase not in LIGHTWEIGHT_DATASET_PHASES and not read_only_refresh)


def dataset_io_pressure_percent():
    """Linux PSI is optional; unsupported systems rely on the read budget."""
    try:
        line = Path('/proc/pressure/io').read_text(encoding='ascii').splitlines()[0]
        match = re.search(r'\bavg10=([0-9]+(?:\.[0-9]+)?)\b', line)
        return float(match.group(1)) if match else None
    except (OSError, ValueError, IndexError):
        return None


def dataset_refresh_resource_reason(config, staging, minimum_free):
    if minimum_free and shutil.disk_usage(staging).free < minimum_free:
        return 'low_disk_space'
    threshold = config.get('dataset_refresh_io_pressure_pct', DEFAULT_DATASET_REFRESH_IO_PRESSURE_PCT)
    pressure = dataset_io_pressure_percent() if threshold else None
    if pressure is not None and pressure >= threshold:
        return 'disk_pressure'
    return None


def verify_refresh_source(source, expected_fingerprint, check):
    if DATASET_FILES['scan'](source, check, stage='source_verify')['fingerprint'] != expected_fingerprint:
        raise ValueError('源数据在重新加载期间发生修改，请重试')


def run_dataset_operation(config, work, stop):
    """Execute a phase with restart receipts and resumable chunk offsets."""
    identity, token, phase = work['id'], work['token'], work['phase']
    if not all(re.fullmatch(r'[0-9a-f]{32}', value) for value in (identity, token)):
        return
    prefix = f"/api/dataset-storage/agent/{config['node_id']}"
    base = f'{prefix}/{identity}'
    state = Path(config['training_state_dir']).parent / '.dataset-state' / identity / token
    state.mkdir(parents=True, mode=0o700, exist_ok=True)
    locks = []
    training_lock = None
    read_only_refresh = concurrent_refresh_work(work)
    minimum_free = [0]
    resource_checked = [0.0, None]
    current_progress = [None]
    resume_progress = [None]
    last = [0.0]
    def check(stage=None):
        if stop.is_set():
            raise OSError('Agent 正在退出，等待下次恢复')
        if stage or time.monotonic() - last[0] > 4:
            dataset_api(config, base + '/heartbeat', 'POST', {'token': token, 'phase': phase, 'committing': stage == 'committing'})
            last[0] = time.monotonic()
        if read_only_refresh:
            resumed = resume_progress[0]
            while True:
                now = time.monotonic()
                if now - resource_checked[0] >= 2:
                    resource_checked[:] = [now, dataset_refresh_resource_reason(config, state, minimum_free[0])]
                if not resource_checked[1]:
                    break
                report_progress({'stage': 'paused_resource', 'resource_reason': resource_checked[1],
                                 'processed': (resumed or {}).get('processed', 0),
                                 'total': (resumed or {}).get('total'),
                                 'unit': (resumed or {}).get('unit', 'files')})
                if stop.wait(2):
                    raise OSError('Agent 正在退出，等待下次恢复')
                if time.monotonic() - last[0] > 4:
                    dataset_api(config, base + '/heartbeat', 'POST', {'token': token, 'phase': phase})
                    last[0] = time.monotonic()
            if current_progress[0] and current_progress[0].get('stage') == 'paused_resource' and resumed:
                report_progress(resumed)
    progress_last = [0.0, None, None]
    progress_sequence = [0]
    def report_progress(value):
        # Old centres omit this run identity; retain their existing heartbeat protocol.
        if not work.get('progress_run_id'):
            return
        now = time.monotonic()
        stage, done, total = value['stage'], value.get('processed', 0), value.get('total')
        finished = total is not None and done == total and (stage, done) != (progress_last[1], progress_last[2])
        if stage != progress_last[1] or now - progress_last[0] >= 2 or finished:
            progress_sequence[0] += 1
            dataset_api(config, base + '/heartbeat', 'POST', {
                'token': token, 'phase': phase, 'progress_run_id': work['progress_run_id'], 'progress': {**value, 'sequence': progress_sequence[0]}})
            progress_last[:] = [now, stage, done]
            current_progress[0] = value
            if stage != 'paused_resource':
                resume_progress[0] = value
    check.progress = report_progress
    if read_only_refresh:
        rate = config.get('dataset_refresh_read_rate_bytes_per_second', DEFAULT_DATASET_REFRESH_READ_RATE)
        scheduled = [time.monotonic()]
        def throttle_bytes(count):
            scheduled[0] = max(time.monotonic() - 0.25, scheduled[0]) + count / rate
            while True:
                delay = scheduled[0] - time.monotonic()
                if delay <= 0:
                    break
                if stop.wait(min(delay, 0.25)):
                    raise OSError('Agent 正在退出，等待下次恢复')
                check()
        check.throttle_bytes = throttle_bytes
    try:
        # Cross-process receipt ownership; never allow an expired lease to run
        # a second writer while the first still holds filesystem locks.
        lock = (state / '.lock').open('a')
        locks.append(lock)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        owner = dataset_staging_owner(config)
        owner_path = state / 'owner.json'
        if owner_path.exists() or owner_path.is_symlink():
            if not dataset_staging_owned_by(state, owner):
                LOGGER.warning('数据集操作缓存属于其他中心或节点，保留现有回执：%s', identity)
                return
        receipt = state / 'result.json'
        if receipt.is_file():
            result = json.loads(receipt.read_text(encoding='utf-8'))
            if read_only_refresh:
                source = DATASET_FILES['inside'](work['path'], config['dataset_roots'])
                verify_refresh_source(source, result['fingerprint'], check)
        else:
            check()
            # Only a successful authenticated lease check may claim a new or
            # partial legacy cache. Old completed receipts keep their identity.
            if owner is not None and not owner_path.exists():
                dataset_atomic_json(owner_path, owner)
            report_progress({'stage': phase})
            if phase not in LIGHTWEIGHT_DATASET_PHASES and not read_only_refresh:
                training_lock = acquire_training_lock(config)
                if training_lock is None:
                    report_progress({'stage': 'waiting_lock'})
                    return  # Busy workspace is retried after the lease expires.
            roots = config['dataset_roots']
            if phase == 'discover':
                def report(progress):
                    dataset_api(config, base + '/heartbeat', 'POST', {'token': token, 'phase': phase, 'result': progress})
                    report_progress({'stage': 'discover', 'processed': progress['visited'],
                                     'unit': 'directories', 'current_file': progress.get('current_path', '')})
                result = DATASET_DISCOVERY['discover'](work['path'], roots, work['limits'], report, check)
            elif phase == 'check':
                raw = Path(work['path']).expanduser()
                path = raw if raw.is_absolute() else Path(config['training_cwd']) / raw
                result = DATASET_FILES['check_root'](str(path.absolute()), roots, work.get('create', False))
                result['allowed_roots'] = roots
            else:
                # Keep old Agents' global exclusive lock contract. Negotiated
                # refresh readers share it; all writers remain exclusive.
                for root in sorted(roots):
                    if Path(root).is_dir():
                        handle = (Path(root) / '.dataset-storage.lock').open('a')
                        locks.append(handle)
                        try:
                            mode = fcntl.LOCK_SH if read_only_refresh else fcntl.LOCK_EX
                            fcntl.flock(handle, mode | fcntl.LOCK_NB)
                        except BlockingIOError:
                            report_progress({'stage': 'waiting_lock'})
                            return
                archive = state / 'bundle.tar'
                if phase == 'export':
                    source = DATASET_FILES['inside'](work['path'], roots)
                    manifest_file = state / 'manifest.json'
                    reused_archive = archive.is_file() and manifest_file.is_file()
                    if reused_archive:
                        if read_only_refresh:
                            minimum_free[0] = min(
                                config.get('dataset_refresh_min_free_bytes', DEFAULT_DATASET_REFRESH_MIN_FREE),
                                shutil.disk_usage(state).free)
                        result = json.loads(manifest_file.read_text())
                        if DATASET_FILES['digest'](archive, check, stage='archive_hash') != result['archive_sha256']:
                            archive.unlink()
                            reused_archive = False
                    elif archive.is_file():
                        # A prior archive-hash failure has no durable receipt.
                        # Never transfer an unverified orphan bundle.
                        archive.unlink()
                    if not archive.is_file():
                        manifest = DATASET_FILES['scan'](source, check)
                        size = manifest['total_bytes']
                        archive_budget = size + 10 * 1024 + sum(
                            2048 + 2 * len(item['path'].encode('utf-8')) for item in manifest['files'])
                        used = sum(p.stat().st_size for p in state.parents[1].glob('*/*/bundle*'))
                        if used + archive_budget * 2 > config.get('dataset_staging_quota_bytes', 100 * 1024**3):
                            raise ValueError('Agent 数据集中转配额不足')
                        free = shutil.disk_usage(state).free
                        if free < archive_budget * 2 + 100 * 1024**2:
                            raise ValueError('Agent 中转磁盘空间不足')
                        if read_only_refresh:
                            minimum_free[0] = min(
                                config.get('dataset_refresh_min_free_bytes', DEFAULT_DATASET_REFRESH_MIN_FREE),
                                max(0, free - archive_budget * 2 - 100 * 1024**2))
                        result = DATASET_FILES['pack'](source, archive, check, manifest=manifest)
                        dataset_atomic_json(manifest_file, result)
                    elif reused_archive and read_only_refresh:
                        verify_refresh_source(source, result['fingerprint'], check)
                    offset = dataset_api(config, base + '/chunk', 'POST', {'token': token, 'offset': 0, 'data': ''})['offset']
                    report_progress({'stage': 'transfer', 'processed': offset, 'total': result['archive_size'], 'unit': 'bytes'})
                    with archive.open('rb') as stream:
                        while offset < result['archive_size']:
                            check()
                            stream.seek(offset)
                            chunk = stream.read(512 * 1024)
                            if not chunk:
                                raise ValueError('导出文件在传输中被截断')
                            if read_only_refresh:
                                check.throttle_bytes(len(chunk))
                            offset = dataset_api(config, base + '/chunk', 'POST', {'token': token, 'offset': offset, 'data': base64.b64encode(chunk).decode()})['offset']
                            report_progress({'stage': 'transfer', 'processed': offset, 'total': result['archive_size'], 'unit': 'bytes'})
                    if read_only_refresh:
                        verify_refresh_source(source, result['fingerprint'], check)
                elif phase == 'import':
                    manifest = work['manifest']
                    root = DATASET_FILES['inside'](work['root'], roots)
                    size = manifest['archive_size']
                    if not isinstance(size, int) or size < 0 or size > 51 * 1024**3:
                        raise ValueError('导入文件大小无效')
                    used = sum(p.stat().st_size for p in state.parents[1].glob('*/*/bundle*'))
                    remaining = max(0, size - (archive.stat().st_size if archive.exists() else 0))
                    if used + remaining > config.get('dataset_staging_quota_bytes', 100 * 1024**3):
                        raise ValueError('Agent 数据集中转配额不足')
                    if shutil.disk_usage(state).free < remaining + 100 * 1024**2:
                        raise ValueError('Agent 中转磁盘空间不足')
                    offset = archive.stat().st_size if archive.exists() else 0
                    if offset > size:
                        archive.unlink()
                        offset = 0
                    report_progress({'stage': 'transfer', 'processed': offset, 'total': size, 'unit': 'bytes'})
                    with archive.open('ab') as stream:
                        while offset < size:
                            check()
                            response = dataset_api(config, base + f'/chunk?token={token}&offset={offset}')
                            chunk = base64.b64decode(response['data'], validate=True)
                            if not chunk or response['offset'] != offset or offset + len(chunk) > size:
                                raise ValueError('接收数据集分块不完整')
                            stream.write(chunk)
                            stream.flush()
                            offset += len(chunk)
                            report_progress({'stage': 'transfer', 'processed': offset, 'total': size, 'unit': 'bytes'})
                    if DATASET_FILES['digest'](archive, check, stage='archive_hash') != manifest['archive_sha256']:
                        archive.unlink()
                        raise ValueError('数据集归档 SHA-256 校验失败，请重试')
                    result = DATASET_FILES['publish'](archive, root, work['name'], manifest, token, work['config_text'], check)
                elif phase == 'remove':
                    for guard in work.get('config_guards', []):
                        cfg = DATASET_FILES['inside'](guard['path'], roots)
                        if not cfg.is_file() or DATASET_FILES['digest'](cfg, check) != guard['sha256']:
                            raise ValueError('同节点已登记的数据集配置发生变化，请重新登记核对引用后再清理源数据')
                    result = DATASET_FILES['remove_dataset'](work['path'], roots, work['fingerprint'], check)
                else:
                    raise ValueError('不支持的数据集操作')
            # Persist receipt before ACK. Replayed token never re-deletes a newly
            # created directory or guesses publication success from just a path.
            dataset_atomic_json(receipt, result)
        dataset_api(config, base + '/finish', 'POST', {'token': token, 'phase': phase, 'result': result},
                    timeout=config.get('model_transfer_finalize_timeout_seconds', 1800))
        (state / 'acked').touch()
        (state / 'bundle.tar').unlink(missing_ok=True)
        # Cleanup runs independently in the heartbeat loop, after center
        # reconciliation and acquiring each receipt's cross-process lock.
    except (HTTPError, URLError):
        LOGGER.warning('数据集操作连接中断，将保留断点等待重试：%s', identity)
    except OSError as exc:
        if stop.is_set():
            return
        LOGGER.warning('数据集文件操作失败：%s %s', identity, str(exc)[:200])
        try:
            dataset_api(config, base + '/finish', 'POST', {'token': token, 'phase': phase, 'error': '远程文件访问失败，请检查磁盘、权限和目录：' + str(exc)[:900]})
        except (HTTPError, URLError, OSError, ValueError):
            pass
    except (ValueError, KeyError, TypeError) as exc:
        LOGGER.warning('数据集操作失败：%s %s', identity, str(exc)[:200])
        try:
            dataset_api(config, base + '/finish', 'POST', {'token': token, 'phase': phase, 'error': str(exc)[:1200]})
        except (HTTPError, URLError, OSError, ValueError):
            pass
    finally:
        release_training_lock(training_lock)
        for handle in reversed(locks):
            handle.close()


def recover_active_workers(config: dict, active_training: dict) -> None:
    """Reconcile durable runs before telemetry or fresh claims; never spawn duplicates."""
    active_states = load_active_runs(config)
    for run_id, (worker, _worker_stop) in list(active_training.items()):
        if not worker.is_alive():
            active_training.pop(run_id, None)
    known = set(active_training)
    for active_state in active_states:
        task = active_state["task"]
        run_id = active_state["run_id"]
        if run_id in known:
            continue
        worker_stop = threading.Event()
        identity = (_recovery_identity(active_state)
                    if active_state["phase"] in {"launching", "running"} else None)
        process_lock = False
        if identity:
            try:
                process_lock = process_holds_task_locks(config, task, identity["pid"])
            except (OSError, ValueError, TypeError):
                process_lock = False
        if process_lock:
            worker = threading.Thread(
                target=recover_remote_task,
                args=(config, active_state, worker_stop, None, True),
                name=f"remote-training-{run_id[:8]}", daemon=True)
            LOGGER.info("已识别并接管训练进程 %s（持有其 GPU 与输出目录锁）", run_id)
        elif active_state["phase"] in {"claimed", "confirmed"}:
            worker = threading.Thread(target=run_remote_task,
                                      args=(config, task, worker_stop),
                                      name=f"remote-training-{run_id[:8]}", daemon=True)
            LOGGER.info("正在恢复尚未启动的远程任务 %s", run_id)
        else:
            worker = threading.Thread(target=recover_remote_task,
                                      args=(config, active_state, worker_stop),
                                      name=f"remote-training-{run_id[:8]}", daemon=True)
            LOGGER.info("正在恢复远程任务 %s", run_id)
        active_training[run_id] = (worker, worker_stop)
        worker.start()


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO Forge 算力监控与可选远程训练 Agent")
    parser.add_argument("--config", required=True, type=Path, help="平台下载的 JSON 配置")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="采集并上报一次后退出，用于验证连接")
    mode.add_argument("--check-config", action="store_true", help="仅在本机检查配置和训练环境，不连接中心")
    parser.add_argument("--shutdown-policy", choices=("preserve", "stop"), default="preserve",
                        help="退出策略：preserve 保留训练（默认），stop 同时停止训练；平台停止按钮始终有效")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        if args.check_config:
            LOGGER.error("配置检查失败：%s", exc)
        else:
            LOGGER.error("配置无效或无法读取。请使用平台生成的 JSON 文件并设置权限 chmod 600。")
        return 2
    if args.check_config:
        LOGGER.info("model_test_roots（独立模型测试可读目录）：%s",
                    "、".join(config["model_test_roots"]) or "未配置")
        LOGGER.info("dataset_roots（数据集可管理范围）：%s", '、'.join(config['dataset_roots']))
        LOGGER.info("数据集中转配额：%s 字节", config['dataset_staging_quota_bytes'])
        LOGGER.info("远程文件最终校验超时：%s 秒", config["model_transfer_finalize_timeout_seconds"])
        if config["enable_training"]:
            LOGGER.info("训练工作目录：%s", config["training_cwd"])
            LOGGER.info("allowed_training_roots（允许访问）：%s", "、".join(config["allowed_training_roots"]))
            LOGGER.info("training_output_roots（任务输出可选目录）：%s",
                        "、".join(str(root) for root in configured_output_roots(config)))
        capability = training_capability(config)
        if config["enable_training"] and not capability["ready"]:
            LOGGER.error("配置格式有效，但训练环境未就绪：%s", capability["reason"])
            return 1
        if config["enable_training"]:
            LOGGER.info("配置检查通过；远程训练已就绪；YOLO %s；共享训练锁%s占用",
                        capability["yolo_version"] or "版本未知", "已" if capability["lock_occupied"] else "未")
        else:
            LOGGER.info("配置检查通过；当前仅启用算力监控")
        return 0
    try:
        instance_lock = acquire_agent_instance_lock(config)
    except OSError as exc:
        LOGGER.error("Agent 单实例锁创建失败：%s", exc)
        return 2
    if instance_lock is None:
        LOGGER.error("同一配置已有 Agent 进程在运行，本进程已退出")
        return 3

    def finish(code: int) -> int:
        release_training_lock(instance_lock)
        return code
    try:
        register_workspace(config)
    except (OSError, ValueError, TypeError) as exc:
        LOGGER.error("Agent 实例与共享训练工作区登记失败，已停止启动：%s", exc)
        return finish(2)
    stop = threading.Event()
    reload_requested = False

    def shutdown(signum, _frame):
        nonlocal reload_requested
        if signum == signal.SIGHUP:
            reload_requested = True
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, shutdown)
    failures = 0
    interval = config["report_interval_seconds"]
    active_training: dict[str, tuple[threading.Thread, threading.Event]] = {}
    active_operation: threading.Thread | None = None
    active_dataset_operation: threading.Thread | None = None
    active_dataset_phase: str | None = None
    active_dataset_read_only_refresh = False
    dataset_reclamation: threading.Thread | None = None
    last_unready_reason: str | None = None
    LOGGER.info("监控 Agent 已启动，节点 %s，Agent %s；远程训练 %s", config["node_id"], VERSION,
                "已启用" if config["enable_training"] else "未启用")
    while not stop.is_set():
        recovery_ready = True
        if config["enable_training"] and not args.once:
            try:
                recover_active_workers(config, active_training)
            except (OSError, ValueError, TypeError, KeyError):
                recovery_ready = False
                LOGGER.exception("活动训练核验失败；暂停领取新任务，保留进程和记录")
        if stop.is_set():
            break
        try:
            payload = collect(config["nvidia_smi_bin"], config)
            interval, refresh_request_id = send(config, payload)
            if refresh_request_id and not args.once:
                refreshed = fulfill_refresh_request(config, refresh_request_id, stop)
                if refreshed is not None:
                    payload = refreshed
            capability = payload.get("training") or {}
            training_ready = bool(capability.get("ready"))
            if failures or args.once:
                LOGGER.info("采样上报成功：GPU %s，进程 %s", len(payload["compute"]["gpus"]),
                            sum(len(gpu["processes"]) for gpu in payload["compute"]["gpus"]))
            failures = 0
            delay = interval
            if (config.get('dataset_roots') and not args.once
                    and (dataset_reclamation is None or not dataset_reclamation.is_alive())
                    and time.time() - config.get('_dataset_reconcile_at', 0) >= 300):
                dataset_reclamation = threading.Thread(target=run_dataset_reclamation, args=(config,),
                                                      name='dataset-reclamation', daemon=True)
                dataset_reclamation.start()
            if config.get('dataset_roots') and not args.once and (active_dataset_operation is None or not active_dataset_operation.is_alive()):
                try:
                    busy = not recovery_ready or bool(active_training or load_active_runs(config)) or bool(capability.get('lock_occupied'))
                    work = poll_dataset_operation(config, busy)
                    if work:
                        active_dataset_phase = work['phase']
                        active_dataset_read_only_refresh = concurrent_refresh_work(work)
                        active_dataset_operation = threading.Thread(target=run_dataset_operation, args=(config, work, stop), name='dataset-operation', daemon=True)
                        active_dataset_operation.start()
                except HTTPError as exc:
                    if exc.code != 404:  # Older centres do not expose this additive API.
                        LOGGER.warning('数据集操作轮询失败，下次心跳重试')
                except (URLError, OSError, ValueError):
                    LOGGER.warning('数据集操作轮询失败，下次心跳重试')
            if config["enable_training"] and not args.once and (active_operation is None or not active_operation.is_alive()):
                try:
                    work = api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/operations/poll", "POST", {}).get("operation")
                    if work:
                        active_operation = threading.Thread(target=run_operations, args=(config, work, stop),
                                                            name="remote-file-operation", daemon=True)
                        active_operation.start()
                except (HTTPError, URLError, OSError, ValueError, TypeError):
                    LOGGER.warning("远程文件操作轮询失败，下次心跳重试")
            if config["enable_training"] and not args.once:
                try:
                    pending_finals_ready = retry_pending_finals(config)
                    if not pending_finals_ready:
                        delay = max(delay, interval)
                    active_states = load_active_runs(config)
                    legacy_active = any(not (state["task"].get("gpu_assignment") or [])
                                        for state in active_states)
                    if dataset_blocks_training(active_dataset_operation, active_dataset_phase, active_dataset_read_only_refresh):
                        pass  # Finish file work before claiming a fresh training run.
                    elif legacy_active:
                        # Old centers and CPU runs keep the node-wide serial contract.
                        pass
                    elif not recovery_ready:
                        pass  # Do not claim while an existing run cannot be verified.
                    elif not pending_finals_ready:
                        # Keep retrying terminal acknowledgments before asking the
                        # center for more work, so local state cannot grow unbounded.
                        LOGGER.warning("仍有训练最终状态待平台确认，暂停领取新任务")
                    elif not training_ready:
                        reason = str(capability.get("reason") or "训练环境未就绪")
                        if reason != last_unready_reason:
                            LOGGER.warning("远程训练未就绪，暂停领取新任务：%s", reason)
                            last_unready_reason = reason
                    else:
                        last_unready_reason = None
                        # The center arbitrates each GPU UUID reservation. Continue
                        # claiming while it has independently schedulable work.
                        for _ in range(16):
                            if stop.is_set():
                                break
                            response = api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/poll", "POST", {})
                            task = response.get("task")
                            if not task:
                                break
                            run_id = task["run_id"]
                            save_active_run(config, task, "claimed")
                            worker_stop = threading.Event()
                            worker = threading.Thread(target=run_remote_task,
                                                      args=(config, task, worker_stop),
                                                      name=f"remote-training-{run_id[:8]}", daemon=True)
                            active_training[run_id] = (worker, worker_stop)
                            worker.start()
                            LOGGER.info("已领取远程任务 %s / Attempt %s / Run %s", task.get("task_id"),
                                        task.get("attempt_number"), run_id)
                            if not (task.get("gpu_assignment") or []):
                                # The payload came from an older center or CPU path.
                                # Never ask it for another task in this cycle.
                                break
                except (HTTPError, URLError, OSError, ValueError, TypeError):
                    LOGGER.warning("远程任务轮询失败，将在下次心跳重试")
        except HTTPError as exc:
            # Do not log response bodies, headers or configuration containing secrets.
            LOGGER.warning("上报返回 HTTP %s。401 请检查 Token；403 请检查节点是否禁用；422 请检查采样与时钟。", exc.code)
            if args.once:
                return finish(1)
            failures += 1
            delay = min(60, 5 * 2 ** min(failures, 4))
        except (URLError, OSError, ValueError, TypeError):
            LOGGER.warning("采样或连接失败，请检查中心地址、网络、证书和 nvidia-smi。将自动重试。")
            if args.once:
                return finish(1)
            failures += 1
            delay = min(60, 5 * 2 ** min(failures, 4))
        if args.once:
            return finish(0)
        stop.wait(delay + (random.random() if failures else 0))
    for _worker, worker_stop in active_training.values():
        worker_stop.preserve_training = reload_requested or args.shutdown_policy == "preserve"
        worker_stop.set()
    for worker, _worker_stop in active_training.values():
        worker.join()
    if active_operation and active_operation.is_alive():
        active_operation.join()
    if active_dataset_operation and active_dataset_operation.is_alive():
        active_dataset_operation.join()
    LOGGER.info("Agent 已停止；训练退出策略：%s",
                "保留训练" if reload_requested or args.shutdown_policy == "preserve" else "停止训练")
    if reload_requested:
        # Exec keeps MAINPID and the service cgroup intact. Do not release the
        # instance flock before exec: its non-inheritable FD closes atomically.
        # All training workers have saved state and closed only their lock copies.
        LOGGER.info("正在重载 Agent；训练进程保持运行")
        try:
            os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        except OSError:
            LOGGER.exception("Agent 重载失败；训练仍保留，请重新启动 Agent")
            return finish(1)
    return finish(0)



DATASET_PROVENANCE_SOURCE = '"""Small, portable evidence of the dataset configuration used by a run.\n\nA catalogue fingerprint describes the last completed validation, not an\nimmutable copy of the training images. Never scan a large dataset at launch.\n"""\nfrom __future__ import annotations\n\nimport hashlib\nimport re\nfrom datetime import datetime, timezone\nfrom pathlib import Path\n\n\nMAX_CONFIG_BYTES = 1024 * 1024\n\n\ndef capture(raw, *, task_type=\'detect\', node_id=\'local\', runtime_path=None):\n    snapshot = {\n        \'version\': 1, \'node_id\': str(node_id), \'task_type\': str(task_type),\n        \'source\': \'runtime_config\', \'verification\': \'path_only\' if task_type == \'classify\' else \'config_only\',\n        \'captured_at\': datetime.now(timezone.utc).isoformat(),\n    }\n    if task_type == \'classify\':\n        return snapshot\n    try:\n        with Path(raw).open(\'rb\') as handle:\n            content = handle.read(MAX_CONFIG_BYTES + 1)\n        if len(content) <= MAX_CONFIG_BYTES:\n            snapshot[\'config_sha256\'] = hashlib.sha256(content).hexdigest()\n        else:\n            snapshot[\'verification\'] = \'unavailable\'\n    except OSError:\n        snapshot[\'verification\'] = \'unavailable\'\n    if runtime_path is not None:\n        try:\n            with Path(runtime_path).open(\'rb\') as handle:\n                content = handle.read(MAX_CONFIG_BYTES + 1)\n            if len(content) <= MAX_CONFIG_BYTES:\n                snapshot[\'runtime_config_sha256\'] = hashlib.sha256(content).hexdigest()\n        except OSError:\n            pass\n    return snapshot\n\n\ndef clean(snapshot):\n    """Accept only bounded configuration evidence from an authenticated Agent."""\n    if not isinstance(snapshot, dict) or snapshot.get(\'version\') != 1:\n        raise ValueError(\'数据集快照版本无效\')\n    result = {\'version\': 1, \'source\': \'runtime_config\'}\n    for key in (\'node_id\', \'task_type\', \'captured_at\'):\n        value = snapshot.get(key)\n        if not isinstance(value, str) or not value or len(value) > 100:\n            raise ValueError(\'数据集快照字段无效\')\n        result[key] = value\n    try:\n        stamp = datetime.fromisoformat(result[\'captured_at\'].replace(\'Z\', \'+00:00\'))\n        if stamp.tzinfo is None:\n            raise ValueError()\n    except ValueError:\n        raise ValueError(\'数据集快照时间无效\') from None\n    verification = snapshot.get(\'verification\')\n    if verification not in (\'config_only\', \'path_only\', \'unavailable\'):\n        raise ValueError(\'数据集快照验证类型无效\')\n    result[\'verification\'] = verification\n    for key in (\'config_sha256\', \'runtime_config_sha256\'):\n        digest = snapshot.get(key)\n        if digest is not None:\n            if not isinstance(digest, str) or not re.fullmatch(r\'[0-9a-f]{64}\', digest):\n                raise ValueError(\'数据集配置摘要无效\')\n            result[key] = digest\n    return result\n'
DATASET_PROVENANCE = {}
exec(compile(DATASET_PROVENANCE_SOURCE, "<dataset_provenance>", "exec"), DATASET_PROVENANCE)

if __name__ == "__main__":
    raise SystemExit(main())
