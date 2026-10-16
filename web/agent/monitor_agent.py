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
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

VERSION = "1.6.0"
TRACE_VERSION = 1
TRACE_MAXIMUM = 24
TRACE_INTERVAL = 10
LOGGER = logging.getLogger("yolo-monitor-agent")


REMOTE_TRACE_RUNNER = r'''#!/usr/bin/env python3
import argparse, json, os, sys, time
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

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--trace-dir", required=True); parser.add_argument("--maximum", type=int, default=24); parser.add_argument("--interval", type=int, default=10); parser.add_argument("--command-json", required=True)
    args, _ = parser.parse_known_args()
    from ultralytics import YOLO
    command = json.loads(args.command_json); values = {}
    for item in command[3:]:
        if "=" in item:
            key, value = item.split("=", 1)
            try:
                from ultralytics.cfg import smart_value; value = smart_value(value)
            except Exception: pass
            values[key] = value
    model = YOLO(str(values.pop("model")), task=str(values.pop("task", command[1])))
    values.pop("mode", None); collector = Collector(args.trace_dir, args.maximum, args.interval)
    model.add_callback("on_train_start", collector.start); model.add_callback("on_train_batch_start", collector.batch_start); model.add_callback("on_train_batch_end", collector.batch_end); model.add_callback("on_train_end", collector.end)
    model.train(**values)
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


def collect_gpus(binary: str) -> tuple[list[dict], str | None]:
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


def training_capability(config: dict) -> dict:
    """Report whether this Agent can safely accept a remote training run."""
    enabled = bool(config.get("enable_training"))
    if not enabled:
        return {"enabled": False, "ready": False, "reason": "远程训练未启用",
                "supported_parameters": sorted(TRAINING_KEYS), "yolo_version": None,
                "python_version": platform.python_version(), "output_writable": False,
                "lock_occupied": False}
    try:
        cwd = Path(config["training_cwd"])
        roots = [Path(item) for item in config["allowed_training_roots"]]
        if not cwd.is_dir():
            raise ValueError("训练工作目录不存在")
        if not roots:
            raise ValueError("未配置允许的训练目录")
        output_root = roots[0]
        output_root.mkdir(parents=True, exist_ok=True)
        probe = output_root / ".platform-write-check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        yolo = config["yolo_bin"]
        if "/" in yolo and (not Path(yolo).is_file() or not os.access(yolo, os.X_OK)):
            raise ValueError("YOLO 可执行文件不存在或不可执行")
        return {"enabled": True, "ready": True, "reason": None,
                "supported_parameters": sorted(TRAINING_KEYS), "yolo_version": None,
                "python_version": platform.python_version(), "output_writable": True,
                "lock_occupied": False}
    except (OSError, ValueError, KeyError) as exc:
        return {"enabled": True, "ready": False, "reason": str(exc)[:400],
                "supported_parameters": sorted(TRAINING_KEYS), "yolo_version": None,
                "python_version": platform.python_version(), "output_writable": False,
                "lock_occupied": False}


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
    gpus, message = collect_gpus(binary)
    payload = {
        "agent_version": VERSION,
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
    binary = config.get("nvidia_smi_bin", "nvidia-smi")
    if not isinstance(binary, str) or not binary:
        raise ValueError("nvidia_smi_bin 必须是命令或可执行文件路径")
    enabled = config.get("enable_training", False)
    if type(enabled) is not bool:
        raise ValueError("enable_training 必须为布尔值")
    roots = config.get("allowed_training_roots", [])
    if not isinstance(roots, list) or any(not isinstance(root, str) or not Path(root).is_absolute() for root in roots):
        raise ValueError("allowed_training_roots 必须是绝对路径列表")
    if enabled and not roots:
        raise ValueError("启用训练时必须配置 allowed_training_roots")
    trace_enabled = config.get("enable_training_trace", True)
    if type(trace_enabled) is not bool:
        raise ValueError("enable_training_trace 必须为布尔值")
    yolo_bin = config.get("yolo_bin", "yolo")
    training_cwd = config.get("training_cwd", "/data/yolo/yolo")
    if not isinstance(yolo_bin, str) or not yolo_bin or not isinstance(training_cwd, str) or not Path(training_cwd).is_absolute():
        raise ValueError("yolo_bin/training_cwd 配置无效")
    cwd_path = Path(training_cwd).resolve(strict=True) if enabled else Path(training_cwd).resolve()
    lock_value = config.get("training_lock_file", str(cwd_path / ".yolo-forge-training.lock"))
    if not isinstance(lock_value, str) or not Path(lock_value).is_absolute():
        raise ValueError("training_lock_file 必须是绝对路径")
    lock_path = Path(lock_value).resolve()
    if enabled and not lock_path.parent.is_dir():
        raise ValueError("training_lock_file 的父目录必须已存在")
    return {"center_url": center, "node_id": node_id, "token": token,
            "report_interval_seconds": interval, "nvidia_smi_bin": binary,
            "enable_training": enabled, "allowed_training_roots": [str(Path(item).resolve()) for item in roots],
            "yolo_bin": yolo_bin, "training_cwd": str(cwd_path),
            "training_lock_file": str(lock_path),
            "training_state_dir": str(path.resolve().parent / ".training-state"),
            "enable_training_trace": trace_enabled}


def acquire_training_lock(config: dict):
    """Try to reserve this physical training workspace across Agent processes."""
    lock_path = Path(config.get("training_lock_file") or Path(config["training_cwd"]) / ".yolo-forge-training.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    os.chmod(lock_path, 0o660)
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


def release_training_lock(handle) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def api_json(config: dict, path: str, method: str = "GET", payload: dict | None = None,
             extra_headers: dict[str, str] | None = None) -> dict:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if payload is not None else None
    headers = {"Authorization": "Bearer " + config["token"], "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    request = Request(config["center_url"] + path, data=data, method=method, headers=headers)
    with build_opener(NoRedirect()).open(request, timeout=20) as response:
        body = response.read(2 * 1024 * 1024)
    return json.loads(body or b"{}")


TRAINING_KEYS = {"patience", "lr0", "pretrained", "cache", "amp", "close_mosaic", "cos_lr", "weight_decay", "momentum", "warmup_epochs", "freeze", "multi_scale", "rect", "seed", "deterministic", "fraction", "save_period", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "box", "cls", "dfl", "save", "verbose", "single_cls", "classes", "profile", "compile", "overlap_mask", "mask_ratio", "dropout", "val", "split", "save_json", "conf", "iou", "max_det", "plots", "lrf", "warmup_momentum", "warmup_bias_lr", "nbs", "bgr", "cutmix", "copy_paste_mode", "auto_augment", "erasing", "cls_pw", "pose", "kobj", "rle", "angle", "dlog", "dgrad", "dlam"}
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
        raise ValueError(f"远程服务器上的{label}不在 Agent 允许目录内：{candidate}")
    return candidate


def scalar_parameter(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)) and math.isfinite(value):
        return str(value)
    if isinstance(value, str) and len(value) <= 500 and not any(ord(char) < 32 for char in value):
        return value
    raise ValueError("不支持的参数类型")


def build_training_command(config: dict, task: dict) -> list[str]:
    params = task["parameters"]
    roots = config["allowed_training_roots"]
    model = within_allowed(task["model_path"], roots, label="基础模型文件")
    dataset = within_allowed(task["data_path"], roots, label="数据集 YAML 文件")
    run_dir = Path(task["run_dir"]).resolve()
    output_root = within_allowed(params["remote_output_root"], roots, file_required=False,
                                 label="训练输出根目录")
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


def operation_paths(config: dict, payload: dict, *, allow_missing_root: bool = False) -> tuple[Path, Path | None] | None:
    """Validate all deletion/scan/fetch targets against an exact task/Attempt layout."""
    task_id, task_type = payload["task_id"], payload["task_type"]
    attempt = payload["attempt_number"]
    if (not re.fullmatch(r"[a-zA-Z0-9_-]{1,160}", task_id) or
            task_type not in {"detect", "segment", "classify", "pose", "obb"} or
            type(attempt) is not int or attempt < 1):
        raise ValueError("远程文件操作任务信息无效")
    root = Path(payload["output_root"])
    root_missing = not root.exists()
    if (not root.is_absolute() or root.is_symlink() or root.resolve() != root or
            (not root_missing and not root.is_dir()) or (root_missing and not allow_missing_root)):
        raise ValueError("输出根目录无效或包含符号链接")
    run_dir = Path(payload["run_dir"])
    expected = root / task_type / task_id / f"attempt-{attempt:03d}"
    if run_dir != expected or run_dir.resolve() != expected or run_dir.is_symlink():
        raise ValueError("只能操作该任务的独立 Attempt 目录")
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
    # A task created for another server may reference a root that has never
    # existed here. A delete can safely become a no-op, but scan/fetch cannot.
    if root_missing:
        return None
    if not any(root == Path(item) or Path(item) in root.parents for item in config["allowed_training_roots"]):
        raise ValueError("输出根目录不在 Agent 允许目录内")
    return run_dir, log


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


def delete_remote_files(config: dict, payload: dict) -> None:
    # Validate the complete plan before deleting the first file. rmtree never follows
    # directory symlinks inside the owned Attempt; the Attempt itself cannot be one.
    # A failed launch may not have created the output root yet. Treat that
    # exact no-file state as already cleaned while retaining all path checks.
    plans = [operation_paths(config, run, allow_missing_root=True) for run in payload["runs"]]
    targets = [paths for paths in plans if paths is not None]
    protected = [Path(path).resolve() for path in payload.get("protected_paths", [])]
    protected.extend([Path(config["training_cwd"]).resolve(), Path(config["yolo_bin"]).resolve()])
    for run_dir, log in targets:
        if (run_dir.exists() and not run_dir.is_dir()) or (log and log.exists() and not log.is_file()):
            raise ValueError("Attempt 或日志的文件类型不正确，已拒绝清理")
        for path in protected:
            if path == run_dir or run_dir in path.parents or (log and path == log):
                raise ValueError("任务目录包含仍被引用的模型、数据或训练程序，已拒绝清理")
    for run_dir, log in targets:
        if run_dir.exists():
            if not run_dir.is_dir():
                raise ValueError("Attempt 路径不是目录")
            shutil.rmtree(run_dir)
        if log:
            log.unlink(missing_ok=True)
        for parent in {run_dir.parent, log.parent if log else run_dir.parent}:
            try:
                parent.rmdir()  # remove empty task directory only
            except OSError:
                pass


def execute_operation(config: dict, operation: dict, stop: threading.Event) -> None:
    endpoint = f"/api/compute/nodes/{config['node_id']}/agent/operations/{operation['id']}"
    result = {"lease": operation["lease"], "success": True}
    try:
        payload = operation["payload"]
        if operation["kind"] == "delete":
            delete_remote_files(config, payload)
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
                    for retry in range(3):
                        try:
                            response = api_json(config, endpoint + "/chunk", "POST", chunk)
                            break
                        except (OSError, URLError):
                            if retry == 2 or stop.wait(2):
                                raise
                    if response.get("status") == "succeeded":
                        return
                    offset += len(data)
                    if not data:
                        raise ValueError("文件读取不完整")
                return  # lease expires and a restarted Agent can retry from zero
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


def read_remote_progress(run_dir: Path) -> tuple[int, dict, list[dict]]:
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return 0, {}, []
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return 0, {}, []
    metrics = []
    latest = {}
    completed = 0
    for row in rows:
        try:
            epoch = int(float((row.get("epoch") or "0").strip()))
        except (ValueError, AttributeError):
            continue
        values = {}
        for key, raw in row.items():
            if not key or key.strip() in {"epoch", "time"} or raw is None:
                continue
            try:
                value = float(raw.strip())
                values[key.strip()] = value if math.isfinite(value) else None
            except (ValueError, AttributeError):
                values[key.strip()] = raw[:200]
        try:
            elapsed = float(row.get("time") or 0)
        except ValueError:
            elapsed = 0
        completed = max(completed, epoch)
        latest = values
        metrics.append({"epoch": epoch, "elapsed": elapsed, "values": values})
    return completed, latest, metrics


def recent_log_lines(path: Path) -> list[str]:
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\r\n")[:4000] for line in deque(handle, maxlen=100)]
    except OSError:
        return []


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
    if payload.get("status") not in {"completed", "stopped", "failed"}:
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
    if len(metrics) > 256 and response.get("status") not in {"completed", "stopped", "failed"}:
        save_pending_final(config, run_id, dict(payload, metrics=metrics[256:]))
        return False
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
    """Wrap remote detect training with the bundled real-batch collector."""
    if not config.get("enable_training_trace", False) or task.get("task_type") != "detect":
        return command
    trace_dir = Path(task["run_dir"]) / "training-trace"
    state_dir = Path(config["training_state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    runner = state_dir / "training_trace_runner.py"
    if not runner.is_file() or runner.read_text(encoding="utf-8") != REMOTE_TRACE_RUNNER:
        runner.write_text(REMOTE_TRACE_RUNNER, encoding="utf-8")
        os.chmod(runner, 0o700)
    yolo_bin = str(config.get("yolo_bin", ""))
    python_bin = str(Path(yolo_bin).parent / "python") if "/" in yolo_bin else sys.executable
    if not Path(python_bin).is_file():
        python_bin = sys.executable
    return [python_bin, str(runner), "--trace-dir", str(trace_dir),
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


def _run_remote_task(config: dict, task: dict, stop: threading.Event) -> None:
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
        if response.get("stop_requested"):
            request_stop("platform_stop")

    try:
        command = prepare_trace_command(config, task, build_training_command(config, task))
        log_path = Path(task["log_path"])
        with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] REMOTE COMMAND {shlex.join(command)}\n")
            environment = os.environ.copy()
            environment["MKL_THREADING_LAYER"] = "GNU"
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(command, cwd=config["training_cwd"], env=environment, stdout=log_handle,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            report_progress({"status": "running", "pid": process.pid, "run_dir": task["run_dir"],
                             "completed_epochs": task.get("completed_epochs", 0),
                             "progress": task.get("progress", 0), "latest_metrics": task.get("latest_metrics", {})})
            while process.poll() is None and not stop.is_set():
                if stop.wait(2):
                    break
                try:
                    control = api_json(config, endpoint + "/control")
                    if control.get("stop"):
                        request_stop("platform_stop")
                except (OSError, ValueError, HTTPError, URLError):
                    pass
                completed, latest, metrics = read_remote_progress(Path(task["run_dir"]))
                completed = max(completed, task.get("completed_epochs", 0))
                latest = latest or task.get("latest_metrics", {})
                if time.monotonic() - last_manifest_at >= 30:
                    try:
                        report_manifest(config, manifest_payload)
                        last_manifest_at = time.monotonic()
                    except (OSError, ValueError, HTTPError, URLError):
                        last_manifest_at = time.monotonic()
                new_metrics = [item for item in metrics if item["epoch"] not in last_metrics][:256]
                if time.monotonic() - last_trace_at >= 10:
                    try:
                        report_training_trace(config, task, sent_trace_images)
                    except (OSError, ValueError, HTTPError, URLError, json.JSONDecodeError):
                        pass
                    last_trace_at = time.monotonic()
                log_lines = recent_log_lines(log_path)
                status = "stopping" if stop.is_set() or requested_stop else "running"
                report_progress({"status": status, "pid": process.pid,
                             "completed_epochs": completed, "progress": min(99.99, completed / max(1, int(task['parameters']['epochs'])) * 100),
                             "latest_metrics": latest, "metrics": new_metrics, "log_tail": log_lines,
                             "run_dir": task["run_dir"]})
            if stop.is_set() and process.poll() is None:
                request_stop("agent_shutdown")
            try:
                exit_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
        completed, latest, metrics = read_remote_progress(Path(task["run_dir"]))
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
        final_payload = {"_claim_lease": lease, "status": final_status, "completed_epochs": completed,
                         "progress": 100 if final_status == "completed" else min(99.99, completed / max(1, int(task['parameters']['epochs'])) * 100),
                         "latest_metrics": latest, "metrics": [item for item in metrics if item["epoch"] not in last_metrics],
                         "log_tail": log_lines, "exit_code": exit_code, "command": command,
                         "error_message": error, "stop_reason": stop_reason, "run_dir": task["run_dir"]}
        pending_path = save_pending_final(config, run_id, final_payload)
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
            pending_path = save_pending_final(config, run_id, {"_claim_lease": lease, "status": "failed", "error_message": str(exc)[:4000], "exit_code": 1,
                                                             "completed_epochs": task.get("completed_epochs", 0),
                                                             "progress": task.get("progress", 0), "latest_metrics": task.get("latest_metrics", {}),
                                                             "log_tail": recent_log_lines(Path(task["log_path"]))})
            deliver_pending_final(config, pending_path)
        except Exception:
            LOGGER.exception("远程训练错误状态上报失败")
        LOGGER.error("远程训练任务失败：%s", exc)


def run_remote_task(config: dict, task: dict, stop: threading.Event, training_lock=None) -> None:
    """Run one task while holding the server-wide lock until all final reporting finishes."""
    lock = training_lock or acquire_training_lock(config)
    if lock is None:
        LOGGER.warning("本机已有另一个 Agent 在训练，拒绝并发启动任务 %s", task.get("run_id"))
        return
    try:
        lease = task.get("claim_lease")
        if lease:
            api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/runs/{task['run_id']}/confirm",
                     "POST", {"lease": lease})
        _run_remote_task(config, task, stop)
    finally:
        release_training_lock(lock)


def send(config: dict, payload: dict) -> int:
    request = Request(config["center_url"] + "/api/compute/nodes/" + config["node_id"] + "/report",
                      data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(), method="POST",
                      headers={"Authorization": "Bearer " + config["token"], "Content-Type": "application/json"})
    with build_opener(NoRedirect()).open(request, timeout=15) as response:
        result = json.loads(response.read(65536))
    interval = result.get("report_interval_seconds", config["report_interval_seconds"])
    return interval if type(interval) is int and 5 <= interval <= 300 else config["report_interval_seconds"]


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO Forge 算力监控与可选远程训练 Agent")
    parser.add_argument("--config", required=True, type=Path, help="平台下载的 JSON 配置")
    parser.add_argument("--once", action="store_true", help="采集并上报一次后退出，用于验证连接")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config)
    except (OSError, ValueError, TypeError):
        LOGGER.error("配置无效或无法读取。请使用平台生成的 JSON 文件并设置权限 chmod 600。")
        return 2
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    failures = 0
    interval = config["report_interval_seconds"]
    active_training: threading.Thread | None = None
    active_operation: threading.Thread | None = None
    training_stop = threading.Event()
    LOGGER.info("监控 Agent 已启动，节点 %s，Agent %s；远程训练 %s", config["node_id"], VERSION,
                "已启用" if config["enable_training"] else "未启用")
    while not stop.is_set():
        try:
            payload = collect(config["nvidia_smi_bin"], config)
            interval = send(config, payload)
            if failures or args.once:
                LOGGER.info("采样上报成功：GPU %s，进程 %s", len(payload["compute"]["gpus"]),
                            sum(len(gpu["processes"]) for gpu in payload["compute"]["gpus"]))
            failures = 0
            delay = interval
            if config["enable_training"] and not args.once and (active_operation is None or not active_operation.is_alive()):
                try:
                    work = api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/operations/poll", "POST", {}).get("operation")
                    if work:
                        active_operation = threading.Thread(target=run_operations, args=(config, work, stop),
                                                            name="remote-file-operation", daemon=True)
                        active_operation.start()
                except (HTTPError, URLError, OSError, ValueError, TypeError):
                    LOGGER.warning("远程文件操作轮询失败，下次心跳重试")
            if config["enable_training"] and not args.once and (active_training is None or not active_training.is_alive()):
                training_lock = None
                try:
                    if not retry_pending_finals(config):
                        stop.wait(delay)
                        continue
                    training_lock = acquire_training_lock(config)
                    if training_lock is None:
                        stop.wait(min(delay, 5))
                        continue
                    response = api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/poll", "POST", {})
                    task = response.get("task")
                    if task:
                        training_stop = threading.Event()
                        active_training = threading.Thread(target=run_remote_task, args=(config, task, training_stop, training_lock),
                                                          name="remote-training", daemon=True)
                        active_training.start()
                        training_lock = None  # ownership transferred to the training thread
                        LOGGER.info("已领取远程任务 %s / Attempt %s", task.get("task_id"), task.get("attempt_number"))
                except (HTTPError, URLError, OSError, ValueError, TypeError):
                    LOGGER.warning("远程任务轮询失败，将在下次心跳重试")
                finally:
                    release_training_lock(training_lock)
        except HTTPError as exc:
            # Do not log response bodies, headers or configuration containing secrets.
            LOGGER.warning("上报返回 HTTP %s。401 请检查 Token；403 请检查节点是否禁用；422 请检查采样与时钟。", exc.code)
            if args.once:
                return 1
            failures += 1
            delay = min(60, 5 * 2 ** min(failures, 4))
        except (URLError, OSError, ValueError, TypeError):
            LOGGER.warning("采样或连接失败，请检查中心地址、网络、证书和 nvidia-smi。将自动重试。")
            if args.once:
                return 1
            failures += 1
            delay = min(60, 5 * 2 ** min(failures, 4))
        if args.once:
            return 0
        stop.wait(delay + (random.random() if failures else 0))
    if active_training and active_training.is_alive():
        training_stop.set()
        active_training.join(timeout=20)
    if active_operation and active_operation.is_alive():
        active_operation.join(timeout=20)
    LOGGER.info("Agent 已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
