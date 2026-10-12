#!/usr/bin/env python3
"""YOLO Forge compute Agent with opt-in remote training. Python 3.9+."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import re
import shlex
import signal
import socket
import stat
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

VERSION = "1.1.0"
LOGGER = logging.getLogger("yolo-monitor-agent")


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


def collect(binary: str = "nvidia-smi") -> dict:
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
    return {
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
    yolo_bin = config.get("yolo_bin", "yolo")
    training_cwd = config.get("training_cwd", "/data/yolo/yolo")
    if not isinstance(yolo_bin, str) or not yolo_bin or not isinstance(training_cwd, str) or not Path(training_cwd).is_absolute():
        raise ValueError("yolo_bin/training_cwd 配置无效")
    cwd_path = Path(training_cwd).resolve(strict=True) if enabled else Path(training_cwd).resolve()
    return {"center_url": center, "node_id": node_id, "token": token,
            "report_interval_seconds": interval, "nvidia_smi_bin": binary,
            "enable_training": enabled, "allowed_training_roots": [str(Path(item).resolve()) for item in roots],
            "yolo_bin": yolo_bin, "training_cwd": str(cwd_path)}


def api_json(config: dict, path: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if payload is not None else None
    request = Request(config["center_url"] + path, data=data, method=method,
                      headers={"Authorization": "Bearer " + config["token"], "Content-Type": "application/json"})
    with build_opener(NoRedirect()).open(request, timeout=20) as response:
        body = response.read(2 * 1024 * 1024)
    return json.loads(body or b"{}")


TRAINING_KEYS = {"patience", "lr0", "pretrained", "cache", "amp", "close_mosaic", "cos_lr", "weight_decay", "momentum", "warmup_epochs", "freeze", "multi_scale", "rect", "seed", "deterministic", "fraction", "save_period", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "box", "cls", "dfl", "save", "verbose", "single_cls", "classes", "profile", "compile", "overlap_mask", "mask_ratio", "dropout", "val", "split", "save_json", "conf", "iou", "max_det", "plots", "lrf", "warmup_momentum", "warmup_bias_lr", "nbs", "bgr", "cutmix", "copy_paste_mode", "auto_augment", "erasing", "cls_pw", "pose", "kobj", "rle", "angle", "dlog", "dgrad", "dlam"}
RESERVED_KEYS = {"model", "data", "project", "name", "device", "epochs", "batch", "imgsz", "workers", "optimizer", "task", "mode", "exist_ok", "save_dir", "resume", "source"}


def within_allowed(path: str, roots: list[str], *, file_required: bool = True) -> Path:
    candidate = Path(path).resolve(strict=True)
    if file_required and not candidate.is_file():
        raise ValueError(f"要求文件不存在：{candidate}")
    if not file_required and not candidate.is_dir():
        raise ValueError(f"要求目录不存在：{candidate}")
    if not any(candidate == Path(root) or Path(root) in candidate.parents for root in roots):
        raise ValueError(f"路径不在 Agent 允许目录内：{candidate}")
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
    model = within_allowed(task["model_path"], roots)
    dataset = within_allowed(task["data_path"], roots)
    run_dir = Path(task["run_dir"]).resolve()
    output_root = Path(params["remote_output_root"]).resolve(strict=True)
    if not output_root.is_dir() or not any(output_root == Path(root) or Path(root) in output_root.parents for root in roots):
        raise ValueError("训练输出根目录不在 Agent 允许目录内")
    if output_root not in run_dir.parents:
        raise ValueError("训练运行目录必须位于配置的输出根目录内")
    if Path(task["log_path"]).resolve() != run_dir / "platform.log":
        raise ValueError("日志路径必须是当前 Attempt 的 platform.log")
    if run_dir.exists():
        raise ValueError("训练 Attempt 目录已存在，禁止覆盖旧训练产物")
    run_dir.mkdir(parents=True, exist_ok=False)
    yolo_bin = str(Path(config["yolo_bin"]).resolve(strict=True)) if "/" in config["yolo_bin"] else config["yolo_bin"]
    if "/" in yolo_bin and not os.access(yolo_bin, os.X_OK):
        raise ValueError("配置的 yolo_bin 不存在或不可执行")
    command = [yolo_bin, task["task_type"], "train", f"model={model}", f"data={dataset}",
               f"epochs={int(params['epochs'])}", f"batch={int(params['batch'])}", f"imgsz={int(params['imgsz'])}",
               f"device={scalar_parameter(params['device'])}", f"workers={int(params['workers'])}",
               f"optimizer={scalar_parameter(params['optimizer'])}", f"project={run_dir.parent}", f"name={run_dir.name}", "exist_ok=True"]
    checkpoint = task.get("checkpoint_path")
    if checkpoint:
        checkpoint_path = within_allowed(checkpoint, roots)
        command.extend([f"model={checkpoint_path}", "resume=True"])
    extras = params.get("config", {})
    for key, value in extras.items():
        if key in TRAINING_KEYS and value is not None:
            command.append(f"{key}={scalar_parameter(value)}")
    for key, value in (extras.get("custom_args") or {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key) or key in RESERVED_KEYS or key in TRAINING_KEYS:
            raise ValueError(f"不允许的自定义训练参数：{key}")
        command.append(f"{key}={scalar_parameter(value)}")
    return command


def read_remote_progress(run_dir: Path) -> tuple[int, dict, list[dict]]:
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        return 0, {}, []
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))[-256:]
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


def run_remote_task(config: dict, task: dict, stop: threading.Event) -> None:
    run_id = task["run_id"]
    endpoint = f"/api/compute/nodes/{config['node_id']}/agent/runs/{run_id}"
    log_lines: list[str] = []
    process = None
    try:
        command = build_training_command(config, task)
        log_path = Path(task["log_path"])
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(f"\n[{datetime.now(timezone.utc).isoformat()}] REMOTE COMMAND {shlex.join(command)}\n")
            environment = os.environ.copy()
            environment["MKL_THREADING_LAYER"] = "GNU"
            process = subprocess.Popen(command, cwd=config["training_cwd"], env=environment, stdout=log_handle,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            response = api_json(config, endpoint + "/update", "POST", {"status": "running", "pid": process.pid,
                                "run_dir": task["run_dir"], "command": command})
            requested_stop = bool(response.get("stop_requested"))
            last_metrics: set[int] = set()
            while process.poll() is None and not stop.is_set():
                time.sleep(2)
                try:
                    control = api_json(config, endpoint + "/control")
                    if control.get("stop") and not requested_stop:
                        requested_stop = True
                        os.killpg(process.pid, signal.SIGTERM)
                except (OSError, ValueError, HTTPError, URLError):
                    pass
                completed, latest, metrics = read_remote_progress(Path(task["run_dir"]))
                new_metrics = [item for item in metrics if item["epoch"] not in last_metrics]
                last_metrics.update(item["epoch"] for item in new_metrics)
                log_lines = recent_log_lines(log_path)
                status = "stopping" if stop.is_set() or requested_stop else "running"
                try:
                    response = api_json(config, endpoint + "/update", "POST", {"status": status, "pid": process.pid,
                             "completed_epochs": completed, "progress": min(99.99, completed / max(1, int(task['parameters']['epochs'])) * 100),
                             "latest_metrics": latest, "metrics": new_metrics, "log_tail": log_lines,
                             "run_dir": task["run_dir"]})
                    if response.get("stop_requested") and not requested_stop:
                        requested_stop = True
                        os.killpg(process.pid, signal.SIGTERM)
                except (OSError, ValueError, HTTPError, URLError):
                    pass
            if stop.is_set() and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_code = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
        completed, latest, metrics = read_remote_progress(Path(task["run_dir"]))
        log_lines = recent_log_lines(Path(task["log_path"]))
        final_status = "stopped" if stop.is_set() or requested_stop else "completed" if exit_code == 0 else "failed"
        error = None if final_status != "failed" else f"训练进程异常退出，退出码 {exit_code}"
        final_payload = {"status": final_status, "completed_epochs": completed,
                         "progress": 100 if final_status == "completed" else min(99.99, completed / max(1, int(task['parameters']['epochs'])) * 100),
                         "latest_metrics": latest, "metrics": metrics, "log_tail": log_lines, "exit_code": exit_code,
                         "error_message": error, "run_dir": task["run_dir"]}
        while True:
            try:
                api_json(config, endpoint + "/update", "POST", final_payload)
                break
            except HTTPError as exc:
                if exc.code in {401, 403, 404, 422}:
                    LOGGER.error("最终训练状态无法上报，HTTP %s；本地退出码 %s", exc.code, exit_code)
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
            except OSError:
                pass
        try:
            api_json(config, endpoint + "/update", "POST", {"status": "failed", "error_message": str(exc)[:4000], "exit_code": 1})
        except Exception:
            LOGGER.exception("远程训练错误状态上报失败")
        LOGGER.error("远程训练任务失败：%s", exc)


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
    training_stop = threading.Event()
    LOGGER.info("监控 Agent 已启动，节点 %s，Agent %s；远程训练 %s", config["node_id"], VERSION,
                "已启用" if config["enable_training"] else "未启用")
    while not stop.is_set():
        try:
            payload = collect(config["nvidia_smi_bin"])
            interval = send(config, payload)
            if failures or args.once:
                LOGGER.info("采样上报成功：GPU %s，进程 %s", len(payload["compute"]["gpus"]),
                            sum(len(gpu["processes"]) for gpu in payload["compute"]["gpus"]))
            failures = 0
            delay = interval
            if config["enable_training"] and not args.once and (active_training is None or not active_training.is_alive()):
                try:
                    response = api_json(config, f"/api/compute/nodes/{config['node_id']}/agent/poll", "POST", {})
                    task = response.get("task")
                    if task:
                        training_stop = threading.Event()
                        active_training = threading.Thread(target=run_remote_task, args=(config, task, training_stop),
                                                          name="remote-training", daemon=True)
                        active_training.start()
                        LOGGER.info("已领取远程任务 %s / Attempt %s", task.get("task_id"), task.get("attempt_number"))
                except (HTTPError, URLError, OSError, ValueError, TypeError):
                    LOGGER.warning("远程任务轮询失败，将在下次心跳重试")
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
    LOGGER.info("Agent 已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
