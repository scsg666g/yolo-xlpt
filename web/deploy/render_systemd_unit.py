#!/usr/bin/env python3
"""Render a systemd unit for a checkout without installing or restarting it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
INSTANCES = {
    "development": {"port": 18088, "label": "development"},
    "production": {"port": 18089, "label": "production"},
}


def _unit_path(path: Path) -> str:
    value = str(path)
    return value.replace("%", "%%").replace("\\", "\\\\").replace(" ", "\\x20").replace("\t", "\\t")


def _exec_arg(value: str | Path) -> str:
    escaped = str(value).replace("%", "%%").replace("$", "$$")
    escaped = escaped.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_unit(instance: str, project_root: Path, python: Path) -> str:
    if instance not in INSTANCES:
        raise ValueError(f"unsupported instance: {instance}")
    root = project_root.expanduser().resolve()
    interpreter = python.expanduser().resolve()
    app_dir = root / "web" / "backend" / "app"
    env_file = root / "web" / ".env"
    if not app_dir.is_dir():
        raise ValueError(f"backend app directory not found under checkout: {root}")
    if not interpreter.is_file():
        raise ValueError(f"Python interpreter not found: {interpreter}")

    instance_config = INSTANCES[instance]
    port = instance_config["port"]
    label = instance_config["label"]
    lines = [
        "[Unit]",
        f"Description=YOLO Training Platform ({label}, port {port})",
        "Wants=network-online.target",
        "After=network-online.target",
        "StartLimitIntervalSec=60",
        "StartLimitBurst=5",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={_unit_path(root / 'web' / 'backend')}",
        f"EnvironmentFile=-{_unit_path(env_file)}",
        "Environment=PYTHONUNBUFFERED=1",
        f"Environment=PLATFORM_INSTANCE_NAME={label}",
        f"ExecStartPre=/usr/bin/test -d {_exec_arg(app_dir)}",
        f"ExecStartPre=/usr/bin/test -r {_exec_arg(env_file)}",
        "ExecStart="
        + " ".join(
            _exec_arg(arg)
            for arg in (
                interpreter,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
                "--workers",
                "1",
            )
        ),
        "Restart=on-failure",
        "RestartSec=3",
        "TimeoutStopSec=45",
        "KillSignal=SIGTERM",
        "UMask=0077",
        "NoNewPrivileges=true",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=INSTANCES, default="development")
    parser.add_argument("--project-root", type=Path, default=CHECKOUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--output",
        type=Path,
        help="write the rendered unit here; stdout is used when omitted",
    )
    args = parser.parse_args()
    try:
        unit = render_unit(args.instance, args.project_root, args.python)
    except ValueError as error:
        parser.error(str(error))

    if args.output:
        args.output.expanduser().write_text(unit, encoding="utf-8")
    else:
        sys.stdout.write(unit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
