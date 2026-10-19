from __future__ import annotations

import os
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from sqlalchemy.engine import make_url

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = BACKEND_ROOT.parent
SOURCE_PROJECT_ROOT = WEB_ROOT.parent
PLATFORM_INSTANCE_NAME = os.getenv("PLATFORM_INSTANCE_NAME", SOURCE_PROJECT_ROOT.name).strip()[:80] or SOURCE_PROJECT_ROOT.name


def read_platform_version() -> str:
    """Read the checkout version without depending on the process working directory."""
    configured = os.getenv("PLATFORM_VERSION", "").strip()
    if configured:
        return configured[:40]
    try:
        value = (SOURCE_PROJECT_ROOT / "VERSION").read_text(encoding="ascii").strip()
    except OSError:
        try:
            return package_version("yolo-training-platform")
        except PackageNotFoundError:
            return "0.0.0"
    return value[:40] or "0.0.0"


PLATFORM_VERSION = read_platform_version()


def nonnegative_int_env(name: str, default: int) -> int:
    """Read a byte-count setting and fail early on an unsafe deployment value."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是非负整数（单位：字节）") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} 必须是非负整数（单位：字节）")
    return parsed


def project_path(value: str | Path, *, base: Path = SOURCE_PROJECT_ROOT) -> Path:
    """Resolve configured relative paths against this checkout, not the shell CWD."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def resolve_yolo_bin(configured: str | None = None) -> Path:
    """Locate the YOLO CLI without assuming a server-specific environment path."""
    value = configured if configured is not None else os.getenv("YOLO_BIN", "")
    if value.strip():
        candidate = Path(value.strip()).expanduser()
        if candidate.is_absolute() or candidate.parent != Path("."):
            return (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
        located = shutil.which(value.strip())
        return Path(located).resolve() if located else (PROJECT_ROOT / candidate).resolve()
    environment_cli = Path(sys.executable).resolve().with_name("yolo")
    if environment_cli.is_file():
        return environment_cli
    located = shutil.which("yolo")
    return Path(located).resolve() if located else environment_cli


PROJECT_ROOT = project_path(os.getenv("YOLO_PROJECT_ROOT", "."))
ULTRALYTICS_ROOT = project_path(os.getenv("ULTRALYTICS_ROOT", "ultralytics"), base=PROJECT_ROOT)
YOLO_BIN = resolve_yolo_bin()
NVIDIA_SMI_BIN = os.getenv("NVIDIA_SMI_BIN", "nvidia-smi")
TRAINING_MKL_THREADING_LAYER = os.getenv("PLATFORM_MKL_THREADING_LAYER", "GNU")
DATA_ROOT = project_path(os.getenv("PLATFORM_DATA_ROOT", "web/data"), base=PROJECT_ROOT)
RUNS_ROOT = project_path(os.getenv("PLATFORM_RUNS_ROOT", str(DATA_ROOT / "runs")), base=PROJECT_ROOT)
RUNTIME_ROOT = project_path(os.getenv("PLATFORM_RUNTIME_ROOT", str(DATA_ROOT / "task_runtime")), base=PROJECT_ROOT)
LOG_ROOT = project_path(os.getenv("PLATFORM_LOG_ROOT", str(DATA_ROOT / "log")), base=PROJECT_ROOT)

# Persistent model-test files are user-owned data. Keep a reserve large enough
# for normalized originals, annotated outputs, and their ZIP export. A value of
# zero disables a quota or minimum-free-space check intentionally.
MODEL_TEST_HISTORY_QUOTA_BYTES = nonnegative_int_env(
    "PLATFORM_MODEL_TEST_HISTORY_QUOTA_BYTES", 20 * 1024**3,
)
MODEL_TEST_TASK_QUOTA_BYTES = nonnegative_int_env(
    "PLATFORM_MODEL_TEST_TASK_QUOTA_BYTES", 5 * 1024**3,
)
MODEL_TEST_MIN_FREE_BYTES = nonnegative_int_env(
    "PLATFORM_MODEL_TEST_MIN_FREE_BYTES", 2 * 1024**3,
)
MODEL_TEST_IMAGE_RESERVE_BYTES = nonnegative_int_env(
    "PLATFORM_MODEL_TEST_IMAGE_RESERVE_BYTES", 64 * 1024**2,
)


def resolve_database_url(value: str) -> str:
    """Anchor relative SQLite URLs to this project so dev and prod stay isolated."""
    url = make_url(value)
    if url.get_backend_name() == "sqlite" and url.database not in (None, ":memory:"):
        database = Path(url.database)
        if not database.is_absolute():
            url = url.set(database=str(project_path(database, base=PROJECT_ROOT)))
    return url.render_as_string(hide_password=False)


DATABASE_URL = resolve_database_url(os.getenv("DATABASE_URL", "sqlite:///web/data/platform.db"))

MODEL_ROOTS = [PROJECT_ROOT / "models", PROJECT_ROOT / "pt", ULTRALYTICS_ROOT / "weights"]
DATASET_ROOTS = [PROJECT_ROOT / "数据集", ULTRALYTICS_ROOT / "xl"]

for directory in (DATA_ROOT, RUNS_ROOT, RUNTIME_ROOT, LOG_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
