from __future__ import annotations

import os
import re
import shutil
import sys
import uuid
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
    """Read a non-negative integer setting and fail early on unsafe values."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} 必须是非负整数") from exc
    if parsed < 0:
        raise RuntimeError(f"{name} 必须是非负整数")
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
# Immutable checkpoints fetched from remote model-test roots live outside the
# history tree, so they need their own admission limit.  The minimum-free check
# reserves the bytes still outstanding for every active transfer instead of
# looking only at the first chunk already written to disk.
REMOTE_MODEL_CACHE_QUOTA_BYTES = nonnegative_int_env(
    "PLATFORM_REMOTE_MODEL_CACHE_QUOTA_BYTES", 20 * 1024**3,
)
REMOTE_MODEL_CACHE_MIN_FREE_BYTES = nonnegative_int_env(
    "PLATFORM_REMOTE_MODEL_CACHE_MIN_FREE_BYTES", 2 * 1024**3,
)


def project_path_list_env(name: str, default: str, *, base: Path = SOURCE_PROJECT_ROOT) -> list[Path]:
    """Resolve an ``os.pathsep`` separated list without depending on the process CWD."""
    configured = os.getenv(name, default)
    values = [value.strip() for value in configured.split(os.pathsep) if value.strip()]
    if not values:
        raise RuntimeError(f"{name} 至少需要配置一个目录")
    paths: list[Path] = []
    for value in values:
        path = project_path(value, base=base)
        if path not in paths:
            paths.append(path)
    return paths


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
# Standalone model tests intentionally have a narrower trust boundary than the
# training model catalogue. Deployments may add roots with the platform path
# separator (":" on Linux); relative entries stay portable across checkouts.
MODEL_TEST_ROOTS = project_path_list_env("PLATFORM_MODEL_TEST_ROOTS", "pt", base=PROJECT_ROOT)

for directory in (DATA_ROOT, RUNS_ROOT, RUNTIME_ROOT, LOG_ROOT):
    directory.mkdir(parents=True, exist_ok=True)


def platform_instance_id() -> str:
    """Return a stable installation ID stored beneath the configured data root."""
    configured = os.getenv("PLATFORM_INSTANCE_ID", "").strip().lower()
    if configured:
        if not re.fullmatch(r"[0-9a-f]{32}", configured):
            raise RuntimeError("PLATFORM_INSTANCE_ID 必须是 32 位十六进制 ID")
        return configured

    identity_file = DATA_ROOT / ".platform-instance-id"
    try:
        value = identity_file.read_text(encoding="ascii").strip().lower()
        if re.fullmatch(r"[0-9a-f]{32}", value):
            return value
        raise RuntimeError("平台实例 ID 文件格式无效")
    except FileNotFoundError:
        value = uuid.uuid4().hex
        temporary = DATA_ROOT / f".platform-instance-id.{value}.tmp"
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(value + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, identity_file)
                return value
            except FileExistsError:
                return platform_instance_id()
        finally:
            temporary.unlink(missing_ok=True)

# Grace period before escalating a local training stop to SIGKILL.
TRAINING_STOP_TIMEOUT_SECONDS = max(1, min(300, int(os.getenv("TRAINING_STOP_TIMEOUT_SECONDS", "15"))))

# 数据集目录白名单；相对路径相对于项目根目录，界面不能越过此范围。
DATASET_STORAGE_ALLOWED_ROOTS = project_path_list_env(
    "PLATFORM_DATASET_STORAGE_ALLOWED_ROOTS", str(PROJECT_ROOT), base=PROJECT_ROOT,
)
DATASET_STORAGE_QUOTA_BYTES = nonnegative_int_env("PLATFORM_DATASET_STORAGE_QUOTA_BYTES", 100 * 1024**3)
# Read-only reloads have their own worker lane. These limits apply while a
# local training run is active; 0 disables the corresponding limit. The PSI
# check is optional on hosts without Linux I/O pressure accounting.
DATASET_REFRESH_READ_BYTES_PER_SECOND = nonnegative_int_env(
    "PLATFORM_DATASET_REFRESH_READ_BYTES_PER_SECOND", 32 * 1024**2)
DATASET_REFRESH_IO_PRESSURE_PCT = nonnegative_int_env(
    "PLATFORM_DATASET_REFRESH_IO_PRESSURE_PCT", 25)
DATASET_REFRESH_MIN_FREE_BYTES = nonnegative_int_env(
    "PLATFORM_DATASET_REFRESH_MIN_FREE_BYTES", 512 * 1024**2)
# Zero disables reclamation for the corresponding snapshot class.
DATASET_STORAGE_RETENTION_SECONDS = nonnegative_int_env("PLATFORM_DATASET_STORAGE_RETENTION_SECONDS", 24 * 3600)
DATASET_FAILED_RETENTION_SECONDS = nonnegative_int_env("PLATFORM_DATASET_FAILED_RETENTION_SECONDS", 7 * 24 * 3600)
