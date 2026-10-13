from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.engine import make_url

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = BACKEND_ROOT.parent
SOURCE_PROJECT_ROOT = WEB_ROOT.parent
PLATFORM_INSTANCE_NAME = os.getenv("PLATFORM_INSTANCE_NAME", SOURCE_PROJECT_ROOT.name).strip()[:80] or SOURCE_PROJECT_ROOT.name


def project_path(value: str | Path, *, base: Path = SOURCE_PROJECT_ROOT) -> Path:
    """Resolve configured relative paths against this checkout, not the shell CWD."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


PROJECT_ROOT = project_path(os.getenv("YOLO_PROJECT_ROOT", "."))
ULTRALYTICS_ROOT = project_path(os.getenv("ULTRALYTICS_ROOT", "ultralytics"), base=PROJECT_ROOT)
YOLO_BIN = Path(os.getenv("YOLO_BIN", "/data/conda/envs/yolo/bin/yolo")).resolve()
NVIDIA_SMI_BIN = os.getenv("NVIDIA_SMI_BIN", "nvidia-smi")
TRAINING_MKL_THREADING_LAYER = os.getenv("PLATFORM_MKL_THREADING_LAYER", "GNU")
DATA_ROOT = project_path(os.getenv("PLATFORM_DATA_ROOT", "web/data"), base=PROJECT_ROOT)
RUNS_ROOT = project_path(os.getenv("PLATFORM_RUNS_ROOT", str(DATA_ROOT / "runs")), base=PROJECT_ROOT)
RUNTIME_ROOT = project_path(os.getenv("PLATFORM_RUNTIME_ROOT", str(DATA_ROOT / "task_runtime")), base=PROJECT_ROOT)
LOG_ROOT = project_path(os.getenv("PLATFORM_LOG_ROOT", str(DATA_ROOT / "log")), base=PROJECT_ROOT)


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
