from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = BACKEND_ROOT.parent
PROJECT_ROOT = Path(os.getenv("YOLO_PROJECT_ROOT", str(WEB_ROOT.parent))).resolve()
ULTRALYTICS_ROOT = Path(os.getenv("ULTRALYTICS_ROOT", str(PROJECT_ROOT / "ultralytics"))).resolve()
YOLO_BIN = Path(os.getenv("YOLO_BIN", "/data/conda/envs/yolo/bin/yolo")).resolve()
DATA_ROOT = Path(os.getenv("PLATFORM_DATA_ROOT", str(WEB_ROOT / "data"))).resolve()
RUNS_ROOT = Path(os.getenv("PLATFORM_RUNS_ROOT", str(DATA_ROOT / "runs"))).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_ROOT / 'platform.db'}")

MODEL_ROOTS = [PROJECT_ROOT / "models", PROJECT_ROOT / "pt", ULTRALYTICS_ROOT / "weights"]
DATASET_ROOTS = [PROJECT_ROOT / "数据集", ULTRALYTICS_ROOT / "xl"]

for directory in (DATA_ROOT, RUNS_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

