from pathlib import Path

import pytest

from app import parameter_catalog


def test_catalog_reads_training_sections_and_cross_section_keys(monkeypatch, tmp_path: Path):
    default_cfg = tmp_path / "default.yaml"
    default_cfg.write_text(
        """# Train settings -----
epochs: 100
time:
amp: true
patience: 100
fraction: 1.0
# Predict settings -----
source:
classes:
augment: false
# Hyperparameters -----
lr0: 0.01
distill_model:
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(parameter_catalog, "DEFAULT_CFG_PATH", default_cfg)

    catalog = parameter_catalog.load_parameter_catalog()

    assert set(catalog) == {"time", "amp", "patience", "fraction", "classes", "lr0", "distill_model"}
    assert catalog["time"]["type"] == "number"
    assert catalog["time"]["nullable"] is True
    assert catalog["amp"]["type"] == "boolean"
    assert catalog["patience"]["integer"] is True
    assert catalog["patience"]["minimum"] == 0
    assert catalog["fraction"]["fraction_or_count"] is True
    assert catalog["classes"]["section"] == "Training-compatible"
    assert "source" not in catalog
    assert "augment" not in catalog


def test_catalog_returns_empty_when_default_config_is_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(parameter_catalog, "DEFAULT_CFG_PATH", tmp_path / "missing.yaml")

    assert parameter_catalog.load_parameter_catalog() == {}


def test_dynamic_parameter_validation_uses_catalog_types_and_ranges():
    catalog = {
        "amp": {"type": "boolean", "default": True, "nullable": False},
        "patience": {"type": "number", "default": 100, "nullable": False, "integer": True, "minimum": 0},
        "lr0": {"type": "number", "default": 0.01, "nullable": False, "integer": False,
                "minimum": 0.0, "maximum": 1.0},
        "fraction": {"type": "number", "default": 1.0, "nullable": False, "integer": False,
                     "minimum_exclusive": 0.0, "fraction_or_count": True},
        "time": {"type": "number", "default": None, "nullable": True, "integer": False},
        "split": {"type": "text", "default": "val", "nullable": False},
    }
    parameter_catalog.validate_parameter_values(
        {"amp": False, "patience": 0, "lr0": 0.5, "fraction": 200, "time": None, "split": "train"},
        catalog,
    )

    invalid = [
        {"amp": "false"},
        {"patience": 1.5},
        {"patience": -1},
        {"lr0": float("nan")},
        {"lr0": float("inf")},
        {"fraction": 1.5},
        {"fraction": 0},
        {"time": 10**400},
        {"split": "   "},
    ]
    for values in invalid:
        with pytest.raises(ValueError):
            parameter_catalog.validate_parameter_values(values, catalog)


def test_dynamic_parameter_validation_rejects_unknown_keys():
    with pytest.raises(ValueError, match="不支持"):
        parameter_catalog.validate_parameter_values(
            {"made_up": 1}, {"amp": {"type": "boolean", "default": True, "nullable": False}},
        )
