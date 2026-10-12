from pathlib import Path

from app import parameter_catalog


def test_catalog_reads_training_sections_and_cross_section_keys(monkeypatch, tmp_path: Path):
    default_cfg = tmp_path / "default.yaml"
    default_cfg.write_text(
        """# Train settings -----
epochs: 100
time:
amp: true
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

    assert set(catalog) == {"time", "amp", "classes", "lr0", "distill_model"}
    assert catalog["time"]["type"] == "number"
    assert catalog["amp"]["type"] == "boolean"
    assert catalog["classes"]["section"] == "Training-compatible"
    assert "source" not in catalog
    assert "augment" not in catalog


def test_catalog_returns_empty_when_default_config_is_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(parameter_catalog, "DEFAULT_CFG_PATH", tmp_path / "missing.yaml")

    assert parameter_catalog.load_parameter_catalog() == {}
