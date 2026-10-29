from pathlib import Path
from importlib.util import module_from_spec, spec_from_file_location

import pytest

RENDERER_PATH = Path(__file__).resolve().parents[2] / "deploy" / "render_systemd_unit.py"
RENDERER_SPEC = spec_from_file_location("render_systemd_unit", RENDERER_PATH)
RENDERER = module_from_spec(RENDERER_SPEC)
RENDERER_SPEC.loader.exec_module(RENDERER)
render_unit = RENDERER.render_unit


@pytest.mark.parametrize(
    ("instance", "port"),
    [("development", "18088"), ("production", "18089")],
)
def test_rendered_unit_uses_checkout_and_selected_instance(tmp_path: Path, instance, port):
    root = tmp_path / "项目 % checkout"
    (root / "web" / "backend" / "app").mkdir(parents=True)
    interpreter = tmp_path / "python environment" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")

    unit = render_unit(instance, root, interpreter)
    escaped_root = str(root.resolve()).replace("%", "%%").replace(" ", r"\x20")

    assert f"WorkingDirectory={escaped_root}/web/backend" in unit
    assert f"EnvironmentFile=-{escaped_root}/web/.env" in unit
    assert f'"--port" "{port}"' in unit
    assert "/data/yolo-" not in unit
    assert "/data/conda/" not in unit


def test_render_unit_rejects_a_checkout_without_backend(tmp_path: Path):
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="backend app directory"):
        render_unit("development", tmp_path, interpreter)
