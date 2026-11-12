"""Launcher recovery must retain current workflows and keep XPS informational."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_launcher_keeps_workflows_and_labels_xps_work_in_progress():
    spec = importlib.util.spec_from_file_location("surfaces_start", ROOT / "start.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    html = module.get_start_widget("/apps/surfaces", "/").value
    assert "CP2K XPS · Work in progress" in html
    assert "/apps/surfaces/xps_guide.ipynb" in html
    for _, items in module.ITEMS:
        for _, _, _, filename in items:
            assert (ROOT / filename).is_file(), filename
            assert f"/apps/surfaces/{filename}" in html
    assert "submit_scf.ipynb" in html
    assert "submit_neb.ipynb" in html
    assert "submit_benchmark.ipynb" in html


def test_xps_wip_page_has_no_profile_or_submission_path(monkeypatch):
    from aiida import engine

    def unexpected_submit(*args, **kwargs):
        raise AssertionError("The XPS WIP page must not submit scientific work")

    monkeypatch.setattr(engine, "submit", unexpected_submit)
    notebook = json.loads((ROOT / "xps_guide.ipynb").read_text())
    source = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "load_profile" not in source
    assert "aiida_nanotech_empa" not in source
    exec(compile(source, "xps_guide.ipynb", "exec"), {})
