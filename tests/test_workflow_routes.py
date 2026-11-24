"""Every assistant destination is an actual maintained notebook, not a default."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "assistant.json").read_text())
PAGES = MANIFEST["workflow_pages"]


@pytest.mark.parametrize("workflow", list(PAGES))
def test_each_declared_workflow_is_a_real_chat_notebook(workflow):
    route = PAGES[workflow]
    notebook = ROOT / route["notebook"]
    assert notebook.parent == ROOT and notebook.is_file()
    data = json.loads(notebook.read_text())
    code = "\n".join(
        "".join(c.get("source", [])) for c in data["cells"] if c["cell_type"] == "code"
    )
    assert "create_panel" in code


@pytest.mark.parametrize(
    "workflow,notebook",
    [
        ("stm", "submit_spm.ipynb"),
        ("pdos", "submit_pdos.ipynb"),
        ("neb", "submit_neb.ipynb"),
        ("geometry_optimization", "submit_geometry_optimization.ipynb"),
        ("stm_results", "view_stm.ipynb"),
        ("pdos_results", "view_pdos.ipynb"),
        ("neb_results", "view_neb.ipynb"),
    ],
)
def test_scientific_destination_is_explicit(workflow, notebook):
    assert PAGES[workflow]["notebook"] == notebook


def test_no_integrated_canonical_notebook_is_left_out():
    actual = set()
    # Read the maintained route owners, not numbered runtime notebook copies.
    from surfaces_tools.assistant_bindings import FORM_ROOTS

    actual.update(FORM_ROOTS)
    declared = {route["notebook"] for route in PAGES.values()}
    assert actual <= declared
    assert PAGES["xps"]["notebook"] == "xps_guide.ipynb"
    assert "Work in progress" in PAGES["xps"]["title"]
