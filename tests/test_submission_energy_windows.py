"""Exercise actual notebook energy controls and submission guards."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import ipywidgets as ipw
import pytest


@pytest.mark.parametrize("notebook", ["submit_pdos.ipynb", "submit_spm.ipynb"])
def test_energy_window_controls_and_invalid_submission(notebook, capsys):
    path = Path(__file__).resolve().parents[1] / notebook
    cells = json.loads(path.read_text())["cells"]
    controls = (
        {"slab_emin", "slab_emax"} if "pdos" in notebook else {"elim_float_slider"}
    )
    namespace = {"ipw": ipw, "style": {}, "layout": {}, "layout_small": {}}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not any(f"{name} =" in source for name in controls):
            continue
        tree = ast.parse(source)
        tree.body = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in controls for t in node.targets)
        ]
        exec(compile(tree, str(path), "exec"), namespace)

    source = next(
        "".join(c["source"])
        for c in cells
        if "def get_builder():" in "".join(c["source"])
    )
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace.update(
        submit_out=ipw.Output(),
        clear_output=lambda: None,
        structure_selector=SimpleNamespace(structure=object()),
        cp2k_code=SimpleNamespace(value="selected-code"),
        do_overlap=SimpleNamespace(value=False),
        drop_spm_type=SimpleNamespace(value="STM"),
    )
    exec(compile(tree, str(path), "exec"), namespace)

    if "pdos" in notebook:
        low, high = namespace["slab_emin"], namespace["slab_emax"]
        assert (low.value, high.value) == (-2.0, 2.0)
        low.value, high.value = -10, 10
        assert (low.value, high.value) == (-10, 10)
        invalid = [(3, 2), (2, 2)]
    else:
        slider = namespace["elim_float_slider"]
        assert slider.value == (-2.0, 2.0)
        slider.value = (-10, 10)
        assert slider.value == (-10, 10)
        invalid = [(2, 2)]

    for lower, upper in invalid:
        if "pdos" in notebook:
            low.value, high.value = lower, upper
        else:
            slider.value = (lower, upper)
        assert namespace["get_builder"]() is None
        assert "must be lower" in capsys.readouterr().out
