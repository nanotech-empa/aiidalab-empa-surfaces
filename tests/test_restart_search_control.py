"""The explicit restart path takes precedence over the optional WFN search."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import ipywidgets as ipw
import pytest

NOTEBOOKS = [
    "submit_geometry_optimization.ipynb",
    "submit_scf.ipynb",
    "submit_pdos.ipynb",
    "submit_spm.ipynb",
    "submit_phonons.ipynb",
    "submit_adsorption_energy.ipynb",
]


@pytest.mark.parametrize("notebook", NOTEBOOKS)
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("manual", [False, True])
def test_restart_choice(notebook, skip, manual):
    path = Path(__file__).resolve().parents[1] / notebook
    cells = json.loads(path.read_text())["cells"]
    source = next(
        "".join(c["source"])
        for c in cells
        if "parent_calc_folder_path =" in "".join(c["source"])
    )
    nodes = list(ast.walk(ast.parse(source)))
    assignment = next(
        node
        for node in nodes
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "parent_calc_folder_path"
            for target in node.targets
        )
    )
    branch = next(
        node
        for node in nodes
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "parent_calc_folder_path"
    )
    tree = ast.Module(body=[assignment, branch], type_ignores=[])
    found_wfn, explicit_folder = SimpleNamespace(pk=1), object()
    search = Mock(return_value=found_wfn)
    remote = Mock(return_value=explicit_folder)
    code = SimpleNamespace(computer=SimpleNamespace(hostname="localhost"))
    builder = SimpleNamespace(cp2k_code=code, code=code)
    namespace = {
        "builder": builder,
        "parent_calc_folder": ipw.Text(value=" /scratch/restart " if manual else ""),
        "skip_wfn_restart_search": ipw.Checkbox(value=skip),
        "structure_selector": SimpleNamespace(
            structure_node=SimpleNamespace(is_stored=True)
        ),
        "dft_params_dict": {},
        "dft_params": {},
        "parameters": {"dft_params": {}},
        "code": code,
        "wfn": SimpleNamespace(structure_available_wfn=search),
        "orm": SimpleNamespace(RemoteData=remote),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    if manual:
        assert builder.parent_calc_folder is explicit_folder
        assert remote.call_args.kwargs["remote_path"] == "/scratch/restart"
        search.assert_not_called()
    elif skip:
        assert not hasattr(builder, "parent_calc_folder")
        search.assert_not_called()
        remote.assert_not_called()
    else:
        assert builder.parent_calc_folder is found_wfn
        search.assert_called_once()
