"""Submission notebooks must work without the optional openBIS importer."""

import ast
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from IPython.core.inputtransformer2 import TransformerManager

NOTEBOOKS = (
    "submit_adsorption_energy.ipynb",
    "submit_casscf.ipynb",
    "submit_geometry_optimization.ipynb",
    "submit_gw.ipynb",
    "submit_gw_ic.ipynb",
    "submit_neb.ipynb",
    "submit_nics.ipynb",
    "submit_pdos.ipynb",
    "submit_phonons.ipynb",
    "submit_replica_chain.ipynb",
    "submit_scf.ipynb",
    "submit_spm.ipynb",
)


@pytest.mark.parametrize("notebook_name", NOTEBOOKS)
@pytest.mark.parametrize("availability", ("absent", "older", "available"))
def test_optional_openbis_importer(monkeypatch, notebook_name, availability):
    notebook = json.loads(
        (Path(__file__).parents[1] / notebook_name).read_text(encoding="utf-8")
    )
    transformer = TransformerManager()
    source = "\n".join(
        transformer.transform_cell("".join(cell["source"]))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(source)
    symbol = "OpenbisStructureImporterWidget"
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.ImportFrom) and child.module == "aiidalab_eln"
            for child in node.body
        )
    ]
    selections = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Starred)
        and isinstance(node.value, ast.IfExp)
        and any(
            isinstance(child, ast.Name) and child.id == symbol
            for child in ast.walk(node.value)
        )
    ]
    assert len(imports) == len(selections) == 1

    module = ModuleType("aiidalab_eln")
    if availability == "available":
        setattr(module, symbol, lambda **kwargs: kwargs["title"])
    monkeypatch.setitem(
        sys.modules, "aiidalab_eln", None if availability == "absent" else module
    )

    # Execute only the optional import and list entry, never workflow submission.
    namespace = {}
    import_code = compile(
        ast.Module(body=imports, type_ignores=[]), notebook_name, "exec"
    )
    exec(import_code, namespace)  # noqa: S102 - trusted, isolated notebook import block
    result = eval(
        compile(ast.Expression(selections[0]), notebook_name, "eval"), namespace
    )
    assert result == (["From openBIS"] if availability == "available" else [])
