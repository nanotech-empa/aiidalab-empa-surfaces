"""Exercise the optional importer gate in every actual submission notebook."""

import ast
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize("notebook", NOTEBOOKS)
@pytest.mark.parametrize("available", [False, True])
def test_optional_importer_gate(monkeypatch, notebook, available):
    importer = object()
    created = []

    def create_importer(*, title):
        assert title == "From openBIS"
        created.append(title)
        return importer

    module = SimpleNamespace(OpenbisStructureImporterWidget=create_importer)
    real_import = builtins.__import__

    def optional_import(name, *args, **kwargs):
        if name != "aiidalab_eln":
            return real_import(name, *args, **kwargs)
        if not available:
            raise ImportError("optional package not installed")
        return module

    monkeypatch.setattr(builtins, "__import__", optional_import)
    path = Path(__file__).resolve().parents[1] / notebook
    namespace = {}
    gates = []
    importer_lists = []
    for cell in json.loads(path.read_text())["cells"]:
        source = "".join(cell["source"])
        if "aiidalab_eln" not in source and "From openBIS" not in source:
            continue
        source = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("%")
        )
        tree = ast.parse(source)
        gates.extend(
            n
            for n in tree.body
            if isinstance(n, ast.Try)
            and any(
                isinstance(child, ast.Import)
                and any(alias.name == "aiidalab_eln" for alias in child.names)
                for child in n.body
            )
        )
        importer_lists.extend(
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Starred)
            and "From openBIS" in ast.get_source_segment(source, n)
        )
    assert len(gates) == len(importer_lists) == 1
    exec(compile(ast.Module(body=gates, type_ignores=[]), str(path), "exec"), namespace)
    selected = eval(
        compile(ast.Expression(body=importer_lists[0]), str(path), "eval"), namespace
    )
    assert selected == ([importer] if available else [])
    assert len(created) == int(available)
