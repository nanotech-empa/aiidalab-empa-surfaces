"""Validate the SCF notebook builders against the current backend interface."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import ase
import ipywidgets as ipw
import pytest
from aiida import orm, plugins

from surfaces_tools.widgets import inputs

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def scf_notebook(aiida_profile, aiida_localhost):
    codes = {
        name: orm.InstalledCode(
            computer=aiida_localhost,
            filepath_executable="/bin/true",
            default_calc_job_plugin=entry_point,
        ).store()
        for name, entry_point in (
            ("code_input_widget", "cp2k"),
            ("bader_code", "nanotech_empa.bader"),
            ("sparse_overlap_code", "nanotech_empa.sparse_overlap"),
        )
    }
    namespace = {
        "ipw": ipw,
        "inputs": inputs,
        "orm": orm,
        "Cp2kScfWorkChain": plugins.WorkflowFactory("nanotech_empa.cp2k.scf"),
        "structure_selector": SimpleNamespace(
            structure_node=orm.StructureData(
                ase=ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[8] * 3)
            )
        ),
        "input_details": SimpleNamespace(
            return_final_dictionary=lambda: (True, "", {"dft_params": {}})
        ),
        "resources": SimpleNamespace(
            walltime_seconds=600, nodes=1, tasks_per_node=1, threads_per_task=1
        ),
        "workflow_description": SimpleNamespace(value="SCF input regression"),
        "output": ipw.Output(),
        "clear_output": lambda: None,
        **{name: ipw.Text(value=code.uuid) for name, code in codes.items()},
    }
    notebook_path = Path(__file__).resolve().parents[1] / "submit_scf.ipynb"
    notebook = json.loads(notebook_path.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith("# Protocol."):
            exec(compile(source, str(notebook_path), "exec"), namespace)
        elif "def prepare_scf():" in source:
            # Execute the actual submission function without launching the app.
            tree = ast.parse(source)
            tree.body = [
                item for item in tree.body if isinstance(item, ast.FunctionDef)
            ]
            exec(compile(tree, str(notebook_path), "exec"), namespace)
    yield namespace
    # Release the widgets created by this notebook cell, including hidden controls.
    for value in namespace.values():
        if isinstance(value, ipw.Widget):
            value.close()


@pytest.mark.parametrize("mode", ["plain", "overlap", "sparse", "bader"])
@pytest.mark.parametrize("self_consistent", [False, True])
@pytest.mark.parametrize("smearing", [False, True])
def test_scf_builder(scf_notebook, mode, self_consistent, smearing):
    ui = scf_notebook
    ui["write_overlap_matrix"].value = mode in ("overlap", "sparse")
    ui["retrieve_sparse_overlap"].value = mode == "sparse"
    ui["compute_bader_charges"].value = mode == "bader"
    diagonalisation = ui["diagonalisation_smearing"]
    diagonalisation.enable_diagonalisation.value = self_consistent
    diagonalisation.enable_smearing.value = smearing

    builder = ui["prepare_scf"]()
    workflow = ui["Cp2kScfWorkChain"]
    assert workflow.spec().inputs.validate(builder._inputs(prune=True)) is None
    assert builder.run_diag_scf.value == (mode in ("overlap", "sparse"))
    assert (
        builder.overlap_matrix.value
        == {
            "plain": "none",
            "overlap": "remote_only",
            "sparse": "remote_and_sparse_retrieved",
            "bader": "none",
        }[mode]
    )
    assert ("bader_code" in builder) == (mode == "bader")
    assert ("sparse_overlap_code" in builder) == (mode == "sparse")
    parameters = builder.dft_params.get_dict()
    if mode in ("overlap", "sparse"):
        assert parameters["added_mos"] == 100
        assert parameters["elpa_switch"] is True
        assert parameters["sc_diag"] == self_consistent
        assert ("smear_t" in parameters) == (self_consistent and smearing)
        if self_consistent and smearing:
            assert parameters["smear_t"] == 150
            assert parameters["force_multiplicity"] is True
    else:
        assert parameters == {"periodic": "XYZ"}
    if mode == "bader":
        assert builder.bader_cutoff.value == 1200


def test_hidden_overlap_settings_are_ignored(scf_notebook):
    ui = scf_notebook
    ui["write_overlap_matrix"].value = True
    ui["retrieve_sparse_overlap"].value = True
    ui["write_overlap_matrix"].value = False
    ui["sparse_overlap_code"].value = ""
    builder = ui["prepare_scf"]()
    assert builder.overlap_matrix.value == "none"
    assert not builder.run_diag_scf.value
    assert "sparse_overlap_code" not in builder


def test_bader_disables_overlap(scf_notebook):
    ui = scf_notebook
    ui["write_overlap_matrix"].value = True
    ui["retrieve_sparse_overlap"].value = True
    ui["compute_bader_charges"].value = True
    assert not ui["write_overlap_matrix"].value
    assert ui["write_overlap_matrix"].disabled
    assert not ui["retrieve_sparse_overlap"].value
    builder = ui["prepare_scf"]()
    assert builder.overlap_matrix.value == "none"
    assert not builder.run_diag_scf.value
    assert "bader_code" in builder


@pytest.mark.parametrize("mode", ["bader", "sparse"])
def test_missing_postprocessing_code(scf_notebook, mode):
    ui = scf_notebook
    if mode == "bader":
        ui["compute_bader_charges"].value = True
        ui["bader_code"].value = ""
    else:
        ui["write_overlap_matrix"].value = True
        ui["retrieve_sparse_overlap"].value = True
        ui["sparse_overlap_code"].value = ""
    assert ui["prepare_scf"]() is None
