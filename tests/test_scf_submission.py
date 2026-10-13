"""Validate the SCF notebook builders against the current backend interface."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import ase
import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import pytest
from aiida import orm, plugins

from surfaces_tools.widgets import dft_settings, inputs

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
            ("unfolding_code", "nanotech_empa.cp2k_unfolding"),
        )
    }
    namespace = {
        "ipw": ipw,
        "np": np,
        "plt": plt,
        "inputs": inputs,
        "dft_settings": dft_settings,
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
        assert parameters["periodic"] == "XYZ"
        assert not any(
            key in parameters
            for key in (
                "sc_diag",
                "elpa_switch",
                "added_mos",
                "smear_t",
                "force_multiplicity",
            )
        )
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


@pytest.mark.parametrize("path", ["", "G-X-G"])
@pytest.mark.parametrize("sparse", [False, True])
def test_unfolding_builder(scf_notebook, path, sparse):
    ui = scf_notebook
    ui["compute_unfolding"].value = True
    ui["unfolding_a1"].value = "2 0 0"
    ui["unfolding_path"].value = path
    ui["retrieve_sparse_overlap"].value = sparse
    builder = ui["prepare_scf"]()
    assert (
        ui["Cp2kScfWorkChain"].spec().inputs.validate(builder._inputs(prune=True))
        is None
    )
    assert builder.run_diag_scf.value
    assert builder.overlap_matrix.value == (
        "remote_and_sparse_retrieved" if sparse else "remote_only"
    )
    assert builder.unfolding_primitive_vectors.value == "2 0 0"
    assert builder.unfolding_lattice_type.value == "auto"
    assert ("unfolding_path" in builder) == bool(path)
    if path:
        assert builder.unfolding_path.value == path
    assert builder.dft_params["added_mos"] > 0
    for obsolete in (
        "compute_unfolding",
        "compute_bader_charges",
        "write_overlap_matrix",
        "retrieve_sparse_overlap",
        "unfolding_options",
        "unfolding_primitive_basis_atoms",
        "unfolding_basis_cluster_tol",
    ):
        assert obsolete not in builder


def test_unfolding_bader_mutual_exclusion(scf_notebook):
    ui = scf_notebook
    ui["compute_unfolding"].value = True
    ui["compute_bader_charges"].value = True
    assert not ui["compute_unfolding"].value
    builder = ui["prepare_scf"]()
    assert "unfolding_code" not in builder
    assert not builder.run_diag_scf.value


@pytest.mark.parametrize("missing", ["code", "vector"])
def test_unfolding_missing_input(scf_notebook, missing):
    ui = scf_notebook
    ui["compute_unfolding"].value = True
    ui["unfolding_a1"].value = "2 0 0"
    if missing == "code":
        ui["unfolding_code"].value = ""
    else:
        ui["unfolding_a1"].value = ""
    assert ui["prepare_scf"]() is None


def test_preview_oblique_path_matches_tool_and_bragg_plane(scf_notebook):
    ui = scf_notebook
    primitive = np.array([[1, 0, 0], [-0.4, 1.3, 0]])
    points, path = ui["_standard_kpath"](2, "auto", primitive)
    assert path != ["G", "X", "S", "Y", "G"]
    assert ui["_parse_path_labels"]("G-H1-A1-G", path) == ["G", "H1", "A1", "G"]
    assert ui["_parse_path_labels"]("", path) == path
    reciprocal = 2 * np.pi * np.linalg.inv(primitive[:, :2].T).T
    for label, fractional in points.items():
        point = reciprocal @ fractional
        for i in range(-3, 4):
            for j in range(-3, 4):
                g = reciprocal @ [i, j]
                assert point @ g <= g @ g / 2 + 1e-10, label


def test_preview_closed_path_keeps_final_gamma(scf_notebook):
    ui = scf_notebook
    primitive = np.array([[1, 0, 0]])
    points, _ = ui["_standard_kpath"](1, "auto", primitive)
    indices, path_points = ui["_project_kpoints_to_path"](
        np.array([[0.0], [0.5]]), points, ["G", "X", "G"], primitive
    )
    assert indices.tolist() == [0, 1, 0]
    assert len(path_points) == 3


def test_selected_dft_settings_reach_builder(scf_notebook):
    ui = scf_notebook
    settings = ui["dft_settings_widget"]
    settings.xc_functional.value = "PBE0"
    settings.hfx_fraction.value = 0.3
    settings.basis_set_file_names.value = "BASIS_CUSTOM, BASIS_ADMM_UZH"
    settings.potential_file_name.value = "POTENTIAL_CUSTOM"
    settings.basis_set_overrides.value = '{"H": "custom-basis"}'
    builder = ui["prepare_scf"]()
    parameters = builder.dft_params.get_dict()
    assert parameters["xc_functional"] == "PBE0"
    assert parameters["hfx_fraction"] == 0.3
    assert parameters["basis_set_file_names"] == ["BASIS_CUSTOM", "BASIS_ADMM_UZH"]
    assert parameters["potential_file_name"] == "POTENTIAL_CUSTOM"
    assert parameters["basis_set_overrides"] == {"H": "custom-basis"}
    assert (
        ui["Cp2kScfWorkChain"].spec().inputs.validate(builder._inputs(prune=True))
        is None
    )


def test_invalid_basis_override_blocks_submission(scf_notebook):
    scf_notebook["dft_settings_widget"].basis_set_overrides.value = "[1]"
    assert scf_notebook["prepare_scf"]() is None
