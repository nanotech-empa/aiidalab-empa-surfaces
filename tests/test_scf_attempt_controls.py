"""Exercise the actual prepare_scf and AiiDA builders without running a process."""

import ast
import contextlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import ase
import ipywidgets as ipw
import pytest
from aiida import orm, plugins

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def selected_code(filename):
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "submit_scf.ipynb").read_text()
    )
    nodes = []
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] != "code" or text.startswith("%%"):
            continue
        text = "\n".join(line for line in text.splitlines() if not line.startswith("%"))
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_scf":
                nodes.append(node)
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name)
                and t.id in {"max_cp2k_attempts", "attempt_help"}
                for t in node.targets
            ):
                nodes.append(node)
    return compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec")


@pytest.mark.parametrize("functional", ["PBE", "PBE0"])
def test_attempt_limit_reaches_real_builder_without_changing_science(
    aiida_profile_clean, aiida_localhost, functional
):
    from aiida_nanotech_empa.workflows.cp2k import cp2k_utils

    assert aiida_profile_clean.is_test_profile
    workchain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
        label="never-executed-widget-test",
    ).store()
    atoms = ase.Atoms(
        "CH3",
        positions=[[6, 6, 6], [7, 6, 6], [5.5, 6.86, 6], [5.5, 5.14, 6]],
        cell=[12, 12, 12],
        pbc=False,
    )
    structure = orm.StructureData(ase=atoms).store()
    parameters = {
        "dft_params": {
            "xc_functional": functional,
            "periodic": "XYZ",
            "uks": True,
            "multiplicity": 2,
            "charge": 0,
        }
    }

    def unexpected_restart_search(**kwargs):
        raise AssertionError("Fresh fixture must not search existing WFN files")

    namespace = {
        "ipw": ipw,
        "orm": orm,
        "Cp2kScfWorkChain": workchain,
        "output": contextlib.nullcontext(),
        "clear_output": lambda: None,
        "structure_selector": NS(structure_node=structure),
        "code_input_widget": NS(value=code.uuid),
        "input_details": NS(
            return_final_dictionary=lambda: (True, "", copy.deepcopy(parameters))
        ),
        "dft_settings_widget": NS(update_dft_params=lambda params: None),
        "diagonalisation_smearing": NS(
            enable_diagonalisation=NS(value=False), smearing_enabled=False
        ),
        "resources": NS(
            walltime_seconds=600, nodes=1, tasks_per_node=2, threads_per_task=1
        ),
        "protocol": NS(value="standard"),
        "workflow_description": NS(value="synthetic-attempt-test"),
        "parent_calc_folder": NS(value=""),
        "skip_wfn_restart_search": NS(value=True),
        "overlap_ndigits": NS(value=14),
        "overlap_threshold": NS(value=1e-10),
        "bader_cutoff": NS(value=1200),
        "added_mos": NS(value=100),
        "wfn": NS(structure_available_wfn=unexpected_restart_search),
    }
    for flag in [
        "compute_bader_charges",
        "compute_unfolding",
        "write_overlap_matrix",
        "retrieve_sparse_overlap",
    ]:
        namespace[flag] = NS(value=False)
    exec(selected_code("submit_scf.ipynb"), namespace)
    original = namespace["prepare_scf"]()._inputs(prune=True)
    original.pop("max_iterations")
    exec(selected_code("submit_scf.ipynb"), namespace)
    control = namespace["max_cp2k_attempts"]
    assert control.value == 5
    assert control.min == 1
    control.value = 1
    builder = namespace["prepare_scf"]()
    actual = builder._inputs(prune=True)
    assert actual.pop("max_iterations").value == 1

    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, orm.Dict):
            return value.get_dict()
        if isinstance(value, orm.BaseType):
            return value.value
        if isinstance(value, orm.Node):
            return value.uuid
        return value

    assert plain(actual) == plain(original)
    assert builder.dft_params.get_dict()["periodic"] == "XYZ"
    assert "added_mos" not in builder.dft_params.get_dict()
    assert "parent_calc_folder" not in actual
    assert builder.options.get_dict()["resources"] == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 2,
        "num_cores_per_mpiproc": 1,
    }

    inputs = NS(max_iterations=builder.max_iterations)
    for key in ("clean_workdir", "on_unhandled_failure", "pause_on_max_iterations"):
        setattr(inputs, key, workchain.spec().inputs[key].default())
    base = plugins.WorkflowFactory("cp2k.base").get_builder()
    cp2k_utils.set_restart_policy(inputs, base)
    assert base.max_iterations.value == 1
    assert base.clean_workdir.value is False
    assert base.on_unhandled_failure.value == "pause"
    assert base.pause_on_max_iterations.value is True
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0


@pytest.mark.parametrize("functional", ["PBE"])
@pytest.mark.parametrize("bad_walltime", ["not-a-time", "00:00:00"])
def test_invalid_walltime_is_rejected_before_submit(
    aiida_profile_clean, aiida_localhost, functional, monkeypatch, bad_walltime
):
    from aiida_nanotech_empa.workflows.cp2k import cp2k_utils

    assert aiida_profile_clean.is_test_profile
    workchain = plugins.WorkflowFactory("nanotech_empa.cp2k.scf")
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
        label="never-executed-widget-test",
    ).store()
    atoms = ase.Atoms(
        "CH3",
        positions=[[6, 6, 6], [7, 6, 6], [5.5, 6.86, 6], [5.5, 5.14, 6]],
        cell=[12, 12, 12],
        pbc=False,
    )
    structure = orm.StructureData(ase=atoms).store()
    parameters = {
        "dft_params": {
            "xc_functional": functional,
            "periodic": "XYZ",
            "uks": True,
            "multiplicity": 2,
            "charge": 0,
        }
    }

    def unexpected_restart_search(**kwargs):
        raise AssertionError("Fresh fixture must not search existing WFN files")

    namespace = {
        "ipw": ipw,
        "orm": orm,
        "Cp2kScfWorkChain": workchain,
        "output": contextlib.nullcontext(),
        "clear_output": lambda: None,
        "structure_selector": NS(structure_node=structure),
        "code_input_widget": NS(value=code.uuid),
        "input_details": NS(
            return_final_dictionary=lambda: (True, "", copy.deepcopy(parameters))
        ),
        "dft_settings_widget": NS(update_dft_params=lambda params: None),
        "diagonalisation_smearing": NS(
            enable_diagonalisation=NS(value=False), smearing_enabled=False
        ),
        "resources": NS(
            walltime_seconds=600, nodes=1, tasks_per_node=2, threads_per_task=1
        ),
        "protocol": NS(value="standard"),
        "workflow_description": NS(value="synthetic-attempt-test"),
        "parent_calc_folder": NS(value=""),
        "skip_wfn_restart_search": NS(value=True),
        "overlap_ndigits": NS(value=14),
        "overlap_threshold": NS(value=1e-10),
        "bader_cutoff": NS(value=1200),
        "added_mos": NS(value=100),
        "wfn": NS(structure_available_wfn=unexpected_restart_search),
    }
    for flag in [
        "compute_bader_charges",
        "compute_unfolding",
        "write_overlap_matrix",
        "retrieve_sparse_overlap",
    ]:
        namespace[flag] = NS(value=False)
    exec(selected_code("submit_scf.ipynb"), namespace)
    original = namespace["prepare_scf"]()._inputs(prune=True)
    original.pop("max_iterations")
    exec(selected_code("submit_scf.ipynb"), namespace)
    control = namespace["max_cp2k_attempts"]
    assert control.value == 5
    assert control.min == 1
    control.value = 1
    builder = namespace["prepare_scf"]()
    actual = builder._inputs(prune=True)
    assert actual.pop("max_iterations").value == 1

    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, orm.Dict):
            return value.get_dict()
        if isinstance(value, orm.BaseType):
            return value.value
        if isinstance(value, orm.Node):
            return value.uuid
        return value

    assert plain(actual) == plain(original)
    assert builder.dft_params.get_dict()["periodic"] == "XYZ"
    assert "added_mos" not in builder.dft_params.get_dict()
    assert "parent_calc_folder" not in actual
    assert builder.options.get_dict()["resources"] == {
        "num_machines": 1,
        "num_mpiprocs_per_machine": 2,
        "num_cores_per_mpiproc": 1,
    }

    inputs = NS(max_iterations=builder.max_iterations)
    for key in ("clean_workdir", "on_unhandled_failure", "pause_on_max_iterations"):
        setattr(inputs, key, workchain.spec().inputs[key].default())
    base = plugins.WorkflowFactory("cp2k.base").get_builder()
    cp2k_utils.set_restart_policy(inputs, base)
    assert base.max_iterations.value == 1
    assert base.clean_workdir.value is False
    assert base.on_unhandled_failure.value == "pause"
    assert base.pause_on_max_iterations.value is True
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0

    # Audit the real resource parser and Submit callback. No submission is allowed.
    import aiidalab_widgets_base as awb
    from aiidalab_widgets_empa.computational_resources import ProcessResourcesWidget

    resources = ProcessResourcesWidget()
    resources.nodes_widget.value = 1
    resources.tasks_per_node_widget.value = 2
    resources.threads_per_task_widget.value = 1
    resources.walltime_widget.value = bad_walltime
    assert resources.walltime_seconds in (None, 0)
    if resources.walltime_seconds is None:
        assert resources.wrong_syntax.layout.visibility == "visible"
    namespace["resources"] = resources
    invalid_builder = namespace["prepare_scf"]()
    assert invalid_builder is None
    calls = []

    class SubmissionIntercepted(Exception):
        pass

    def intercept_submit(builder):
        calls.append(builder.options.get_dict())
        raise SubmissionIntercepted("No process or scientific job may be submitted")

    monkeypatch.setattr(awb.process.engine, "submit", intercept_submit)
    button = awb.SubmitButtonWidget(
        workchain,
        namespace["prepare_scf"],
        disable_after_submit=False,
        append_output=True,
    )
    button.on_btn_submit_press()
    assert calls == []
    assert "did not recieve" in button.submit_out.value
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0
    resources.walltime_widget.value = "00:10:00"
    assert resources.walltime_seconds == 600
    assert namespace["prepare_scf"]().options.get_dict()["max_wallclock_seconds"] == 600
