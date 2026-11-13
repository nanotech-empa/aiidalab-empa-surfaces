"""Actual GUI and workflow builders, intercepted before any process submission."""

import ast
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
import ipywidgets as ipw
from surfaces_tools.widgets import dft_settings
from ase.build import molecule
from aiida import orm, plugins
from aiida.common import AttributeDict
from surfaces_tools.widgets.inputs import InputDetails
from surfaces_tools.utils.atom_indices import string_range_to_list

pytest_plugins = ["aiida.tools.pytest_fixtures"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(aiida_profile_clean, aiida_localhost):
    assert aiida_profile_clean.is_test_profile
    atoms = molecule("H2")
    atoms.cell = [12, 12, 12]
    atoms.center()
    atoms.pbc = False
    structure = orm.StructureData(ase=atoms).store()
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
    ).store()
    return (structure, code)


def pdos_builder(version, structure, code, details, attempts=1):
    nb_path = ROOT / "submit_pdos.ipynb"
    nb = json.loads(nb_path.read_text())
    function = next(
        (
            n
            for c in nb["cells"]
            if "def get_builder():" in "".join(c["source"])
            for n in ast.parse("".join(c["source"])).body
            if isinstance(n, ast.FunctionDef) and n.name == "get_builder"
        )
    )
    resource = NS(nodes=1, tasks_per_node=4, threads_per_task=1, walltime_seconds=1200)
    cards = [
        NS(
            name=NS(value=name),
            charge=NS(value=0),
            indices=NS(value=indices),
            resources=resource,
        )
        for name, indices in [("all", "1..2"), ("molecule", "1..2")]
    ]
    env = {
        "orm": orm,
        "Cp2kPdosWorkChain": plugins.WorkflowFactory("nanotech_empa.cp2k.pdos"),
        "submit_out": contextlib.nullcontext(),
        "clear_output": lambda: None,
        "structure_selector": NS(
            structure=structure.get_ase(), structure_node=structure
        ),
        "cp2k_code": NS(value=code.uuid),
        "do_overlap": NS(value=False),
        "overlap_code": NS(value=None),
        "protocol": NS(value="standard"),
        "workflow_description": NS(value="synthetic-model-regression"),
        "elpa_check": NS(value=False),
        "uks_widget": NS(uks=False),
        "diagonalisation_smearing": NS(
            enable_diagonalisation=NS(value=True), smearing_enabled=False
        ),
        "fragment_list": NS(fragments=cards),
        "input_details": details,
        "dft_settings_widget": NS(update_dft_params=lambda values: None),
        "slab_emin": NS(value=-1.0),
        "slab_emax": NS(value=1.0),
        "molecule_n_homo": NS(value=1),
        "molecule_n_lumo": NS(value=1),
        "awb": NS(utils=NS(string_range_to_list=string_range_to_list)),
        "projections": NS(items=[NS(selection=NS(value="1..2"), label=NS(value="H2"))]),
        "parent_calc_folder": NS(value=""),
        "skip_wfn_restart_search": NS(value=True),
    }
    env.update(ipw=ipw, dft_settings=dft_settings)
    advanced_source = next(
        "".join(cell["source"])
        for cell in nb["cells"]
        if "max_cp2k_attempts = ipw.BoundedIntText" in "".join(cell["source"])
    )
    exec(
        compile(advanced_source, "<actual PDOS Advanced settings>", "exec"),
        env,
    )
    env["skip_wfn_restart_search"].value = True
    if "max_cp2k_attempts" in env:
        env["max_cp2k_attempts"].value = attempts
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "<actual PDOS builder>",
            "exec",
        ),
        env,
    )
    return env["get_builder"](), env


def linked_details(structure):
    details = InputDetails(structure_info_separate=True)
    details.structure = structure.get_ase()
    details.electrostatics.periodic.value = "XYZ"
    details.electrostatics.solver.value = "PERIODIC"
    details.electrostatics.active.value = True
    return details


def test_actual_control_is_visible_in_normal_advanced_display(source):
    structure, code = source
    builder, env = pdos_builder("candidate", structure, code, linked_details(structure))
    control = env["max_cp2k_attempts"]
    assert isinstance(control, ipw.BoundedIntText)
    assert control.value == control.min == 1
    assert control.description == "Max CP2K attempts per stage:"
    assert control in env["advanced_settings"].children[0].children
    assert env["advanced_settings"].get_title(0) == "Restart"
    notebook = json.loads((ROOT / "submit_pdos.ipynb").read_text())
    shown = []
    display_source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "display(workflow_description, advanced_settings" in "".join(cell["source"])
    )
    exec(
        compile(
            display_source,
            "<normal Advanced settings display>",
            "exec",
        ),
        {**env, "btn_submit": object(), "display": lambda *v: shown.extend(v)},
    )
    assert env["advanced_settings"] in shown
    control.value = 0
    assert control.value == 1
    assert env["get_builder"]().max_iterations.value == 1
    assert orm.QueryBuilder().append(orm.ProcessNode).count() == 0


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, orm.Dict):
        return value.get_dict()
    if isinstance(value, orm.List):
        return value.get_list()
    if isinstance(value, orm.BaseType):
        return value.value
    if isinstance(value, orm.Node):
        return value.uuid
    return value


@pytest.mark.parametrize("attempts", [1, 3])
def test_actual_builder_changes_only_attempts_and_forwards_to_both_native_stages(
    source, attempts
):
    structure, code = source
    details = linked_details(structure)
    builder, _ = pdos_builder("candidate", structure, code, details, attempts)
    assert builder.max_iterations.value == attempts
    assert not builder.do_overlap.value and not builder.dft_params.get_dict().get(
        "vdw", False
    )
    assert builder.dft_params["periodic"] == "XYZ"
    assert builder.dft_params["poisson_solver"] == "PERIODIC"
    assert "parent_calc_folder" not in builder._inputs(prune=True)
    inputs = AttributeDict(builder._inputs(prune=True))
    inputs.update(
        clean_workdir=orm.Bool(False),
        on_unhandled_failure=orm.Str("pause"),
        pause_on_max_iterations=orm.Bool(True),
    )
    pdos = plugins.WorkflowFactory("nanotech_empa.cp2k.pdos")
    diags = []
    h = NS(inputs=inputs, ctx=AttributeDict(), report=lambda *_: None)
    h.submit = lambda b: diags.append(b) or NS()
    h.to_context = lambda **kw: h.ctx.update(kw)
    pdos.setup(h)
    pdos.run_diags(h)
    assert len(diags) == 1  # overlap off: only full-system OT and diagonalization
    d = diags[0]
    assert d.max_iterations.value == attempts
    diag = plugins.WorkflowFactory("nanotech_empa.cp2k.diag")
    di = AttributeDict(d._inputs(prune=True))
    di.clean_workdir = orm.Bool(False)
    captured = []
    remote = orm.RemoteData(
        computer=code.computer, remote_path="/tmp/no-native-pdos-attempt-test"
    )
    g = NS(inputs=di, ctx=AttributeDict(), report=lambda *_: None)
    g.submit = lambda b: (
        captured.append(b) or NS(is_finished_ok=True, outputs=NS(remote_folder=remote))
    )
    g.to_context = lambda **kw: g.ctx.update(kw)
    g._handle_periodicity = lambda atoms: diag._handle_periodicity(g, atoms)
    g.update_ot_input_dict = lambda p: diag.update_ot_input_dict(g, p)
    g.update_diag_input_dict = lambda p: diag.update_diag_input_dict(g, p)
    diag.setup(g)
    diag.run_ot_scf(g)
    diag.run_diag_scf(g)
    assert len(captured) == 2
    for cp in captured:
        assert cp.max_iterations.value == attempts
        assert dict(cp.cp2k.metadata.options.resources) == {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 4,
            "num_cores_per_mpiproc": 1,
        }
        assert cp.cp2k.metadata.options.max_wallclock_seconds == 1200
        assert cp.cp2k.parameters["FORCE_EVAL"]["SUBSYS"]["CELL"]["PERIODIC"] == "XYZ"
        assert cp.cp2k.structure.pbc == (True, True, True)
    assert orm.QueryBuilder().append(orm.CalcJobNode).count() == 0
