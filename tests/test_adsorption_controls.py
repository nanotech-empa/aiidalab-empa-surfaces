"""Current notebook controls and real builders with no process submission."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS
import ipywidgets as ipw
import pytest
from aiida import orm
from ase.build import molecule
from aiida_nanotech_empa.workflows.cp2k.fragment_separation import (
    Cp2kFragmentSeparationWorkChain,
)
from surfaces_tools.utils.atom_indices import string_range_to_list

pytest_plugins = ["aiida.tools.pytest_fixtures"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source_inputs(aiida_profile_clean, aiida_localhost):
    assert aiida_profile_clean.is_test_profile
    atoms = molecule("H2O")
    atoms.translate([2, 6, 6])
    other = molecule("H2O")
    other.translate([8, 6, 6])
    atoms += other
    atoms.cell = [12] * 3
    atoms.pbc = True
    source = orm.StructureData(ase=atoms).store()
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="cp2k",
    ).store()
    cube = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/false",
        default_calc_job_plugin="nanotech_empa.cubehandler",
    ).store()
    return {"code": code, "cubehandler_code": cube, "structure": source}


def test_real_gui_controls_and_builder_forwarding(source_inputs):
    nb = json.loads((ROOT / "submit_adsorption_energy.ipynb").read_text())
    env = {
        "ipw": ipw,
        "orm": orm,
        "Cp2kFragmentSeparationWorkChain": Cp2kFragmentSeparationWorkChain,
        "string_range_to_list": string_range_to_list,
    }
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if "advanced_settings = ipw.Accordion(" in text:
            exec(text, env)
        if "def get_builder():" in text:
            funcs = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
            exec(
                compile(
                    ast.Module(body=funcs, type_ignores=[]),
                    "<actual adsorption get_builder>",
                    "exec",
                ),
                env,
            )
    assert (
        env["serial_execution"].value is False and env["max_cp2k_attempts"].value == 5
    )
    env.update(
        computational_resources=NS(value=source_inputs["code"].uuid),
        cubehandler_code=NS(value=source_inputs["cubehandler_code"].uuid),
        structure_selector=NS(structure_node=source_inputs["structure"]),
        fixed_atoms=NS(value="1..3"),
        protocol=NS(value="standard"),
        uks_widget=NS(uks=False),
        input_details=NS(
            electrostatics=NS(return_dict=lambda: {"dft_params": {"periodic": "XYZ"}})
        ),
        fragment_list=NS(
            fragments=[
                NS(
                    name=NS(value=name),
                    indices=NS(value=indices),
                    charge=NS(value=0),
                    resources=NS(
                        walltime_seconds=600,
                        nodes=1,
                        tasks_per_node=2,
                        threads_per_task=1,
                    ),
                )
                for name, indices in [
                    ("all", "1..6"),
                    ("water_a", "1..3"),
                    ("water_b", "4..6"),
                ]
            ]
        ),
    )
    env["skip_wfn_restart_search"].value = True
    normal = env["get_builder"]()
    assert normal.serial_execution.value is False and normal.max_iterations.value == 5
    env["serial_execution"].value = True
    env["max_cp2k_attempts"].value = 1
    serial = env["get_builder"]()
    assert serial.serial_execution.value is True and serial.max_iterations.value == 1
    assert serial.dft_params.get_dict() == normal.dft_params.get_dict()
    assert (
        serial.options == normal.options
        and serial.structure.uuid == normal.structure.uuid
    )
