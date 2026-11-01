"""Restart lookup through actual AiiDA provenance and local file staging."""

import ase
import pytest
from aiida import orm
from aiida.common.links import LinkType

from surfaces_tools.utils import wfn

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.mark.parametrize("input_label", ["code", "cp2k_code", "cp2k__code", None])
def test_cp2k_code_lookup_handles_supported_workchain_inputs(
    aiida_localhost, input_label
):
    code = orm.InstalledCode(
        computer=aiida_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    ).store()
    workchain = orm.WorkChainNode()
    if input_label:
        workchain.base.links.add_incoming(code, LinkType.INPUT_WORK, input_label)
    workchain.store()
    assert wfn.get_cp2k_code(workchain) == (code if input_label else None)


def _structure_from_workchain(computer, directory, label="CP2K_GeoOpt"):
    code = orm.InstalledCode(
        computer=computer,
        filepath_executable="/bin/true",
        default_calc_job_plugin="cp2k",
    ).store()
    workchain = orm.WorkChainNode(label=label)
    for name, value in [
        ("code", code),
        ("dft_params", orm.Dict({"uks": False})),
        ("neb_params", orm.Dict({"number_of_replica": 2})),
    ]:
        workchain.base.links.add_incoming(value.store(), LinkType.INPUT_WORK, name)
    workchain.store()
    inner = orm.WorkChainNode()
    inner.base.links.add_incoming(workchain, LinkType.CALL_WORK, "inner")
    inner.store()
    calculation = orm.CalcJobNode(computer=computer)
    calculation.base.links.add_incoming(inner, LinkType.CALL_CALC, "cp2k")
    calculation.store()
    structure = orm.StructureData(
        ase=ase.Atoms("H", positions=[[0, 0, 0]], cell=[10] * 3)
    )
    structure.label = "Replica 2"
    structure.base.links.add_incoming(calculation, LinkType.CREATE, "output_structure")
    structure.store()
    folder = orm.RemoteData(computer=computer, remote_path=str(directory))
    folder.base.links.add_incoming(calculation, LinkType.CREATE, "remote_folder")
    folder.store()
    folder.base.links.add_incoming(workchain, LinkType.RETURN, "remote_folder")
    return structure, workchain, folder


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "missing-file",
        "wrong-host",
        "spin-mismatch",
        "charge-mismatch",
        "unsupported-process",
    ],
)
def test_restart_lookup_checks_provenance_and_compatibility(
    aiida_localhost, tmp_path, case
):
    label = "other workflow" if case == "unsupported-process" else "CP2K_GeoOpt"
    structure, workchain, folder = _structure_from_workchain(
        aiida_localhost, tmp_path, label
    )
    assert wfn.find_first_workchain(structure) == workchain
    assert wfn.find_first_workchain(workchain) == workchain
    if case != "missing-file":
        (tmp_path / "aiida-RESTART.wfn").write_bytes(b"test wavefunction")
    params = {
        "uks": case == "spin-mismatch",
        "charge": 1 if case == "charge-mismatch" else 0,
    }
    hostname = "another-host" if case == "wrong-host" else aiida_localhost.hostname
    actual = wfn.structure_available_wfn(
        structure, current_hostname=hostname, return_path=False, dft_params=params
    )
    assert actual == (folder if case == "valid" else None)


def test_neb_restart_copy_uses_requested_destination(aiida_localhost, tmp_path):
    structure, _, folder = _structure_from_workchain(
        aiida_localhost, tmp_path, "CP2K_NEB"
    )
    source = tmp_path / "aiida-BAND2-RESTART.wfn"
    payload = b"selected NEB image wavefunction"
    source.write_bytes(payload)
    actual = wfn.structure_available_wfn(
        structure,
        current_hostname=aiida_localhost.hostname,
        return_path=False,
        dft_params={"uks": False},
    )
    assert actual == folder
    assert source.read_bytes() == payload
    assert (tmp_path / "aiida-RESTART.wfn").read_bytes() == payload


def test_unsupported_structure_without_provenance_returns_no_restart(aiida_profile):
    structure = orm.StructureData(
        ase=ase.Atoms("H", positions=[[0, 0, 0]], cell=[10] * 3)
    ).store()
    assert wfn.structure_available_wfn(structure, dft_params={}) is None
