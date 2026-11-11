"""Drive the real linked NebWidget, without assigning InputDetails.neb_state."""

import copy

import pytest
from aiida import orm
from aiida.common import LinkType
from ase.build import molecule

from surfaces_tools.widgets import inputs

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def fixture_nodes(axes="XYZ", invalid="geo"):
    atoms = molecule("NH3")
    atoms.cell = [12, 12, 12]
    atoms.center()
    atoms.pbc = True
    source = orm.StructureData(ase=atoms).store()
    solver = "PERIODIC" if axes == "XYZ" else "ANALYTIC" if len(axes) == 1 else "MT"
    dft = orm.Dict({"periodic": axes, "poisson_solver": solver}).store()

    def process(neb, outputs):
        node = orm.WorkChainNode(process_type="test.neb" if neb else "test.geo")
        node.base.links.add_incoming(
            source, link_type=LinkType.INPUT_WORK, link_label="structure"
        )
        node.base.links.add_incoming(
            dft, link_type=LinkType.INPUT_WORK, link_label="dft_params"
        )
        if neb:
            node.base.links.add_incoming(
                orm.Dict({"number_of_replica": 5}).store(),
                link_type=LinkType.INPUT_WORK,
                link_label="neb_params",
            )
        node.store()
        if outputs:
            for i in range(5):
                replica = orm.StructureData(ase=atoms).store()
                replica.base.links.add_incoming(
                    node, link_type=LinkType.RETURN, link_label=f"opt_replica_{i:03d}"
                )
        return node

    valid = process(True, True)
    bad = process(invalid == "missing_outputs", False)
    return source, valid, bad


def linked_widgets(module, source):
    details = module.InputDetails()
    details.neb = True
    details.structure = source.get_ase()
    neb = next(
        section
        for section in details.displayed_sections
        if isinstance(section, module.NebWidget)
    )
    return details, neb


@pytest.mark.parametrize("axes", ["NONE", "Y", "XY", "XYZ"])
@pytest.mark.parametrize("invalid", ["geo", "missing_outputs"])
def test_actual_linked_input_invalid_valid_and_clear_restore_banner(
    aiida_profile_clean, axes, invalid
):
    source, valid, bad = fixture_nodes(axes, invalid)
    frozen = {
        node.uuid: copy.deepcopy(node.base.attributes.all)
        for node in [source, valid, bad]
    }
    details, neb = linked_widgets(inputs, source)
    neb.restart_from.value = str(valid.pk)
    assert details.neb_state["restart_from"] == str(valid.pk)
    assert (
        f"periodic axes <b>{axes}</b>" in details.electrostatics.current_settings.value
    )
    assert neb._restart_replica_count == 5
    neb.restart_from.value = str(bad.pk)
    assert details.neb_state["restart_from"] == str(bad.pk)
    assert "color:red" in neb.restart_info.value.replace(" ", "")
    banner = details.electrostatics.current_settings.value
    assert "Effective calculation model unavailable" in banner
    assert "Currently used for submission" not in banner
    with pytest.raises(ValueError):
        details.electrostatics.return_dict()
    neb.restart_from.value = str(valid.pk)
    banner = details.electrostatics.current_settings.value
    assert (
        "Currently used for submission" in banner
        and f"periodic axes <b>{axes}</b>" in banner
    )
    assert details.electrostatics.return_dict()["dft_params"]["periodic"] == axes
    neb.restart_from.value = ""
    assert details.neb_state["restart_from"] == ""
    assert (
        "Currently used for submission" in details.electrostatics.current_settings.value
    )
    assert "3D periodic model" in details.electrostatics.current_settings.value
    assert "periodic axes <b>XYZ</b>" in details.electrostatics.current_settings.value
    assert (
        "Effective calculation model unavailable"
        not in details.electrostatics.current_settings.value
    )
    assert not details.electrostatics.active.value
    for identifier, attrs in frozen.items():
        assert orm.load_node(identifier).base.attributes.all == attrs
