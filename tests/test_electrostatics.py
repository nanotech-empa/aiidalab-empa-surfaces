"""Opt-in solver controls must preserve and restore the submitted geometry."""

import numpy as np
import pytest
import traitlets as tr
from aiida import orm
from ase import Atoms

from surfaces_tools.widgets.electrostatics import PoissonSettingsWidget
from surfaces_tools.widgets.inputs import InputDetails

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.mark.parametrize(
    "name",
    [
        "submit_geometry_optimization.ipynb",
        "submit_scf.ipynb",
        "submit_phonons.ipynb",
        "submit_neb.ipynb",
        "submit_replica_chain.ipynb",
        "submit_pdos.ipynb",
        "submit_spm.ipynb",
        "submit_adsorption_energy.ipynb",
    ],
)
def test_notebook_links_to_an_existing_structure_manager(name):
    import ast
    import json
    from pathlib import Path

    from IPython.core.inputtransformer2 import TransformerManager

    notebook = json.loads((Path(__file__).resolve().parents[1] / name).read_text())
    selector_created = linked = False
    transformer = TransformerManager()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = transformer.transform_cell("".join(cell["source"]))
        for statement in ast.parse(source).body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "structure_selector":
                    selector_created = True
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "structure_manager"
                ):
                    assert selector_created, (
                        "Structure selector must exist before linking its manager"
                    )
                    linked = True
    assert linked


def molecule():
    return Atoms(
        "H2",
        positions=[[2.0, 3.0, 4.0], [2.0, 3.0, 4.74]],
        cell=[8.0, 9.0, 10.0],
        pbc=True,
    )


def test_default_is_off_and_apply_restore_is_exact():
    published = []
    widget = PoissonSettingsWidget(published.append)
    original = molecule()
    widget.set_structure(original, {"system_type": "Molecule"})
    assert not widget.active.value
    assert not published
    assert widget.return_dict() == {"dft_params": {"periodic": "XYZ"}}
    assert "15 Å" in widget.preview.value
    widget.extra.value = '{"MT": {"ALPHA": 9.0}}'
    widget.active.value = True
    assert widget.return_dict()["dft_params"]["periodic"] == "NONE"
    assert widget.return_dict()["dft_params"]["poisson_settings"] == {
        "MT": {"ALPHA": 9.0}
    }
    assert published[-1].cell[2, 2] > original.cell[2, 2]
    widget.active.value = False
    np.testing.assert_array_equal(published[-1].cell, original.cell)
    np.testing.assert_array_equal(published[-1].positions, original.positions)
    np.testing.assert_array_equal(published[-1].pbc, original.pbc)


def test_wire_guidance_and_explicit_solver_settings():
    published = []
    widget = PoissonSettingsWidget(published.append)
    widget.set_structure(molecule(), {"system_type": "Wire", "periodic_axes": "Z"})
    assert widget.solver.value == "ANALYTIC"
    assert "not implemented for 1D" in widget.guidance.value
    widget.solver.value = "MT"
    assert widget.active.disabled
    assert not published
    widget.solver.value = "ANALYTIC"
    widget.active.value = True
    d = widget.return_dict()["dft_params"]
    assert d == dict(periodic="Z", poisson_solver="ANALYTIC", poisson_settings={})
    np.testing.assert_array_equal(published[-1].cell[2], molecule().cell[2])


def test_invalid_advanced_settings_block_submission_and_can_be_deactivated():
    widget = PoissonSettingsWidget()
    widget.set_structure(molecule(), {"system_type": "Molecule"})
    widget.active.value = True
    widget.extra.value = '{"PERIODIC": "X"}'
    assert not widget.active.disabled
    with pytest.raises(ValueError, match="valid Poisson"):
        widget.return_dict()
    widget.active.value = False
    assert widget.return_dict()["dft_params"] == {"periodic": "XYZ"}


def test_external_structure_change_resets_activation_without_restoring_old_structure():
    published = []
    widget = PoissonSettingsWidget(published.append)
    widget.set_structure(molecule(), {"system_type": "Molecule"})
    widget.active.value = True
    different = molecule()
    different.cell = [80.0, 90.0, 100.0]
    widget.set_structure(different, {"system_type": "Molecule"})
    assert not widget.active.value
    np.testing.assert_array_equal(widget.original.cell, different.cell)
    assert widget.active.button_style == ""
    assert len(published) == 1


def test_unknown_wire_orientation_requires_explicit_axes():
    widget = PoissonSettingsWidget()
    widget.set_structure(molecule(), {"system_type": "Wire", "periodic_axes": None})
    assert widget.periodic.value is None
    assert widget.active.disabled
    widget.periodic.value = "X"
    assert widget.solver.value == "ANALYTIC"
    assert not widget.active.disabled


def test_input_details_applies_to_manager_without_resetting_spin_inputs():
    class Manager(tr.HasTraits):
        structure = tr.Instance(Atoms, allow_none=True)

    original = molecule()
    manager = Manager(structure=original.copy())
    details = InputDetails()
    details.structure_manager = manager
    link = tr.dlink((manager, "structure"), (details, "structure"))
    details.net_charge = 2
    details.uks = True
    details.electrostatics.active.value = True
    assert details.net_charge == 2
    assert details.uks
    assert manager.structure.cell[2, 2] > original.cell[2, 2]
    details.electrostatics.active.value = False
    np.testing.assert_array_equal(manager.structure.cell, original.cell)
    np.testing.assert_array_equal(manager.structure.positions, original.positions)
    assert details.uks
    link.unlink()


def test_separate_structure_details_update_without_repeating_analysis(monkeypatch):
    from surfaces_tools.widgets import analyze_structure
    from surfaces_tools.widgets.inputs import StructureInfoWidget

    calls = []
    analyze = analyze_structure.StructureAnalyzer.analyze

    def counted_analysis(analyzer):
        calls.append(analyzer.structure)
        return analyze(analyzer)

    monkeypatch.setattr(
        analyze_structure.StructureAnalyzer, "analyze", counted_analysis
    )
    details = InputDetails(structure_info_separate=True)
    panel = details.structure_info
    assert panel.layout.display == "none"
    calls.clear()
    details.structure = molecule()
    assert len(calls) == 1
    assert panel.details["numatoms"] == 2
    assert panel.layout.display == ""
    assert not any(
        isinstance(section, StructureInfoWidget)
        for section in details.displayed_sections
    )
    details.structure = Atoms(
        "H2O", positions=[[2, 2, 2], [2, 2.8, 2], [2.8, 2, 2]], cell=[12, 12, 12]
    )
    assert len(calls) == 2
    assert details.structure_info is panel
    assert panel.details["numatoms"] == 3
    assert panel.details["all_elements"] == details.details["all_elements"]
    details.structure = None
    assert panel.details == {}
    assert panel.layout.display == "none"


def test_neb_preparation_uses_shared_cell_translation_and_provenance(aiida_profile):
    geometries = [molecule(), molecule()]
    geometries[1].positions[:, 0] += 3.0
    nodes = [orm.StructureData(ase=g).store() for g in geometries]
    details = InputDetails()
    details.electrostatics.get_geometries = lambda: [g.copy() for g in geometries]
    details.electrostatics.set_structure(geometries[0], {"system_type": "Molecule"})
    details.electrostatics.active.value = True
    prepared = details.prepare_neb_structures(
        dict(initial_uuid=nodes[0].uuid, replica_uuids=[nodes[1].uuid])
    )
    np.testing.assert_allclose(prepared[0].cell, prepared[1].cell)
    shifts = [
        p.get_ase().positions - g.positions
        for p, g in zip(prepared, geometries, strict=True)
    ]
    np.testing.assert_allclose(shifts[0], shifts[1])
    for source, result in zip(nodes, prepared, strict=True):
        assert result.creator.inputs.structure.uuid == source.uuid
        np.testing.assert_array_equal(source.cell, molecule().cell)


def test_cell_changes_reach_the_real_structure_node(aiida_profile):
    import aiidalab_widgets_base as awb

    manager = awb.StructureManagerWidget(
        importers=[], editors=[], node_class="StructureData"
    )
    original = molecule()
    manager.input_structure = original.copy()
    details = InputDetails()
    details.structure_manager = manager
    link = tr.dlink((manager, "structure"), (details, "structure"))
    details.electrostatics.active.value = True
    np.testing.assert_array_equal(manager.structure_node.cell, details.structure.cell)
    assert manager.structure_node.cell[2][2] > original.cell[2, 2]
    details.electrostatics.active.value = False
    np.testing.assert_array_equal(manager.structure_node.cell, original.cell)
    np.testing.assert_array_equal(
        manager.structure_node.get_ase().positions, original.positions
    )
    link.unlink()
    manager.close()
