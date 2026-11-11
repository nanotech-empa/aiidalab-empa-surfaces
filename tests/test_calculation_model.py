"""Model-aware views use transient copies of synthetic legacy structures."""

import copy
import io

import numpy as np
import pytest
from aiida import orm
from aiida.common import LinkType
from ase import Atoms
from ase.io import read, write

from surfaces_tools.utils.calculation_model import structure_for_display
from surfaces_tools.widgets.analyze_structure import StructureAnalyzer
from surfaces_tools.widgets.electrostatics import PoissonSettingsWidget

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.mark.parametrize("axes", ["NONE", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"])
def test_legacy_export_uses_calculation_model_without_rewriting(
    aiida_profile_clean, aiida_localhost, axes
):
    atoms = Atoms(
        "H2", positions=[[5, 6, 6], [5.75, 6, 6]], cell=[12, 12, 12], pbc=False
    )
    node = orm.CalcJobNode(
        computer=aiida_localhost, process_type="aiida.calculations:cp2k"
    )
    solver = "PERIODIC" if axes == "XYZ" else "ANALYTIC" if len(axes) == 1 else "MT"
    params = orm.Dict(
        {
            "FORCE_EVAL": {
                "SUBSYS": {"CELL": {"PERIODIC": axes}},
                "DFT": {"POISSON": {"PERIODIC": axes, "POISSON_SOLVER": solver}},
            }
        }
    ).store()
    node.base.links.add_incoming(
        params, link_type=LinkType.INPUT_CALC, link_label="parameters"
    )
    node.store()
    structure = orm.StructureData(ase=atoms)
    structure.base.links.add_incoming(
        node, link_type=LinkType.CREATE, link_label="output_structure"
    )
    structure.store()
    before = copy.deepcopy(structure.base.attributes.all)
    displayed, notice = structure_for_display(structure)
    expected = tuple(axis in axes for axis in "XYZ")
    assert tuple(displayed.pbc) == expected
    exported = io.StringIO()
    write(exported, displayed, format="extxyz")
    restored = read(io.StringIO(exported.getvalue()), format="extxyz")
    assert tuple(restored.pbc) == expected
    np.testing.assert_allclose(restored.positions, atoms.positions)
    assert "Effective calculation model" in notice
    assert structure.base.attributes.all == before


def test_oblique_cell_summary_preserves_internal_cell():
    atoms = Atoms(
        "BN",
        positions=[[0, 0, 7.5], [1, 1, 7.5]],
        cell=[[10, 0, 0], [-5, 8.660254037844, 0], [0, 0, 15]],
        pbc=True,
    )
    analyzer = StructureAnalyzer()
    analyzer.structure = atoms
    assert "Cell lengths (Å): 10 10 15" in analyzer.details["summary"]
    np.testing.assert_allclose(
        np.fromstring(analyzer.details["cell"], sep=" ").reshape(3, 3), atoms.cell
    )
    assert "Cell angles (°): 90 90 120" in analyzer.details["summary"]


def test_inactive_suggestions_preserve_effective_xyz_and_explicit_apply():
    atoms = Atoms(
        "H2", positions=[[5, 6, 6], [5.75, 6, 6]], cell=[12, 12, 12], pbc=True
    )
    widget = PoissonSettingsWidget()
    widget.set_structure(atoms, {"system_type": "Molecule"})
    for axes in ["NONE", "Y", "XY", "XYZ"]:
        widget.periodic.value = axes
        widget.solver.value = (
            "PERIODIC" if axes == "XYZ" else "ANALYTIC" if len(axes) == 1 else "MT"
        )
        assert "3D periodic model" in widget.current_settings.value
        assert widget.return_dict()["dft_params"] == {"periodic": "XYZ"}
    widget.active.value = True
    assert widget.return_dict()["dft_params"]["periodic"] == "XYZ"
    assert "Undo" in widget.preview.value
