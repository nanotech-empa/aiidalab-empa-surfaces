import numpy as np
from ase import Atoms

from surfaces_tools.widgets.analyze_structure import StructureAnalyzer, get_types


def test_single_layer_slab_does_not_require_layer_spacing():
    atoms = Atoms(
        "Au4",
        positions=[
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        cell=[3.0, 3.0, 10.0],
        pbc=[True, True, True],
    )

    atom_types, layers = get_types(atoms)

    assert np.all(atom_types == 1)
    assert len(layers) == 1


def test_structure_analysis_leaves_original_cell_and_pbc_unchanged():
    atoms = Atoms(
        "H2",
        positions=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.74]],
        cell=[8.0, 9.0, 10.0],
        pbc=False,
    )
    original = atoms.copy()
    analyzer = StructureAnalyzer()
    analyzer.structure = atoms
    assert analyzer.details["system_type"] == "Molecule"
    np.testing.assert_array_equal(atoms.cell, original.cell)
    np.testing.assert_array_equal(atoms.pbc, original.pbc)
