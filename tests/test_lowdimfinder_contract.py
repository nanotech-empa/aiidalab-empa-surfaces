import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from surfaces_tools.widgets.lowdimfinder import LowDimFinder


def test_group_data_exposes_unit_cell_ids():
    finder = LowDimFinder.__new__(LowDimFinder)
    finder.get_reduced_aiida_structures = lambda: []
    finder._dimensionality = [2]
    finder._chemical_formula = ["C"]
    finder._positions = [[[0.0, 0.0, 0.0]]]
    finder._chemical_symbols = [["C"]]
    finder.n_unit = 2
    finder._unit_cell_groups = [[0]]
    finder._cell = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    finder._tags = [[0]]

    assert finder.get_group_data()["unit_cell_ids"] == [[0]]


def test_group_data_maps_supercell_ids_to_unit_cell():
    finder = LowDimFinder.__new__(LowDimFinder)
    finder.get_reduced_aiida_structures = lambda: []
    finder._dimensionality = [2]
    finder._chemical_formula = ["C"]
    finder._positions = [[[0.0, 0.0, 0.0]]]
    finder._chemical_symbols = [["C"]]
    finder.n_unit = 2
    finder._unit_cell_groups = [[3]]
    finder._cell = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    finder._tags = [[0]]

    assert finder.get_group_data()["unit_cell_ids"] == [[1]]


@pytest.mark.parametrize("dimension", [0, 1, 2, 3])
def test_reduced_groups_reuse_symbols_and_keep_input_atom_identity(
    monkeypatch, dimension
):
    if dimension == 0:
        atoms = Atoms(
            "CO", positions=[[2, 2, 2], [2, 2, 3.13]], cell=[15, 15, 15], pbc=True
        )
        expected = [0]
    elif dimension == 1:
        atoms = Atoms(
            "Au2", positions=[[5, 5, 0], [5, 5, 2.9]], cell=[15, 15, 5.8], pbc=True
        )
        expected = [1]
    elif dimension == 2:
        atoms = Atoms(
            "Au4CO",
            positions=[
                [0, 0, 0],
                [2.9, 0, 0],
                [0, 2.9, 0],
                [2.9, 2.9, 0],
                [1.2, 1.2, 7],
                [1.2, 1.2, 8.13],
            ],
            cell=[5.8, 5.8, 15],
            pbc=True,
        )
        expected = [2, 0]
    else:
        atoms = bulk("Au", "fcc", a=4.08, cubic=True)
        expected = [3]
    atoms.set_tags(np.arange(len(atoms)))
    original = atoms.copy()
    finder = LowDimFinder(
        atoms, radii_source="alvarez", radii_offset=-0.75, rotation=True
    )
    supercell_lookups = []
    get_symbols = Atoms.get_chemical_symbols

    def counted_symbols(structure):
        if structure is finder.supercell:
            supercell_lookups.append(len(structure))
        return get_symbols(structure)

    monkeypatch.setattr(Atoms, "get_chemical_symbols", counted_symbols)
    result = finder.get_group_data()

    assert result["dimensionality"] == expected
    # One full-list lookup per reduced group, rather than per output atom.
    assert len(supercell_lookups) == sum(d != 3 for d in expected)
    symbols = np.asarray(get_symbols(atoms))
    for ids, group_symbols, tags in zip(
        result["unit_cell_ids"], result["chemical_symbols"], result["tags"], strict=True
    ):
        np.testing.assert_array_equal(group_symbols, symbols[ids])
        np.testing.assert_array_equal(tags, atoms.get_tags()[ids])
    assert atoms == original
    np.testing.assert_array_equal(atoms.get_tags(), original.get_tags())
