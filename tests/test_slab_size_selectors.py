"""Build/rebuild must retain the layout chosen before Add slab."""

from collections import Counter
from types import SimpleNamespace

import aiidalab_widgets_base as awb
import numpy as np
import pytest
import traitlets as tr
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from surfaces_tools.widgets import auto_representations, editors, inputs, slabs

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def phenalenyl():
    molecule = Chem.AddHs(Chem.MolFromSmiles("[CH]1C=CC2=CC=CC3=CC=CC1=C23"))
    assert AllChem.EmbedMolecule(molecule, randomSeed=42) == 0
    atoms = Atoms(
        [a.GetSymbol() for a in molecule.GetAtoms()],
        positions=molecule.GetConformer().GetPositions(),
        cell=[30, 30, 30],
        pbc=False,
    )
    atoms.center()
    assert Counter(atoms.get_chemical_symbols()) == {"C": 13, "H": 9}
    return atoms


@pytest.fixture
def slab_form(aiida_profile_tmp):
    assert aiida_profile_tmp.is_test_profile
    builder = editors.BuildSlab(title="Build slab")
    details = inputs.InputDetails(structure_info_separate=True)
    manager = awb.StructureManagerWidget(
        importers=[], editors=[builder], storable=False, node_class="StructureData"
    )
    # Preserve the production notebook's ordering: builder sees a new structure
    # before InputDetails has replaced the previous molecule classification.
    links = [
        tr.dlink((manager, "structure"), (builder, "molecule")),
        tr.dlink((manager, "structure"), (details, "structure")),
    ]
    details.structure_manager = manager
    links.append(tr.dlink((details, "details"), (builder, "details")))
    original = phenalenyl()
    manager.input_structure = original.copy()
    return SimpleNamespace(
        builder=builder,
        manager=manager,
        details=details,
        original=original,
        links=links,
    )


def set_size(form, nx, ny, nz):
    form.builder.nx_slider.value = nx
    form.builder.ny_slider.value = ny
    form.builder.nz_slider.value = nz


def assert_built_size(form, nx, ny, nz):
    builder = form.builder
    assert (
        builder.nx_slider.value,
        builder.ny_slider.value,
        builder.nz_slider.value,
    ) == (nx, ny, nz)
    counts = Counter(form.manager.structure.get_chemical_symbols())
    assert counts == {"C": 13, "H": 9 + 2 * nx * ny, "Au": 2 * nx * ny * nz}
    assert len(form.manager.structure) == 22 + 2 * nx * ny * (nz + 1)
    assert (
        form.manager.structure[:22].get_chemical_symbols()
        == form.original.get_chemical_symbols()
    )
    np.testing.assert_allclose(
        form.manager.structure[:22].get_all_distances(),
        form.original.get_all_distances(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        form.manager.structure.cell.lengths()[:2],
        np.array(slabs.slab_lx_ly["Au(111)"]) * [nx, ny],
    )
    molecule, rest = auto_representations.molecule_and_non_molecule_atoms(
        form.details.details, len(form.manager.structure)
    )
    assert molecule == list(range(22))
    assert rest == list(range(22, len(form.manager.structure)))


def test_build_and_rebuild_retain_requested_size_and_molecule(slab_form, monkeypatch):
    seen = []
    guess = slabs.guess_slab_size

    def record_guess(atoms, surface):
        seen.append(len(atoms))
        return guess(atoms, surface)

    monkeypatch.setattr(slabs, "guess_slab_size", record_guess)
    set_size(slab_form, 8, 5, 4)
    slab_form.builder.create_bttn.click()
    assert_built_size(slab_form, 8, 5, 4)
    assert seen == [], "Building must not autosize from the generated slab"

    set_size(slab_form, 10, 6, 4)
    slab_form.builder.create_bttn.click()
    assert_built_size(slab_form, 10, 6, 4)
    assert seen == []


def test_new_molecule_and_surface_still_offer_size_suggestions(slab_form):
    set_size(slab_form, 8, 5, 4)
    slab_form.builder.create_bttn.click()
    slab_form.builder.drop_surface.value = "Ag(111)"
    expected = slabs.guess_slab_size(slab_form.builder._molecule, "Ag(111)")
    assert (
        slab_form.builder.nx_slider.value,
        slab_form.builder.ny_slider.value,
    ) == expected
    assert slab_form.builder.nz_slider.value == 4

    replacement = phenalenyl()
    replacement.positions[:, 0] *= 1.1
    slab_form.manager.input_structure = replacement
    expected = slabs.guess_slab_size(replacement, "Ag(111)")
    assert (
        slab_form.builder.nx_slider.value,
        slab_form.builder.ny_slider.value,
    ) == expected
