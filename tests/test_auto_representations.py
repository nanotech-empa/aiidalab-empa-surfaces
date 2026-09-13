import unittest

from aiidalab_widgets_base import viewers
from ase import Atoms

from surfaces_tools.widgets.auto_representations import (
    AutoRepresentationWidget,
    apply_auto_representations,
    molecule_and_non_molecule_atoms,
)


class AutoRepresentationsTest(unittest.TestCase):
    def test_molecule_and_non_molecule_atoms(self):
        details = {"all_molecules": [[2, 0], [4]]}

        molecules_atoms, non_molecule_atoms = molecule_and_non_molecule_atoms(
            details, natoms=6
        )

        self.assertEqual(molecules_atoms, [0, 2, 4])
        self.assertEqual(non_molecule_atoms, [1, 3, 5])

    def test_apply_auto_representations(self):
        structure = Atoms(
            "Au2CH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            cell=[5.0, 5.0, 8.0],
            pbc=True,
        )
        viewer = viewers.StructureDataViewer()
        viewer.structure = structure

        molecules_atoms, non_molecule_atoms = apply_auto_representations(
            viewer, {"all_molecules": [[2, 3]]}
        )

        self.assertEqual(molecules_atoms, [2, 3])
        self.assertEqual(non_molecule_atoms, [0, 1])
        self.assertEqual(len(viewer._all_representations), 2)
        self.assertEqual(
            viewers.parse_representation_style_id(
                viewer._all_representations[0].style_id
            )["token"],
            "molecules",
        )
        self.assertEqual(viewer._all_representations[0].type.value, "ballstick")
        self.assertEqual(viewer._all_representations[0].selection.value, "3..4")
        self.assertEqual(viewer._all_representations[1].type.value, "spacefill")
        self.assertEqual(viewer._all_representations[1].selection.value, "1..2")
        expected_style_id = viewers.encode_representation_style_id(
            viewer.REPRESENTATION_PREFIX,
            representation_type="spacefill",
            size=3,
            color="element",
            token="nonmolecule",
        )
        self.assertEqual(viewer._all_representations[1].style_id, expected_style_id)

    def test_structure_change_resets_stale_auto_representations(self):
        first_structure = Atoms(
            "Au2CH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            cell=[5.0, 5.0, 8.0],
            pbc=True,
        )
        second_structure = Atoms(
            "Au3CH",
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            cell=[6.0, 5.0, 8.0],
            pbc=True,
        )
        viewer = viewers.StructureDataViewer()
        viewer.structure = first_structure
        auto_rep = AutoRepresentationWidget(viewer)

        apply_auto_representations(viewer, {"all_molecules": [[2, 3]]})
        self.assertEqual(len(viewer._all_representations), 2)
        first_style_id = viewer._all_representations[0].style_id

        viewer.structure = second_structure
        auto_rep.structure = second_structure

        self.assertEqual(len(viewer._all_representations), 1)
        # The current viewer reuses the first style for structures without
        # saved representations, but resets its atom selection to all atoms.
        self.assertEqual(viewer._all_representations[0].style_id, first_style_id)
        self.assertEqual(viewer._all_representations[0].type.value, "ballstick")
        self.assertEqual(viewer._all_representations[0].selection.value, "1..5")
        self.assertEqual(viewer.atoms_not_represented.value, "")
        self.assertEqual(auto_rep.status.value, "")
