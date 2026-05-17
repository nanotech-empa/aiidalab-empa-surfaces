import unittest

import numpy as np
from ase import Atoms

from surfaces_tools.widgets.analyze_structure import get_types


class GetTypesTest(unittest.TestCase):
    def test_single_layer_slab_does_not_require_layer_spacing(self):
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

        self.assertTrue(np.all(atom_types == 1))
        self.assertEqual(len(layers), 1)
