import unittest

from surfaces_tools.widgets.lowdimfinder import LowDimFinder


class LowDimFinderContractTest(unittest.TestCase):
    def test_group_data_exposes_unit_cell_ids(self):
        finder = LowDimFinder.__new__(LowDimFinder)
        finder.get_reduced_aiida_structures = lambda: []
        finder._dimensionality = [2]
        finder._chemical_formula = ["C"]
        finder._positions = [[[0.0, 0.0, 0.0]]]
        finder._chemical_symbols = [["C"]]
        finder._unit_cell_groups = [[0]]
        finder._cell = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        finder._tags = [[0]]

        self.assertEqual(finder.get_group_data()["unit_cell_ids"], [[0]])
