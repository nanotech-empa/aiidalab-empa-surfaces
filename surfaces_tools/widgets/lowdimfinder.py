# -*- coding: utf-8 -*-
"""
Low dimensionality atom group finder, which analyses a bulk crystal and returns
groups of atoms which are held together by weak (van der Waals) forces as
separate structures.
"""

__copyright__ = (
    "Copyright (c), 2014-2022, École Polytechnique Fédérale de Lausanne (EPFL), Switzerland, "
    "Laboratory of Theory and Simulation of Materials (THEOS). All rights reserved."
)
__license__ = (
    "Non-Commercial, End-User Software License Agreement, see LICENSE.txt file"
)
__version__ = "0.3.0"

import numpy as np

from ase import Atoms
from numpy import isscalar
from numbers import Number


## Source: http://chemwiki.ucdavis.edu/Reference/Reference_Tables/Atomic_and_Molecular_Properties/A3%3A_Covalent_Radii
# http://dx.doi.org/10.1039/b801115j, checked in paper
_map_atomic_number_radii_cordero = {
    1: 0.31,
    2: 0.28,
    3: 1.28,
    4: 0.96,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    10: 0.58,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    18: 1.06,
    19: 2.03,
    20: 1.76,
    21: 1.7,
    22: 1.6,
    23: 1.53,
    24: 1.39,
    25: 1.39,
    26: 1.32,
    27: 1.26,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    31: 1.22,
    32: 1.2,
    33: 1.19,
    34: 1.2,
    35: 1.2,
    36: 1.16,
    37: 2.2,
    38: 1.95,
    39: 1.9,
    40: 1.75,
    41: 1.64,
    42: 1.54,
    43: 1.47,
    44: 1.46,
    45: 1.42,
    46: 1.39,
    47: 1.45,
    48: 1.44,
    49: 1.42,
    50: 1.39,
    51: 1.39,
    52: 1.38,
    53: 1.39,
    54: 1.4,
    55: 2.44,
    56: 2.15,
    57: 2.07,
    58: 2.04,
    59: 2.03,
    60: 2.01,
    61: 1.99,
    62: 1.98,
    63: 1.98,
    64: 1.96,
    65: 1.94,
    66: 1.92,
    67: 1.92,
    68: 1.89,
    69: 1.9,
    70: 1.87,
    71: 1.87,
    72: 1.75,
    73: 1.7,
    74: 1.62,
    75: 1.51,
    76: 1.44,
    77: 1.41,
    78: 1.36,
    79: 1.36,
    80: 1.32,
    81: 1.45,
    82: 1.46,
    83: 1.48,
    84: 1.4,
    85: 1.5,
    86: 1.5,
    87: 2.6,
    88: 2.21,
    89: 2.15,
    90: 2.06,
    91: 2,
    92: 1.96,
    93: 1.9,
    94: 1.87,
    95: 1.8,
    96: 1.69,
}

# http://dx.doi.org/10.1002/chem.200901472
_map_atomic_number_radii_pyykko = {
    1: 0.32,
    2: 0.46,
    3: 1.33,
    4: 1.02,
    5: 0.85,
    6: 0.75,
    7: 0.71,
    8: 0.63,
    9: 0.64,
    10: 0.67,
    11: 1.55,
    12: 1.39,
    13: 1.26,
    14: 1.16,
    15: 1.11,
    16: 1.03,
    17: 0.99,
    18: 0.96,
    19: 1.96,
    20: 1.71,
    21: 1.48,
    22: 1.36,
    23: 1.34,
    24: 1.22,
    25: 1.19,
    26: 1.16,
    27: 1.11,
    28: 1.1,
    29: 1.12,
    30: 1.18,
    31: 1.24,
    32: 1.21,
    33: 1.21,
    34: 1.16,
    35: 1.14,
    36: 1.17,
    37: 2.1,
    38: 1.85,
    39: 1.63,
    40: 1.54,
    41: 1.47,
    42: 1.38,
    43: 1.28,
    44: 1.25,
    45: 1.25,
    46: 1.2,
    47: 1.28,
    48: 1.36,
    49: 1.42,
    50: 1.4,
    51: 1.4,
    52: 1.36,
    53: 1.33,
    54: 1.31,
    55: 2.32,
    56: 1.96,
    57: 1.8,
    58: 1.63,
    59: 1.76,
    60: 1.74,
    61: 1.73,
    62: 1.72,
    63: 1.68,
    64: 1.69,
    65: 1.68,
    66: 1.67,
    67: 1.66,
    68: 1.65,
    69: 1.64,
    70: 1.7,
    71: 1.62,
    72: 1.52,
    73: 1.46,
    74: 1.37,
    75: 1.31,
    76: 1.29,
    77: 1.22,
    78: 1.23,
    79: 1.24,
    80: 1.33,
    81: 1.44,
    82: 1.44,
    83: 1.51,
    84: 1.45,
    85: 1.47,
    86: 1.42,
    87: 2.23,
    88: 2.01,
    89: 1.86,
    90: 1.75,
    91: 1.69,
    92: 1.7,
    93: 1.71,
    94: 1.72,
    95: 1.66,
    96: 1.66,
    97: 1.68,
    98: 1.68,
    99: 1.65,
    100: 1.67,
    101: 1.73,
    102: 1.76,
    103: 1.61,
    104: 1.57,
    105: 1.49,
    106: 1.43,
    107: 1.41,
    108: 1.34,
    109: 1.29,
    110: 1.28,
    111: 1.21,
    112: 1.22,
    113: 1.36,
    114: 1.43,
    115: 1.62,
    116: 1.75,
    117: 1.65,
    118: 1.57,
}

# source: Batsanov, S. S. "Van der Waals radii of elements." Inorganic materials 37.9 (2001): 871-885.
# For the elements missing there, we used the radii of Pyykko et al (see above)
# to which we added 0.8 Angstrom (rule of thumb mentioned in Batsanov paper,
# working quite well on the elements present in both references)
_map_atomic_number_radii_van_der_Waals = {
    1: 1.1,
    2: 1.26,
    3: 2.2,
    4: 1.9,
    5: 1.8,
    6: 1.7,
    7: 1.6,
    8: 1.55,
    9: 1.5,
    10: 1.4700000000000002,
    11: 2.4,
    12: 2.2,
    13: 2.1,
    14: 2.1,
    15: 1.95,
    16: 1.8,
    17: 1.8,
    18: 1.76,
    19: 2.8,
    20: 2.4,
    21: 2.3,
    22: 2.15,
    23: 2.05,
    24: 2.05,
    25: 2.05,
    26: 2.05,
    27: 2.0,
    28: 2.0,
    29: 2.0,
    30: 2.1,
    31: 2.1,
    32: 2.1,
    33: 2.05,
    34: 1.9,
    35: 1.9,
    36: 1.97,
    37: 2.9,
    38: 2.55,
    39: 2.4,
    40: 2.3,
    41: 2.15,
    42: 2.1,
    43: 2.05,
    44: 2.05,
    45: 2.0,
    46: 2.05,
    47: 2.1,
    48: 2.2,
    49: 2.2,
    50: 2.25,
    51: 2.2,
    52: 2.1,
    53: 2.1,
    54: 2.1100000000000003,
    55: 3.0,
    56: 2.7,
    57: 2.5,
    58: 2.4299999999999997,
    59: 2.56,
    60: 2.54,
    61: 2.5300000000000002,
    62: 2.52,
    63: 2.48,
    64: 2.49,
    65: 2.48,
    66: 2.4699999999999998,
    67: 2.46,
    68: 2.45,
    69: 2.44,
    70: 2.5,
    71: 2.42,
    72: 2.25,
    73: 2.2,
    74: 2.1,
    75: 2.05,
    76: 2.0,
    77: 2.0,
    78: 2.05,
    79: 2.1,
    80: 2.05,
    81: 2.2,
    82: 2.3,
    83: 2.3,
    84: 2.25,
    85: 2.27,
    86: 2.2199999999999998,
    87: 3.0300000000000002,
    88: 2.8099999999999996,
    89: 2.66,
    90: 2.4,
    91: 2.49,
    92: 2.3,
    93: 2.51,
    94: 2.52,
    95: 2.46,
    96: 2.46,
    97: 2.48,
    98: 2.48,
    99: 2.45,
    100: 2.4699999999999998,
    101: 2.5300000000000002,
    102: 2.56,
    103: 2.41,
    104: 2.37,
    105: 2.29,
    106: 2.23,
    107: 2.21,
    108: 2.14,
    109: 2.09,
    110: 2.08,
    111: 2.01,
    112: 2.02,
    113: 2.16,
    114: 2.23,
    115: 2.42,
    116: 2.55,
    117: 2.45,
    118: 2.37,
}

# source: Alvarez, S., "A cartography of the van der Waals territories",
# Dalton Trans., 2013, 42, 8617
_map_atomic_number_radii_van_der_Waals_alvarez = {
    1: 1.2,
    2: 1.43,
    3: 2.12,
    4: 1.98,
    5: 1.91,
    6: 1.77,
    7: 1.66,
    8: 1.5,
    9: 1.46,
    10: 1.58,
    11: 2.5,
    12: 2.51,
    13: 2.25,
    14: 2.19,
    15: 1.9,
    16: 1.89,
    17: 1.82,
    18: 1.83,
    19: 2.73,
    20: 2.62,
    21: 2.58,
    22: 2.46,
    23: 2.42,
    24: 2.45,
    25: 2.45,
    26: 2.44,
    27: 2.4,
    28: 2.4,
    29: 2.38,
    30: 2.39,
    31: 2.32,
    32: 2.29,
    33: 1.88,
    34: 1.82,
    35: 1.86,
    36: 2.25,
    37: 3.21,
    38: 2.84,
    39: 2.75,
    40: 2.52,
    41: 2.56,
    42: 2.45,
    43: 2.44,
    44: 2.46,
    45: 2.44,
    46: 2.15,
    47: 2.53,
    48: 2.49,
    49: 2.43,
    50: 2.42,
    51: 2.47,
    52: 1.99,
    53: 2.04,
    54: 2.06,
    55: 3.48,
    56: 3.03,
    57: 2.98,
    58: 2.88,
    59: 2.92,
    60: 2.95,
    62: 2.9,
    63: 2.87,
    64: 2.83,
    65: 2.79,
    66: 2.87,
    67: 2.81,
    68: 2.83,
    69: 2.79,
    70: 2.8,
    71: 2.74,
    72: 2.63,
    73: 2.53,
    74: 2.57,
    75: 2.49,
    76: 2.48,
    77: 2.41,
    78: 2.29,
    79: 2.32,
    80: 2.45,
    81: 2.47,
    82: 2.6,
    83: 2.54,
    89: 2.8,
    90: 2.93,
    91: 2.88,
    92: 2.71,
    93: 2.82,
    94: 2.81,
    95: 2.83,
    96: 3.05,
    97: 3.4,
    98: 3.05,
    99: 2.7,
}


class LowDimFinderExc(Exception):
    pass


class WeirdStructureExc(LowDimFinderExc):
    """
    Raised when a weird structure, which the LowDimfinder cannot handle, is found.
    """


class GroupOverTwoCellsNoConnectionsExc(LowDimFinderExc):
    """
    Raised when the group expands over at least two cells, but no connection is found.
    """


class NoRadiiListExc(LowDimFinderExc):
    """
    Raised when no valid radii list is defined.
    """


class LowDimFinder:
    """
    Take an aiida_structure, analyse the structure and return all bonded groups of atoms.

    :param bond_margin: percentage which is added to the
      bond length (default: 0).
    :param radii_offset: distance (in Angstrom) that is added to
      each radius (before using the margin above) (default: 0).
      In the end, the criterion which defines if atoms are bonded or not is
      that two atoms X1 and X2 are bonded if their distance is smaller than
      :math:`[R(X1) + R(X2) + 2\\text{radii_offset}] \\cdot (1 + \\text{bond_margin})`,
      where :math:`R(X)` denotes the radius of a species X.
    :param radii_source: Can be either ``cordero`` (default), ``pyykko``,
        ``vdw`` (for Batsanov van der Walls radii), or ``alvarez``
        (for Alvarez van der Waals radii), the lowdimfinder uses
        the radii list from the corresponding papers.
        Another possibility is ``custom``, in that case the list of radii has to be
        defined with the parameter custom_radii.
    :param custom_radii: If the radii_list is set to ``custom``. The dictionary passed with this
        parameters is used for the radii.
    :param vacuum_space: The space added around the atoms of the structures with lower
        dimensionality. (default: 20 angstrom)
    :param min_supercell: size of the supercell at which the search for lower dimensionality groups
        is started. (default: 3)
    :param max_supercell: size at which the search for lower dimensionality groups is stopped. The search
        always starts at a supercell size of 2. If the dimensionality of the largest group is 0,
        the supercell size is increased by 1, to increase the chance to catch weird structured groups.
        (default: 3)
    :param full_periodicity: If True, it sets the periodic boundary conditions or the reduced
        structures to ``[True, True, True]``, otherwise if False (default) the pbc are set according
        the dimensionality.
        0D: ``[False,False,False]``, 1D: ``[False,False,True]``,
        2D: ``[True, True, False]``, 3D: ``[True, True, True]``.
    :param rotation: If True, it rotates the reduced 1D structures into the z-axis and 2D structures in
        the x-y plane (default: False).
    :param orthogonal_axis_2D: If True (default), define the 3rd vector orthogonal to the layer,
        otherwise set it in direction of the closest unconnected periodic site.
    """

    def __init__(self, aiida_structure, **kwargs):

        self.params = {
            "bond_margin": 0.0,
            "radii_offset": 0.0,
            "vacuum_space": 20,
            "max_supercell": 3,
            "min_supercell": 3,
            "full_periodicity": False,
            "rotation": False,
            "radii_source": "cordero",
            "orthogonal_axis_2D": True,
            "custom_radii": None,
        }
        self.setup_builder(**kwargs)

        # starting with a min_supercell size and increase it until
        # reaching max_supercell
        self.supercell_size = self.params["min_supercell"]

        self.aiida_structure = aiida_structure
        self.structure = self.aiida_structure
        self.n_unit = len(self.structure)  # number of atoms in input structure
        self.supercell = self.structure.repeat(self.supercell_size)

        self._group_number = None  # index of group that is analysed
        self._low_dim_index = (
            0  # index of reduced structure that is currently calculated
        )
        self._found_unit_cell_atoms = set(
            []
        )  # set of all the input structure atoms that have been attributed to a group

        # independent variables, add those you want to have as output to get_group_data()
        self._groups = []
        self._unit_cell_groups = []
        self._atoms_in_one_cell = []
        self._connection_counter = []
        self._connected_positions = []
        self._unconnected_positions = []
        self._vectors = []
        self._shortest_unconnected_vector = []

        # to be returned by get_group_data()
        self._dimensionality = []
        self._chemical_formula = []
        self._positions = []
        self._chemical_symbols = []
        self._cell = []
        self._tags = []

        self._reduced_ase_structures = []
        self._rotated_structures = []
        self.reduced_aiida_structures = []
        self._3D_structures_with_layer_lattice = []

        if self.params["radii_source"] == "custom":
            self._map_atomic_number_radii_custom = self.params["custom_radii"]
        else:
            self._map_atomic_number_radii_custom = None

    def setup_builder(self, **kwargs):
        """
        Change the builder parameters.
        Pass as kwargs the internal parameters to change.

        .. note:: no checks are performed on the validity of the passed parameters.
        """
        for k, v in kwargs.items():
            if k in self.params.keys():
                self.params[k] = v

    def _define_groups(self):  # pylint: disable=too-many-locals
        """
        Return the different groups found in the cell.
        """
        # import time
        # start_time = time.time()
        n_sc = len(self.supercell)
        # get distances between all the atoms
        coords = self.supercell.get_positions()

        _map_atomic_number_radii = {
            "cordero": _map_atomic_number_radii_cordero,
            "pyykko": _map_atomic_number_radii_pyykko,
            "vdw": _map_atomic_number_radii_van_der_Waals,
            "alvarez": _map_atomic_number_radii_van_der_Waals_alvarez,
            "custom": self._map_atomic_number_radii_custom,
        }
        radii_source = self.params["radii_source"]
        try:
            table = _map_atomic_number_radii[radii_source]
        except KeyError:
            raise NoRadiiListExc()
        atomic_numbers = self.supercell.get_atomic_numbers()
        radii = [table[atomic_number] for atomic_number in atomic_numbers]
        radii_array = np.array(radii) + self.params["radii_offset"]
        pairs = get_bonded_pairs(
            coords, radii_array, factor=1.0 + self.params["bond_margin"], use_scipy=True
        )

        set_list = []
        for _ in range(n_sc):
            set_list.append(set())
        for i, j in pairs:
            set_list[i].add(j)
            set_list[j].add(i)

        # At the beginning, set_list contains only first neighbours
        # At the end, set_list will contain all neighbours at any
        # hopping distance

        # Atoms we already visited
        visited = set()
        groups = []

        # I run over all atoms
        for atom in range(len(set_list)):  # pylint: disable=consider-using-enumerate
            # atom not in visited --> new atom group
            if atom not in visited:
                # fresh set of connected atoms.
                # Initially we add the analysed atom.
                connected_atoms = set([atom])
                new_neighs = set_list[atom] - connected_atoms

                all_neighbours = connected_atoms.copy()
                # Loop stops when no new neighbours are found.
                while new_neighs:
                    # conneted atoms are a set of atoms that we know are already
                    # connected to the atom (or atom group)
                    # that we are considering.
                    connected_atoms.update(new_neighs)
                    # I update the set of all neighbours of the connected atoms
                    all_neighbours.update(set.union(*(set_list[a] for a in new_neighs)))
                    # new_neigbours are the all the neighbours of the connected atoms
                    # minus the connected atoms.
                    new_neighs = all_neighbours - connected_atoms

                # Add the connected atoms to the visited list
                visited.update(connected_atoms)
                groups.append(sorted(connected_atoms))

        # Returns list of groups sorted by number of atoms
        self._groups = sorted(groups, key=len, reverse=True)

    def _define_unit_cell_group(self):
        """
        Analyse a group and define which atoms form the unit cell of the reduced group.
        unit_cell_sites are the corresponding periodic sites in the unit cell
        .. note:: the unit_cell_sites do not have to be equivalent, to the unit cell atoms of
            the reduced group. The group can be dispersed over two or more cells
        """
        group = self.groups[self._group_number]

        unit_cell_sites = set([])
        reduced_unit_cell_group = set([])
        for i in group:
            if i % self.n_unit not in unit_cell_sites:
                unit_cell_sites.add(i % self.n_unit)
                reduced_unit_cell_group.add(i)

        self._unit_cell_groups.append(sorted(list(reduced_unit_cell_group)))

        self._found_unit_cell_atoms.update(unit_cell_sites)

        if unit_cell_sites == reduced_unit_cell_group:
            self._atoms_in_one_cell.append(True)
        else:
            self._atoms_in_one_cell.append(False)

    def _define_number_of_connections(self):
        """
        The connections of a periodic site with its images in the other :math:`N^3-1`
        cells are checked.
        """

        # current analysed group
        # search the site with the maximum periodic connection in group
        group = self.groups[self._group_number]

        # translate all idx back to the unit cell
        group_in_unit_cell = [idx % self.n_unit for idx in group]

        # find idx with maximum occurence in list
        counter_dict = {}

        for idx in group_in_unit_cell:
            counter_dict[idx] = counter_dict.get(idx, 0) + 1
        # make a list of tuples with (counter, idx)
        tuple_list = [(counter, idx) for idx, counter in counter_dict.items()]

        # periodic site in unit cell with highest occurrency
        counter, periodic_site_in_unit_cell = max(tuple_list)

        self._connection_counter.append(counter)

        # list of connected periodic sites
        pos = []
        # list of unconnected periodic sites
        unpos = []

        # get the positions of all connected and unconnected sites
        for i in range(self.supercell_size**3):
            if i * self.n_unit + periodic_site_in_unit_cell in group:
                pos.append(
                    self.supercell.get_positions()[
                        i * self.n_unit + periodic_site_in_unit_cell
                    ]
                )
            else:
                unpos.append(
                    self.supercell.get_positions()[
                        i * self.n_unit + periodic_site_in_unit_cell
                    ]
                )

        self._connected_positions.append(pos)
        self._unconnected_positions.append(unpos)

        if (
            self._connection_counter == 1
            and self.supercell_size < self.params["max_supercell"]
            and self._low_dim_index == 0
        ):

            self.supercell_size = self.supercell_size + 1
            # reset all the group data
            self._groups = []
            self._unit_cell_groups = []
            self._atoms_in_one_cell = []
            self._connection_counter = []
            self._connected_positions = []
            self._vectors = []
            self._dimensionality = []

            self.supercell = self.structure.repeat(self.supercell_size)

            self._define_number_of_connections()

    def _define_dimensionality(self):  # pylint: disable=too-many-locals
        """
        Vectors are defined between the connected periodic positions of the group.
        The rank
        of the matrix made by all those vectors gives the dimensionality of the group.
        0D: an array of three orthonormal vectors is created.
        1D: The shortest vector between periodic sites is taken plus two orthonormal
            vectors are created.
        2D: Two shortest linear independent vectors taken plus an orthonormal
            vector created
        3D: Takes the vectors of input structure
        """

        def l_are_equals(a, b):
            # function to compare lengths
            return abs(a - b) <= 1e-5

        def a_are_equals(a, b):
            # function to compare angles (actually, cosines)
            return abs(a - b) <= 1e-5

        # vector to connected positions
        vectors = []
        for i in self._get_connected_positions()[self._low_dim_index]:
            if i is not self._get_connected_positions()[self._low_dim_index][0]:
                vectors.append(i - self._connected_positions[self._low_dim_index][0])
        self._dimensionality.append(np.linalg.matrix_rank(vectors))

        # vectors of first connected site to unconnected periodic sites
        unconnected_vectors = []
        for i in self._get_unconnected_positions()[self._low_dim_index]:
            unconnected_vectors.append(
                i - self._connected_positions[self._low_dim_index][0]
            )

        if self._dimensionality[self._low_dim_index] == 0:
            self._vectors.append(
                [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
            )

        elif self._dimensionality[self._low_dim_index] == 1:
            idx = find_shortest_vector(vectors)
            normal_vector1 = np.cross(
                vectors[idx], [1, 0, 0]  # pylint: disable=invalid-sequence-index
            )
            if np.linalg.norm(normal_vector1) < 0.000001:
                normal_vector1 = np.cross(
                    vectors[idx], [0, 1, 0]  # pylint: disable=invalid-sequence-index
                )
            normal_vector2 = np.cross(
                vectors[idx], normal_vector1  # pylint: disable=invalid-sequence-index
            )
            normal_vector1 = normal_vector1 / np.linalg.norm(normal_vector1)
            normal_vector2 = normal_vector2 / np.linalg.norm(normal_vector2)
            self._vectors.append([normal_vector1, normal_vector2, vectors[0]])

        elif self._dimensionality[self._low_dim_index] == 2:
            idx = find_shortest_vector(vectors)
            vector1 = vectors.pop(idx)
            idx = find_shortest_vector(vectors)
            vector2 = vectors.pop(idx)
            if self.params["orthogonal_axis_2D"]:
                vector3 = np.cross(vector1, vector2)

                while np.linalg.norm(vector3) < 0.0000001:
                    idx = find_shortest_vector(vectors)
                    vector2 = vectors.pop(idx)
                    vector3 = np.cross(vector1, vector2)

            else:
                # to prevent case where in a supercell the two shortest vectors
                # are linearly dependent
                while np.linalg.norm(np.cross(vector1, vector2)) < 0.0000001:
                    idx = find_shortest_vector(vectors)
                    vector2 = vectors.pop(idx)

                # do same for 3rd vector !!!
                idx = find_shortest_vector(unconnected_vectors)
                vector3 = unconnected_vectors.pop(idx)

                while (
                    np.linalg.norm(np.dot(np.cross(vector1, vector2), vector3))
                    < 0.0000001
                ):
                    idx = find_shortest_vector(unconnected_vectors)
                    vector3 = unconnected_vectors.pop(idx)

                self._shortest_unconnected_vector.append(vector3)

            vector3 = vector3 / np.linalg.norm(vector3)

            # check the hexagonal case
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)
            cosphi = np.dot(vector1, vector2) / norm_vector1 / norm_vector2
            if l_are_equals(norm_vector1, norm_vector2) and a_are_equals(cosphi, 0.5):
                # change the lattice vectors to get a "standard" hexagonal cell
                # (compliant with the KpointsData class)
                vector2 = vector2 - vector1

            self._vectors.append([vector1, vector2, vector3])

        elif self._dimensionality[self._low_dim_index] == 3:
            self._vectors.append(self.structure.cell)

    def _define_reduced_ase_structure(
        self,
    ):  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        """
        Append the reduced group of atoms corresponding to the currently analysed group
        to the list of reduced ase structures.
        """
        min_z = None
        max_z = None
        min_x = None
        min_y = None
        max_x = None
        max_y = None

        symbols = []
        positions = []
        tags = []

        # 3 Dimensional --> return input structure
        if self._get_dimensionality()[self._low_dim_index] == 3:

            self._reduced_ase_structures.append(self.structure)
            self._chemical_formula.append(self.structure.get_chemical_formula())
            self._positions.append(self.structure.get_positions().tolist())
            self._chemical_symbols.append(self.structure.get_chemical_symbols())
            self._cell.append(self.structure.cell.tolist())
            self._tags.append(self.structure.get_tags().tolist())

            self._vectors[self._low_dim_index][0] = self.structure.cell.tolist()[0]
            self._vectors[self._low_dim_index][1] = self.structure.cell.tolist()[1]
            self._vectors[self._low_dim_index][2] = self.structure.cell.tolist()[2]
            # has to be called in case a structure is made of a 3D compound
            # and a lower D compound.
            self._get_unit_cell_groups()

            return

        if self._get_dimensionality()[self._low_dim_index] == 2:
            # get positions and chemical symbols
            for i in self._get_unit_cell_groups()[self._low_dim_index]:
                positions.append(
                    [
                        self.supercell.positions[i][0],
                        self.supercell.positions[i][1],
                        self.supercell.positions[i][2],
                    ]
                )
                symbols.append(self.supercell.get_chemical_symbols()[i])
                tags.append(self.supercell.get_tags()[i])
                # min and max z
                normal_vec_z = np.cross(
                    self._vectors[self._low_dim_index][0],
                    self._vectors[self._low_dim_index][1],
                )
                current_z = np.dot(
                    positions[-1], normal_vec_z / np.linalg.norm(normal_vec_z)
                )

                if min_z is None or min_z > current_z:
                    min_z = current_z
                if max_z is None or max_z < current_z:
                    max_z = current_z

            positions = np.array(positions)

            if self.params["orthogonal_axis_2D"]:

                positions = (
                    positions
                    - (min_z - self.params["vacuum_space"] / 2)
                    * self._vectors[self._low_dim_index][2]
                )
                self._vectors[self._low_dim_index][2] = self._vectors[
                    self._low_dim_index
                ][2] * (self.params["vacuum_space"] + max_z - min_z)

            else:
                # in order to keep the distance between the layers equal to the vacuum space
                # get vector orthogonal to layer
                normal_vec = np.cross(
                    self._vectors[self._low_dim_index][0],
                    self._vectors[self._low_dim_index][1],
                )

                # cosinus between third vector and normal vector
                cos = abs(
                    np.dot(normal_vec, self._vectors[self._low_dim_index][2])
                    / np.linalg.norm(normal_vec)
                    / np.linalg.norm(self._vectors[self._low_dim_index][2])
                )

                # translate the lengths into direction of the third vector
                vacuum_space_in_third_direction = (
                    float(self.params["vacuum_space"]) / cos
                )
                max_z = max_z / cos
                min_z = min_z / cos

                positions = (
                    positions
                    - (min_z - vacuum_space_in_third_direction / 2)
                    * self._vectors[self._low_dim_index][2]
                )
                self._vectors[self._low_dim_index][2] = self._vectors[
                    self._low_dim_index
                ][2] * (vacuum_space_in_third_direction + max_z - min_z)

            pbc = [True, True, False]

        elif self._get_dimensionality()[self._low_dim_index] == 1:
            for i in self._get_unit_cell_groups()[self._low_dim_index]:
                positions.append(
                    [
                        self.supercell.positions[i][0],
                        self.supercell.positions[i][1],
                        self.supercell.positions[i][2],
                    ]
                )
                symbols.append(self.supercell.get_chemical_symbols()[i])
                tags.append(self.supercell.get_tags()[i])

                if min_x is None or min_x > np.dot(
                    positions[-1], self._vectors[self._low_dim_index][0]
                ):
                    min_x = np.dot(positions[-1], self._vectors[self._low_dim_index][0])
                if max_x is None or max_x < np.dot(
                    positions[-1], self._vectors[self._low_dim_index][0]
                ):
                    max_x = np.dot(positions[-1], self._vectors[self._low_dim_index][0])
                if min_y is None or min_y > np.dot(
                    positions[-1], self._vectors[self._low_dim_index][1]
                ):
                    min_y = np.dot(positions[-1], self._vectors[self._low_dim_index][1])
                if max_y is None or max_y < np.dot(
                    positions[-1], self._vectors[self._low_dim_index][1]
                ):
                    max_y = np.dot(positions[-1], self._vectors[self._low_dim_index][1])

            positions = np.array(positions)
            positions = (
                positions
                - (min_x - self.params["vacuum_space"] / 2)
                * self._vectors[self._low_dim_index][0]
            )
            positions = (
                positions
                - (min_y - self.params["vacuum_space"] / 2)
                * self._vectors[self._low_dim_index][1]
            )

            self._vectors[self._low_dim_index][0] = self._vectors[self._low_dim_index][
                0
            ] * (self.params["vacuum_space"] + max_x - min_x)
            self._vectors[self._low_dim_index][1] = self._vectors[self._low_dim_index][
                1
            ] * (self.params["vacuum_space"] + max_y - min_y)

            pbc = [False, False, True]

        elif self._get_dimensionality()[self._low_dim_index] == 0:

            for i in self._get_unit_cell_groups()[self._low_dim_index]:
                positions.append(
                    [
                        self.supercell.positions[i][0],
                        self.supercell.positions[i][1],
                        self.supercell.positions[i][2],
                    ]
                )
                symbols.append(self.supercell.get_chemical_symbols()[i])
                tags.append(self.supercell.get_tags()[i])

                if min_x is None or min_x > np.dot(
                    positions[-1], self._vectors[self._low_dim_index][0]
                ):
                    min_x = np.dot(positions[-1], self._vectors[self._low_dim_index][0])
                if max_x is None or max_x < np.dot(
                    positions[-1], self._vectors[self._low_dim_index][0]
                ):
                    max_x = np.dot(positions[-1], self._vectors[self._low_dim_index][0])
                if min_y is None or min_y > np.dot(
                    positions[-1], self._vectors[self._low_dim_index][1]
                ):
                    min_y = np.dot(positions[-1], self._vectors[self._low_dim_index][1])
                if max_y is None or max_y < np.dot(
                    positions[-1], self._vectors[self._low_dim_index][1]
                ):
                    max_y = np.dot(positions[-1], self._vectors[self._low_dim_index][1])
                if min_z is None or min_z > np.dot(
                    positions[-1], self._vectors[self._low_dim_index][2]
                ):
                    min_z = np.dot(positions[-1], self._vectors[self._low_dim_index][2])
                if max_z is None or max_z < np.dot(
                    positions[-1], self._vectors[self._low_dim_index][2]
                ):
                    max_z = np.dot(positions[-1], self._vectors[self._low_dim_index][2])

            pbc = [False, False, False]

            positions = np.array(positions)
            positions = (
                positions
                - (min_x - self.params["vacuum_space"] / 2)
                * self._vectors[self._low_dim_index][0]
            )
            positions = (
                positions
                - (min_y - self.params["vacuum_space"] / 2)
                * self._vectors[self._low_dim_index][1]
            )
            positions = (
                positions
                - (min_z - self.params["vacuum_space"] / 2)
                * self._vectors[self._low_dim_index][2]
            )

            self._vectors[self._low_dim_index][0] = self._vectors[self._low_dim_index][
                0
            ] * (self.params["vacuum_space"] + max_x - min_x)
            self._vectors[self._low_dim_index][1] = self._vectors[self._low_dim_index][
                1
            ] * (self.params["vacuum_space"] + max_y - min_y)
            self._vectors[self._low_dim_index][2] = self._vectors[self._low_dim_index][
                2
            ] * (self.params["vacuum_space"] + max_z - min_z)

        else:
            raise WeirdStructureExc("No dimensionality")

        if self.params["full_periodicity"]:
            pbc = [True, True, True]

        reduced_ase_structure = Atoms(
            cell=self._vectors[self._low_dim_index],
            pbc=pbc,
            symbols=symbols,
            positions=positions,
            tags=tags,
        )
        reduced_ase_structure.set_positions(
            reduced_ase_structure.get_positions(wrap=True)
        )

        if self.params["rotation"]:

            original_struc = self.structure.copy()
            rotated_original_structure, reduced_ase_structure = self._rotate_structures(
                original_struc, reduced_ase_structure
            )

            self._rotated_structures.append(rotated_original_structure)

        self._reduced_ase_structures.append(reduced_ase_structure)
        self._chemical_formula.append(reduced_ase_structure.get_chemical_formula())
        self._positions.append(reduced_ase_structure.get_positions().tolist())
        self._chemical_symbols.append(reduced_ase_structure.get_chemical_symbols())
        self._cell.append(reduced_ase_structure.cell.tolist())
        self._tags.append(reduced_ase_structure.get_tags().tolist())

    def _define_reduced_ase_structures(self):
        """
        Define the ASE structures of groups of atoms, starting with the largest group
        until all atoms of the original structure are assigned to an ASE structure.
        .. note:: Groups containing atoms corresponding to sites in the original structure
            that have already been assigned to an ASE structure, are skipped. Therefore
            every atom in the original structure will be only present in one ASE structure.
        """

        # all the atoms of the input structure, which have to be allocated to groups
        test = set(range(self.n_unit))

        for idx, group in enumerate(self.groups):
            # break as soon as all atoms of the input structure are in a group
            if self._found_unit_cell_atoms == test:
                break
            self._group_number = idx

            # set of the periodic sites of the input structure
            unit_cell_group = {i % self.n_unit for i in group}

            # if the atoms corresponding to the group are only a periodic
            # replica of an existing low dimensionality
            # group, the group is skipped and the next one is analysed..
            if unit_cell_group.issubset(self._found_unit_cell_atoms):
                continue
            self._define_reduced_ase_structure()
            self._low_dim_index = self._low_dim_index + 1
        self._low_dim_index = self._low_dim_index - 1

    def _define_reduced_aiida_structures(self):
        """
        Converts all ASE structures into AiiDA structures
        """
        # from aiida.plugins import DataFactory
        if not self._reduced_ase_structures:
            self._define_reduced_ase_structures()
        # S = DataFactory("structure")

        self.reduced_aiida_structures = []
        for i in self._reduced_ase_structures:
            self.reduced_aiida_structures.append(i)

    def _rotate_structures(self, originalstruc, asestruc):
        """
        Rotates the third vector in z-axis and the first vector in x-axis.
        Layer (2D) into xy-plane, wire (1D) into z-direction

        :param originalstruc: unrotated original ASE structure
        :param asestruc: unrotated reduced ASE structure
        :return: rotated ASE original and reduced structures
        """
        if self.params["orthogonal_axis_2D"]:
            originalstruc.rotate(
                v=asestruc.cell[2], a=[0, 0, 1], center=(0, 0, 0), rotate_cell=True
            )
            asestruc.rotate(
                v=asestruc.cell[2], a=[0, 0, 1], center=(0, 0, 0), rotate_cell=True
            )

        else:
            # turn the normal vector into z direction
            normal_vec = np.cross(asestruc.cell[0], asestruc.cell[1])
            originalstruc.rotate(
                v=normal_vec, a=[0, 0, 1], center=(0, 0, 0), rotate_cell=True
            )
            asestruc.rotate(
                v=normal_vec, a=[0, 0, 1], center=(0, 0, 0), rotate_cell=True
            )
            # turn it into positive z direction (if needed)
            if asestruc.cell[2][2] < 0.0:
                originalstruc.rotate(
                    v=[0, 0, np.sign(asestruc.cell[2][2])],
                    a=[0, 0, 1],
                    center=(0, 0, 0),
                    rotate_cell=True,
                )
                asestruc.rotate(
                    v=[0, 0, np.sign(asestruc.cell[2][2])],
                    a=[0, 0, 1],
                    center=(0, 0, 0),
                    rotate_cell=True,
                )

        # finally rotate the first cell vector of asestruc into x (this is a
        # rotation inside the x-y plane so it will not impact its normal)
        originalstruc.rotate(
            v=asestruc.cell[0], a=[1, 0, 0], center=(0, 0, 0), rotate_cell=True
        )
        asestruc.rotate(
            v=asestruc.cell[0], a=[1, 0, 0], center=(0, 0, 0), rotate_cell=True
        )

        return originalstruc, asestruc

    def _rotate_ase_structure(self, asestruc):
        """
        Rotates the third vector in z-axis and the first vector in x-axis.
        Layer (2D) into xy-plane, wire (1D) into z-direction

        :param asestruc: unrotated reduced ASE structure
        :return: rotated ASE structure
        """
        if self.params["orthogonal_axis_2D"]:
            asestruc.rotate(
                v=asestruc.cell[2], a=[0, 0, 1], center=(0, 0, 0), rotate_cell=True
            )

        else:
            # turn the normal vector into z direction
            normal_vec = np.cross(asestruc.cell[0], asestruc.cell[1])
            asestruc.rotate(
                v=normal_vec, a=[0, 0, 1], center=(0, 0, 0), rotate_cell=True
            )
            # turn it into positive z direction (if needed)
            if asestruc.cell[2][2] < 0.0:
                asestruc.rotate(
                    v=[0, 0, np.sign(asestruc.cell[2][2])],
                    a=[0, 0, 1],
                    center=(0, 0, 0),
                    rotate_cell=True,
                )

        # finally rotate the first cell vector of asestruc into x (this is a
        # rotation inside the x-y plane so it will not impact its normal)
        asestruc.rotate(
            v=asestruc.cell[0], a=[1, 0, 0], center=(0, 0, 0), rotate_cell=True
        )

        return asestruc

    def _get_dimensionality(self):
        """
        Add dimensionality and normalized lattice vectors of the currently
        analyzed group to the corresponding lists

        Return the dimensionality list.
        """
        if len(self._dimensionality) <= self._low_dim_index:
            self._define_dimensionality()
        return self._dimensionality

    @property
    def groups(self):
        """
        Return all atoms in the supercell which are only held together by vdW
        forces as separate groups

        :return: groups of bonded atoms
        """
        if not self._groups:
            self._define_groups()
        return self._groups

    def _get_unit_cell_groups(self):
        """

        Return atom ids used to build the reduced ase structures.
        """

        if len(self._unit_cell_groups) <= self._low_dim_index:
            self._define_unit_cell_group()

        return self._unit_cell_groups

    def _get_number_of_connections(self):
        """
        Return the number of connected periodic positions in the supercell.
        """
        if len(self._connection_counter) <= self._low_dim_index:
            self._define_number_of_connections()
        return self._connection_counter

    def _get_connected_positions(self):
        """
        Return the connected periodic positions in the supercell.
        """
        if len(self._connection_counter) <= self._low_dim_index:
            self._define_number_of_connections()
        return self._connected_positions

    def _get_unconnected_positions(self):
        """
        Return the unconnected periodic positions in the supercell.
        """
        if len(self._connection_counter) <= self._low_dim_index:
            self._define_number_of_connections()
        return self._unconnected_positions

    def _get_reduced_ase_structures(self):
        """
        Return a list with all the lower dimensionality structures, as ASE atoms.
        """
        if not self._reduced_ase_structures:
            self._define_reduced_ase_structures()
        return self._reduced_ase_structures

    def get_reduced_aiida_structures(self):
        """
        Return a list with all the lower dimensionality structures found,
        as AiiDA structures.
        """
        if not self.reduced_aiida_structures:
            self._define_reduced_aiida_structures()
        return self.reduced_aiida_structures

    def get_group_data(self):
        """
        Return a dictionary with list of the dimensionality, chemical_formula,
        cell parameters, positions and chemical symbols of the atoms of the
        extracted structures.
        """
        _ = self.get_reduced_aiida_structures()
        return {
            "dimensionality": self._dimensionality,
            "chemical_formula": self._chemical_formula,
            "positions": self._positions,
            "chemical_symbols": self._chemical_symbols,
            "unit_cell_ids":self._unit_cell_groups,
            "cell": self._cell,
            "tags": self._tags,
        }

    def _get_rotated_structures(self):
        """
        Return rotated original structures if rotation is true.
        """

        _ = self.get_reduced_aiida_structures()

        # from aiida.plugins import DataFactory
        # S = DataFactory("structure")

        rotated_aiida_structures = []

        for asestruc in self._rotated_structures:
            rotated_aiida_structures.append(asestruc)

        return rotated_aiida_structures

    def _get_3D_structures_with_layer_lattice(self):
        """
        Return list of 3D structures with the same lattice as the layer,
        for all 2D structures found.
        """
        if not self._3D_structures_with_layer_lattice:
            self._define_3D_structures_with_layer_lattice()

        return self._3D_structures_with_layer_lattice

    def _define_3D_structures_with_layer_lattice(self):
        """
        Define 3D structures with the same lattice as the layers for all
        2D reduced structures found.
        """
        from aiida.plugins import (  # pylint:disable=import-error,import-outside-toplevel
            DataFactory,  # pylint:disable=import-error,import-outside-toplevel
        )

        S = DataFactory("structure")
        _ = self.get_reduced_aiida_structures()

        # 2D structure counter
        counter = 0

        for idx, dim in enumerate(self._dimensionality):
            # build 3D structure with layer lattice
            if dim == 2 and not self.params["orthogonal_axis_2D"]:

                positions = self.structure.get_positions()
                positions = positions - self.structure.cell[2]

                # get the two first vectors from the reduced ase structure and the shortest unconnected vector.
                asestruc = Atoms(
                    cell=[
                        self._vectors[idx][0],
                        self._vectors[idx][1],
                        self._shortest_unconnected_vector[counter],
                    ],
                    positions=self.structure.get_positions(),
                    symbols=self.structure.get_chemical_symbols(),
                    tags=self.structure.get_tags(),
                    pbc=[True, True, True],
                )

                asestruc.set_positions(asestruc.get_positions(wrap=True))

                if int(round(asestruc.get_volume() / self.structure.get_volume())) != 1:
                    # we need to do something else if the volume changes (it's a supercell)
                    # -> we use the supercell, wrap everything in, and remove
                    # the overlapping stuff
                    the_asestruc = Atoms(
                        cell=[
                            self._vectors[idx][0],
                            self._vectors[idx][1],
                            self._shortest_unconnected_vector[counter],
                        ],
                        positions=self.supercell.get_positions(),
                        symbols=self.supercell.get_chemical_symbols(),
                        pbc=[True, True, True],
                    )

                    pos_and_symbols = zip(
                        the_asestruc.get_scaled_positions().tolist(),
                        the_asestruc.get_chemical_symbols(),
                    )
                    the_pos, the_symbols = zip(*objects_set(pos_and_symbols))
                    asestruc = Atoms(
                        cell=[
                            self._vectors[idx][0],
                            self._vectors[idx][1],
                            self._shortest_unconnected_vector[counter],
                        ],
                        scaled_positions=the_pos,
                        symbols=the_symbols,
                        pbc=[True] * 3,
                    )

                if self.params["rotation"] is True:
                    asestruc = self._rotate_ase_structure(asestruc)

                counter = counter + 1

                self._3D_structures_with_layer_lattice.append(S(ase=asestruc))


def find_shortest_vector(array):
    """
    Takes an array of vectors and finds the shortest one.

    :param array: array of vectors
    :return idx: the index of the shortest vector in the array
    """
    idx = np.array([np.linalg.norm(vector) for vector in array]).argmin()
    return idx


def get_bonded_pairs(
    coords, radii_array, factor=1.0, use_scipy=True
):  # pylint: disable=too-many-locals
    """
    Get the list of pairs of connected atoms in the cell.
    (i, j) in the list means atom i and atom j are connected.

    :param coords: n*3 array with the positions to compare (n=number of
        atoms in the cell)
    :param radii_array: array of length n with the radii for
        each of the n atoms)
    :param factor: constant factor multiplied to all the bonds
    :param use_scipy: True to use the optimized scipy.spatial.cKDTree

    .. note:: if use_scipy is True, it requires scipy version >= 0.16.0

    :return: pairs of connected atoms
    """
    if use_scipy:
        try:
            import scipy.spatial  # pylint: disable=import-outside-toplevel

            tree = scipy.spatial.cKDTree(coords)
        except (ImportError, AttributeError):
            raise ValueError(
                "To use scipy and speed-up the lowdimfinder, you "
                "need scipy with a version >= 0.16.0"
            )
        max_radius = np.max(radii_array) * factor

        # prefiltering the pairs by twice the maximum radius
        pairs = list(tree.query_pairs(2.0 * max_radius))

        # among those find the pairs that are really connected
        if len(pairs) > 0:
            atom1_idx, atom2_idx = zip(*pairs)  # list of atoms within those pairs
            coords1 = coords[np.array(atom1_idx)]
            coords2 = coords[np.array(atom2_idx)]
            # radii of these atoms
            rad1 = radii_array[np.array(atom1_idx)] * factor
            rad2 = radii_array[np.array(atom2_idx)] * factor
            pairs_idx = np.nonzero(
                ((coords1 - coords2) ** 2).sum(axis=1) - (rad1 + rad2) ** 2 < 0.0
            )
            return np.array(pairs)[pairs_idx[0]]
        return []

    # Get the squared_distances between to lists of cartesian coordinates.
    # matrix(i, j) contains the squared distance between atom i and atom j.
    matrix = (coords[:, None, :] - coords[None, :, :]) ** 2
    dist_squared = np.sum(matrix, axis=2)
    radii_matrix = (radii_array + radii_array[:, None]) * factor
    pairs = zip(*np.nonzero(dist_squared - radii_matrix**2 < 0))
    return [pair for pair in pairs if pair[0] < pair[1]]


def numbers_are_equal(a, b, epsilon=1e-14):
    """
    Compare two numbers a and b within epsilon.
    :param a: float, int or any scalar
    :param b: float, int or any scalar
    :param epsilon: threshold above which a is considered different from b
    :return: boolean
    """
    return abs(a - b) < epsilon


def scalars_are_equal(a, b, **kwargs):
    """
    Compare two objects of any type, except list, arrays or dictionaries.
    Numbers are compared thanks to the ``numbers_are_equal`` function.
    :param a: any scalar object
    :param b: any scalar object
    :param kwargs: parameters passed to the function numbers_are_equal
    :return: boolean
    :raise: NonscalarError if either a or b is a list, an array or a dictionary
    """
    if isscalar(a) and isscalar(b):
        if isinstance(a, Number):
            return isinstance(b, Number) and numbers_are_equal(a, b, **kwargs)
        return a == b
    if (a is None) or (b is None):
        return a == b
    raise TypeError("a and b must be scalars")


def objects_set(objects, **kwargs):
    """
    Return a set made of objects compared between them using the
    'objects_are_equal' function
    :param objects: an iterable containing any kind of objects that can
        be compared using the 'objects_are_equal' function (list, dict, etc.)
    :param kwargs: additional keyword arguments to be passed to the
        'objects_are_equal' function (e.g. precision for floating point number)
    :return: a set of non-equal objects
    """
    the_set = []
    for obj in objects:
        if len([o for o in the_set if objects_are_equal(obj, o, **kwargs)]) == 0:
            the_set.append(obj)

    return the_set


def objects_are_equal(obj1, obj2, **kwargs):
    """
    Recursive function.
    Return True if obj1 is the same as obj2. Scalars are
    compared using the function ``scalars_are_equal``.
    Handles strings, floats, ints, booleans, as well as lists, arrays and
    dictionaries containing such objects (possibly nested).
    :param obj1: any object
    :param obj2: any object
    :param kwargs: parameters passed to the function scalars_are_equal
    :return: boolean
    """
    if isinstance(obj1, dict):
        if not isinstance(obj2, dict):
            return False
        obj1_keys = sorted(obj1.keys())
        obj2_keys = sorted(obj2.keys())
        if not objects_are_equal(obj1_keys, obj2_keys, **kwargs):
            return False
        if not objects_are_equal(
            [obj1[k] for k in obj1_keys], [obj2[k] for k in obj2_keys], **kwargs
        ):
            return False
        return True

    if isinstance(obj1, (list, np.ndarray, tuple)):
        if not isinstance(obj2, (list, np.ndarray, tuple)):
            return False
        if len(obj1) != len(obj2):
            return False
        for e1, e2 in zip(obj1, obj2):
            if np.isscalar(e1):
                if not np.isscalar(e2):
                    return False
                if not scalars_are_equal(e1, e2, **kwargs):
                    return False
            else:
                if not objects_are_equal(e1, e2, **kwargs):
                    return False
        return True

    try:
        return scalars_are_equal(obj1, obj2, **kwargs)
    except TypeError:
        raise TypeError("Type of obj1 and obj2 not recognized")
