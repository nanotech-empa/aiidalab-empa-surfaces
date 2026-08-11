import unittest

import numpy as np
from ase import Atoms

from surfaces_tools.widgets.inputs import (
    compare_replica_cells,
    interpolate_replicas,
    replica_group_divisors,
    validate_replica_pair,
)


def water(positions=None, cell=(10.0, 10.0, 10.0), pbc=(True, True, True)):
    return Atoms(
        "OH2",
        positions=positions
        or [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        cell=cell,
        pbc=pbc,
    )


class ValidateReplicaPairTest(unittest.TestCase):
    def test_accepts_identical_compositions(self):
        ok, message = validate_replica_pair(water(), water())
        self.assertTrue(ok)
        self.assertEqual(message, "")

    def test_accepts_moved_atoms(self):
        moved = water()
        moved.positions[1] += [0.3, 0.0, 0.0]
        ok, _ = validate_replica_pair(water(), moved)
        self.assertTrue(ok)

    def test_rejects_an_atom_count_change(self):
        ok, message = validate_replica_pair(water(), water() + Atoms("H", [[3, 3, 3]]))
        self.assertFalse(ok)
        self.assertIn("Atom count changed: 3 -> 4", message)

    def test_rejects_reordered_atoms_and_names_the_index(self):
        reordered = Atoms(
            "HOH",
            positions=water().positions,
            cell=water().cell,
            pbc=water().pbc,
        )
        ok, message = validate_replica_pair(water(), reordered)
        self.assertFalse(ok)
        self.assertIn("index 0", message)
        self.assertIn("O -> H", message)


class CompareReplicaCellsTest(unittest.TestCase):
    def test_quiet_when_cell_and_pbc_match(self):
        self.assertEqual(compare_replica_cells(water(), water()), "")

    def test_flags_a_differing_cell(self):
        difference = compare_replica_cells(water(), water(cell=(10.0, 10.0, 12.0)))
        self.assertIn("cell differs", difference)
        self.assertNotIn("PBC differs", difference)

    def test_flags_a_differing_pbc(self):
        difference = compare_replica_cells(water(), water(pbc=(True, True, False)))
        self.assertIn("PBC differs", difference)
        self.assertNotIn("cell differs", difference)


class ReplicaGroupDivisorsTest(unittest.TestCase):
    def test_smallest_reachable_replica_count(self):
        self.assertEqual(replica_group_divisors(2), [1, 2])

    def test_prime_replica_count(self):
        self.assertEqual(replica_group_divisors(7), [1, 7])

    def test_perfect_square_does_not_repeat_the_root(self):
        self.assertEqual(replica_group_divisors(9), [1, 3, 9])

    def test_single_replica(self):
        self.assertEqual(replica_group_divisors(1), [1])

    def test_zero_is_pinned_rather_than_raising(self):
        # Unreachable through the form, where the count is 2 + intermediates,
        # but the trait can be written from outside the widget.
        self.assertEqual(replica_group_divisors(0), [1])
        self.assertEqual(replica_group_divisors(-3), [1])


class InterpolateReplicasTest(unittest.TestCase):
    def setUp(self):
        self.first = water()
        self.last = water()
        self.last.positions += [1.0, 2.0, 0.0]

    def test_interpolates_onto_the_segment(self):
        for coefficient in (0.1, 0.9):
            with self.subTest(coefficient=coefficient):
                interpolated = interpolate_replicas(self.first, self.last, coefficient)
                expected = (
                    1.0 - coefficient
                ) * self.first.positions + coefficient * self.last.positions
                np.testing.assert_allclose(interpolated.positions, expected)

    def test_preserves_atom_count_and_symbols(self):
        interpolated = interpolate_replicas(self.first, self.last, 0.5)
        self.assertEqual(len(interpolated), len(self.first))
        self.assertEqual(
            interpolated.get_chemical_symbols(), self.first.get_chemical_symbols()
        )
        ok, _ = validate_replica_pair(self.first, interpolated)
        self.assertTrue(ok)

    def test_does_not_mutate_the_endpoints(self):
        before = self.first.positions.copy()
        interpolate_replicas(self.first, self.last, 0.5)
        np.testing.assert_allclose(self.first.positions, before)


if __name__ == "__main__":
    unittest.main()
