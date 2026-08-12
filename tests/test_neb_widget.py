import types
import unittest
from unittest import mock

import numpy as np
import traitlets as tr
from aiida import orm
from ase import Atoms

from surfaces_tools.widgets import inputs
from surfaces_tools.widgets.inputs import (
    compare_replica_cells,
    interpolate_replicas,
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


class ReplicaPerGroupOptionsTest(unittest.TestCase):
    """The "# rep / group" choices, exercised through the widget.

    The divisor arithmetic is inline in NebWidget, so there is nothing to call
    directly. NebWidget builds without an AiiDA profile as long as no replica
    PK is set, which is what lets these run here.
    """

    def widget_with(self, n_replica):
        widget = inputs.NebWidget()
        widget.n_replica.value = n_replica
        return widget

    def test_offers_every_divisor_of_the_count(self):
        options = self.widget_with(15).n_replica_per_group.options
        self.assertEqual(list(options), [1, 3, 5, 15])

    def test_prime_count_offers_only_one_and_itself(self):
        options = self.widget_with(7).n_replica_per_group.options
        self.assertEqual(list(options), [1, 7])

    def test_keeps_a_selection_that_still_divides(self):
        widget = self.widget_with(15)
        widget.n_replica_per_group.value = 5
        widget.n_replica.value = 10
        self.assertEqual(widget.n_replica_per_group.value, 5)

    def test_resets_a_selection_that_no_longer_divides(self):
        widget = self.widget_with(15)
        widget.n_replica_per_group.value = 5
        widget.n_replica.value = 8
        self.assertEqual(widget.n_replica_per_group.value, 1)


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


def shifted_water(dx):
    """Water displaced along x: replicas that differ but stay compatible."""
    atoms = water()
    atoms.positions += [dx, 0.0, 0.0]
    return atoms


class FakeStructure(orm.StructureData):
    """A StructureData that needs no profile: nothing here touches the backend.

    ``label`` shadows the inherited property, whose setter would go looking for
    a backend entity - the widget labels the structures it stores.
    """

    label = ""

    def __init__(self, pk, atoms):
        self._pk, self._atoms = pk, atoms

    @property
    def pk(self):
        return self._pk

    @property
    def uuid(self):
        return f"uuid-{self._pk}"

    @property
    def is_stored(self):
        return True

    def get_ase(self):
        return self._atoms


def fake_neb(pk, structure, n_replica, produced=None):
    """A NEB workchain node. `produced` below `n_replica` means it never finished."""
    produced = n_replica if produced is None else produced
    return types.SimpleNamespace(
        pk=pk,
        uuid=f"uuid-{pk}",
        inputs=types.SimpleNamespace(
            structure=structure, neb_params={"number_of_replica": n_replica}
        ),
        outputs=types.SimpleNamespace(
            **{f"opt_replica_{i:03d}": f"<replica {i}>" for i in range(produced)}
        ),
    )


class WidgetTestCase(unittest.TestCase):
    """Drives a NebWidget against stub nodes, so no AiiDA profile is needed.

    `inputs.orm` is replaced wholesale: PKs resolve out of `self.nodes`, and
    whatever the widget stores lands in `self.stored`. This couples the tests
    to `orm` being a module-level name in inputs.py - if it ever becomes a
    local import, the patch stops biting and these pass vacuously.
    """

    def setUp(self):
        self.nodes = {}
        self.stored = []
        test = self

        class StoredStructure(FakeStructure):
            def __init__(self, ase):
                FakeStructure.__init__(self, 900 + len(test.stored), ase)
                test.stored.append(self)
                test.nodes[self.pk] = self

            def store(self):
                return self

        patcher = mock.patch.object(
            inputs,
            "orm",
            types.SimpleNamespace(
                StructureData=StoredStructure,
                load_node=lambda pk: test.nodes[int(pk)],
            ),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def structure(self, pk, dx=0.0):
        """Register a structure under `pk`, displaced `dx` along x."""
        self.nodes[pk] = FakeStructure(pk, shifted_water(dx))
        return self.nodes[pk]

    def band(self, *pks):
        """A widget whose replicas are `pks`, endpoints first and last."""
        widget = inputs.NebWidget()
        widget.initial_row.pk.value = pks[0]
        widget.last_row.pk.value = pks[-1]
        for _ in pks[1:-1]:
            widget.initial_row.insert.click()
        for row, pk in zip(widget.replica_rows, pks[1:-1]):
            row.pk.value = pk
        return widget


class ValidateReplicasTest(WidgetTestCase):
    """The gate that refuses a broken band at submission time.

    Nothing else stops a submission: the coloured row text is display only.
    """

    def setUp(self):
        super().setUp()
        self.structure(1, 0.0)
        self.structure(2, 4.0)
        self.structure(3, 8.0)
        self.nodes[9] = FakeStructure(9, water() + Atoms("H", [[3.0, 3.0, 3.0]]))

    def test_returns_every_replica_in_order(self):
        nodes = self.band(1, 2, 3).validate_replicas()
        self.assertEqual([node.pk for node in nodes], [1, 2, 3])

    def test_accepts_a_band_of_just_the_endpoints(self):
        nodes = self.band(1, 2).validate_replicas()
        self.assertEqual([node.pk for node in nodes], [1, 2])

    def test_refuses_a_missing_replica(self):
        widget = self.band(1, 2, 3)
        widget.replica_rows[0].pk.value = 0
        with self.assertRaisesRegex(
            ValueError, "Intermediate 1 replica is not defined"
        ):
            widget.validate_replicas()

    def test_refuses_a_mismatch_and_names_both_rows(self):
        with self.assertRaisesRegex(ValueError, "Intermediate 1 vs Initial"):
            self.band(1, 9, 3).validate_replicas()


class InterpolateThroughWidgetTest(WidgetTestCase):
    """Clicking Interpolate, as opposed to the arithmetic it delegates to."""

    def setUp(self):
        super().setUp()
        self.structure(1, 0.0)
        self.structure(2, 10.0)

    def empty_row(self, widget):
        widget.initial_row.insert.click()
        return widget.replica_rows[0]

    def test_stores_a_structure_and_puts_its_pk_in_the_row(self):
        row = self.empty_row(self.band(1, 2))
        row.interpolate.click()
        self.assertEqual(len(self.stored), 1)
        self.assertEqual(row.pk.value, self.stored[0].pk)

    def test_the_factor_places_it_along_the_segment(self):
        row = self.empty_row(self.band(1, 2))
        row.factor.value = 0.25
        row.interpolate.click()
        self.assertAlmostEqual(self.stored[0].get_ase().positions[0][0], 2.5)

    def test_is_dead_until_both_neighbours_are_defined(self):
        widget = inputs.NebWidget()
        row = self.empty_row(widget)
        self.assertTrue(row.interpolate.disabled)
        self.assertIn("Initial and Last", row.interpolate.tooltip)
        widget.initial_row.pk.value = 1
        self.assertIn("Last still missing", row.interpolate.tooltip)
        widget.last_row.pk.value = 2
        self.assertFalse(row.interpolate.disabled)

    def test_the_controls_go_away_once_the_row_is_filled(self):
        row = self.empty_row(self.band(1, 2))
        self.assertEqual(row.interpolation_box.layout.display, "flex")
        row.interpolate.click()
        self.assertEqual(row.interpolation_box.layout.display, "none")


class RestartTest(WidgetTestCase):
    """Entering a PK replaces the band, so the builder goes away."""

    def setUp(self):
        super().setUp()
        self.initial = self.structure(160, 0.0)
        self.nodes[1234] = fake_neb(1234, self.initial, 10)
        self.nodes[5678] = fake_neb(5678, self.initial, 10, produced=0)

    def restarted(self, pk):
        widget = inputs.NebWidget()
        widget.restart_from.value = str(pk)
        return widget

    def test_a_finished_neb_hides_the_builder_and_sets_the_floor(self):
        widget = self.restarted(1234)
        self.assertEqual(widget.replica_box.layout.display, "none")
        self.assertEqual(widget.n_replica.min, 10)
        self.assertEqual(widget.n_replica.value, 10)

    def test_the_count_may_exceed_the_inherited_replicas(self):
        widget = self.restarted(1234)
        widget.n_replica.value = 15
        parameters = widget.return_dict()
        self.assertEqual(parameters["neb_params"]["number_of_replica"], 15)
        self.assertEqual(parameters["initial_uuid"], self.initial.uuid)
        self.assertNotIn("replica_uuids", parameters)

    def test_an_unfinished_neb_is_refused(self):
        widget = self.restarted(5678)
        self.assertIn("no optimised replicas", widget.restart_info.value)
        with self.assertRaisesRegex(ValueError, "no optimised replicas"):
            widget.return_dict()

    def test_a_pk_that_is_not_a_neb_is_refused(self):
        self.assertIn("not a NEB calculation", self.restarted(160).restart_info.value)

    def test_clearing_the_pk_brings_the_builder_back(self):
        widget = self.restarted(1234)
        widget.restart_from.value = ""
        self.assertEqual(widget.replica_box.layout.display, "flex")
        self.assertEqual(widget.n_replica.min, 2)

    def test_a_floor_that_went_stale_is_caught_at_submission(self):
        # The PK was entered while that calculation was still running, so the
        # floor stayed at the row count; nothing re-reads the node until submit.
        widget = self.restarted(5678)
        self.assertEqual(widget.n_replica.min, 2)
        self.nodes[5678] = fake_neb(5678, self.initial, 10)
        with self.assertRaisesRegex(ValueError, "below the 10 replicas inherited"):
            widget.return_dict()


class NebStateTest(WidgetTestCase):
    """InputDetails discards and rebuilds every section when the structure
    changes, so the replica setup has to survive round-tripping through it."""

    def setUp(self):
        super().setUp()
        self.structure(1, 0.0)
        self.structure(2, 4.0)
        self.structure(3, 8.0)

    def section(self, details):
        """What InputDetails._observe_details does: construct, then link."""
        section = inputs.NebWidget()
        for trait in section.traits_to_link():
            tr.link((details, trait), (section, trait))
        return section

    def test_the_setup_survives_a_rebuild(self):
        details = inputs.InputDetails()
        details.neb = True

        first = self.section(details)
        first.initial_row.pk.value = 1
        first.last_row.pk.value = 3
        first.initial_row.insert.click()
        first.replica_rows[0].pk.value = 2
        first.n_replica.value = 9
        first.k_spring.value = "0.07"
        first.optimize_endpoints.value = True

        second = self.section(details)
        self.assertEqual([row.pk.value for row in second.all_rows()], [1, 2, 3])
        self.assertEqual([row.name for row in second.replica_rows], ["Intermediate 1"])
        self.assertEqual(second.n_replica.value, 9)
        self.assertEqual(second.k_spring.value, "0.07")
        self.assertTrue(second.optimize_endpoints.value)


if __name__ == "__main__":
    unittest.main()
