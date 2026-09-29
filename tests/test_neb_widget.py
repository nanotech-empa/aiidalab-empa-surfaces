import types
from unittest import mock

import numpy as np
import pytest
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


def test_accepts_identical_compositions():
    ok, message = validate_replica_pair(water(), water())
    assert ok
    assert message == ""


def test_accepts_moved_atoms():
    moved = water()
    moved.positions[1] += [0.3, 0.0, 0.0]
    ok, _ = validate_replica_pair(water(), moved)
    assert ok


def test_rejects_an_atom_count_change():
    ok, message = validate_replica_pair(water(), water() + Atoms("H", [[3, 3, 3]]))
    assert not ok
    assert "Atom count changed: 3 -> 4" in message


def test_rejects_reordered_atoms_and_names_the_index():
    reordered = Atoms(
        "HOH",
        positions=water().positions,
        cell=water().cell,
        pbc=water().pbc,
    )
    ok, message = validate_replica_pair(water(), reordered)
    assert not ok
    assert "index 0" in message
    assert "O -> H" in message


def test_quiet_when_cell_and_pbc_match():
    assert compare_replica_cells(water(), water()) == ""


def test_flags_a_differing_cell():
    difference = compare_replica_cells(water(), water(cell=(10.0, 10.0, 12.0)))
    assert "cell differs" in difference
    assert "PBC differs" not in difference


def test_flags_a_differing_pbc():
    difference = compare_replica_cells(water(), water(pbc=(True, True, False)))
    assert "PBC differs" in difference
    assert "cell differs" not in difference


def widget_with(n_replica):
    """NebWidget with a given replica count.

    NebWidget builds without an AiiDA profile as long as no replica PK is
    set, which is what lets these run here. The divisor arithmetic for the
    "# rep / group" choices is inline in NebWidget, so there is nothing else
    to call directly.
    """
    widget = inputs.NebWidget()
    widget.n_replica.value = n_replica
    return widget


def test_offers_every_divisor_of_the_count():
    options = widget_with(15).n_replica_per_group.options
    assert list(options) == [1, 3, 5, 15]


def test_prime_count_offers_only_one_and_itself():
    options = widget_with(7).n_replica_per_group.options
    assert list(options) == [1, 7]


def test_keeps_a_selection_that_still_divides():
    widget = widget_with(15)
    widget.n_replica_per_group.value = 5
    widget.n_replica.value = 10
    assert widget.n_replica_per_group.value == 5


def test_resets_a_selection_that_no_longer_divides():
    widget = widget_with(15)
    widget.n_replica_per_group.value = 5
    widget.n_replica.value = 8
    assert widget.n_replica_per_group.value == 1


@pytest.fixture
def replica_endpoints():
    first = water()
    last = water()
    last.positions += [1.0, 2.0, 0.0]
    return first, last


@pytest.mark.parametrize("coefficient", [0.1, 0.9])
def test_interpolates_onto_the_segment(replica_endpoints, coefficient):
    first, last = replica_endpoints
    interpolated = interpolate_replicas(first, last, coefficient)
    expected = (1.0 - coefficient) * first.positions + coefficient * last.positions
    np.testing.assert_allclose(interpolated.positions, expected)


def test_preserves_atom_count_and_symbols(replica_endpoints):
    first, last = replica_endpoints
    interpolated = interpolate_replicas(first, last, 0.5)
    assert len(interpolated) == len(first)
    assert interpolated.get_chemical_symbols() == first.get_chemical_symbols()
    ok, _ = validate_replica_pair(first, interpolated)
    assert ok


def test_does_not_mutate_the_endpoints(replica_endpoints):
    first, last = replica_endpoints
    before = first.positions.copy()
    interpolate_replicas(first, last, 0.5)
    np.testing.assert_allclose(first.positions, before)


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


class _Harness:
    """Drives a NebWidget against stub nodes, so no AiiDA profile is needed.

    PKs resolve out of `self.nodes`, and whatever the widget stores lands in
    `self.stored`.
    """

    def __init__(self):
        self.nodes = {}
        self.stored = []

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


@pytest.fixture
def harness():
    """`inputs.orm` is replaced wholesale for the lifetime of the test.

    This couples the tests to `orm` being a module-level name in inputs.py -
    if it ever becomes a local import, the patch stops biting and these pass
    vacuously.
    """
    h = _Harness()

    class StoredStructure(FakeStructure):
        def __init__(self, ase):
            FakeStructure.__init__(self, 900 + len(h.stored), ase)
            h.stored.append(self)
            h.nodes[self.pk] = self

        def store(self):
            return self

    with mock.patch.object(
        inputs,
        "orm",
        types.SimpleNamespace(
            StructureData=StoredStructure,
            load_node=lambda pk: h.nodes[int(pk)],
        ),
    ):
        yield h


@pytest.fixture
def replicas_harness(harness):
    """The gate that refuses a broken band at submission time.

    Nothing else stops a submission: the coloured row text is display only.
    """
    harness.structure(1, 0.0)
    harness.structure(2, 4.0)
    harness.structure(3, 8.0)
    harness.nodes[9] = FakeStructure(9, water() + Atoms("H", [[3.0, 3.0, 3.0]]))
    return harness


def test_returns_every_replica_in_order(replicas_harness):
    nodes = replicas_harness.band(1, 2, 3).validate_replicas()
    assert [node.pk for node in nodes] == [1, 2, 3]


def test_accepts_a_band_of_just_the_endpoints(replicas_harness):
    nodes = replicas_harness.band(1, 2).validate_replicas()
    assert [node.pk for node in nodes] == [1, 2]


def test_refuses_a_missing_replica(replicas_harness):
    widget = replicas_harness.band(1, 2, 3)
    widget.replica_rows[0].pk.value = 0
    with pytest.raises(ValueError, match="Intermediate 1 replica is not defined"):
        widget.validate_replicas()


def test_refuses_a_mismatch_and_names_both_rows(replicas_harness):
    with pytest.raises(ValueError, match="Intermediate 1 vs Initial"):
        replicas_harness.band(1, 9, 3).validate_replicas()


@pytest.fixture
def interpolate_harness(harness):
    """Clicking Interpolate, as opposed to the arithmetic it delegates to."""
    harness.structure(1, 0.0)
    harness.structure(2, 10.0)
    return harness


def empty_row(widget):
    widget.initial_row.insert.click()
    return widget.replica_rows[0]


def test_stores_a_structure_and_puts_its_pk_in_the_row(interpolate_harness):
    row = empty_row(interpolate_harness.band(1, 2))
    row.interpolate.click()
    assert len(interpolate_harness.stored) == 1
    assert row.pk.value == interpolate_harness.stored[0].pk


def test_the_factor_places_it_along_the_segment(interpolate_harness):
    row = empty_row(interpolate_harness.band(1, 2))
    row.factor.value = 0.25
    row.interpolate.click()
    assert interpolate_harness.stored[0].get_ase().positions[0][0] == pytest.approx(2.5)


def test_is_dead_until_both_neighbours_are_defined(interpolate_harness):
    widget = inputs.NebWidget()
    row = empty_row(widget)
    assert row.interpolate.disabled
    assert "Initial and Last" in row.interpolate.tooltip
    widget.initial_row.pk.value = 1
    assert "Last still missing" in row.interpolate.tooltip
    widget.last_row.pk.value = 2
    assert not row.interpolate.disabled


def test_the_controls_go_away_once_the_row_is_filled(interpolate_harness):
    row = empty_row(interpolate_harness.band(1, 2))
    assert row.interpolation_box.layout.display == "flex"
    row.interpolate.click()
    assert row.interpolation_box.layout.display == "none"


@pytest.fixture
def restart_harness(harness):
    """Entering a PK replaces the band, so the builder goes away."""
    harness.initial = harness.structure(160, 0.0)
    harness.nodes[1234] = fake_neb(1234, harness.initial, 10)
    harness.nodes[5678] = fake_neb(5678, harness.initial, 10, produced=0)
    return harness


def restarted(pk):
    widget = inputs.NebWidget()
    widget.restart_from.value = str(pk)
    return widget


def test_a_finished_neb_hides_the_builder_and_sets_the_floor(restart_harness):
    widget = restarted(1234)
    assert widget.replica_box.layout.display == "none"
    assert widget.n_replica.min == 10
    assert widget.n_replica.value == 10


def test_the_count_may_exceed_the_inherited_replicas(restart_harness):
    widget = restarted(1234)
    widget.n_replica.value = 15
    parameters = widget.return_dict()
    assert parameters["neb_params"]["number_of_replica"] == 15
    assert parameters["initial_uuid"] == restart_harness.initial.uuid
    assert "replica_uuids" not in parameters


def test_an_unfinished_neb_is_refused(restart_harness):
    widget = restarted(5678)
    assert "no optimised replicas" in widget.restart_info.value
    with pytest.raises(ValueError, match="no optimised replicas"):
        widget.return_dict()


def test_a_pk_that_is_not_a_neb_is_refused(restart_harness):
    assert "not a NEB calculation" in restarted(160).restart_info.value


def test_clearing_the_pk_brings_the_builder_back(restart_harness):
    widget = restarted(1234)
    widget.restart_from.value = ""
    assert widget.replica_box.layout.display == "flex"
    assert widget.n_replica.min == 2


def test_a_floor_that_went_stale_is_caught_at_submission(restart_harness):
    # The PK was entered while that calculation was still running, so the
    # floor stayed at the row count; nothing re-reads the node until submit.
    widget = restarted(5678)
    assert widget.n_replica.min == 2
    restart_harness.nodes[5678] = fake_neb(5678, restart_harness.initial, 10)
    with pytest.raises(ValueError, match="below the 10 replicas inherited"):
        widget.return_dict()


def test_a_replica_that_cannot_be_read_is_reported_on_its_own_row(harness):
    """update_replica_info runs from traitlets observers, so it must not raise.

    An exception there escapes as a traceback in the notebook rather than
    reaching the form, which is how a non-structure PK used to crash it.
    """

    class Broken(FakeStructure):
        def get_ase(self):
            raise RuntimeError("cannot parse sites")

    harness.structure(1, 0.0)
    harness.structure(3, 8.0)
    harness.nodes[66] = Broken(66, None)

    widget = harness.band(1, 66, 3)

    assert "Cannot read this replica" in widget.replica_rows[0].info.value
    assert "cannot parse sites" in widget.replica_rows[0].info.value
    # The rest of the form still works.
    assert widget.n_replica.value == 3
    assert list(widget.n_replica_per_group.options) == [1, 3]


@pytest.fixture
def state_harness(harness):
    harness.structure(1, 0.0)
    harness.structure(2, 4.0)
    harness.structure(3, 8.0)
    return harness


def build_section(details):
    """What InputDetails._observe_details does: construct, then link."""
    section = inputs.NebWidget()
    for trait in section.traits_to_link():
        tr.link((details, trait), (section, trait))
    return section


def test_the_setup_survives_a_rebuild(state_harness):
    """InputDetails discards and rebuilds every section when the structure
    changes, so the replica setup has to survive round-tripping through it."""
    details = inputs.InputDetails()
    details.neb = True

    first = build_section(details)
    first.initial_row.pk.value = 1
    first.last_row.pk.value = 3
    first.initial_row.insert.click()
    first.replica_rows[0].pk.value = 2
    first.n_replica.value = 9
    first.k_spring.value = "0.07"
    first.optimize_endpoints.value = True

    second = build_section(details)
    assert [row.pk.value for row in second.all_rows()] == [1, 2, 3]
    assert [row.name for row in second.replica_rows] == ["Intermediate 1"]
    assert second.n_replica.value == 9
    assert second.k_spring.value == "0.07"
    assert second.optimize_endpoints.value
