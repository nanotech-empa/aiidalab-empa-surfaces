import pytest

from surfaces_tools.utils.neb_resources import (
    NebResourceHelper,
    at_least,
    resource_layout,
)


def test_clamps_zero_and_negatives():
    assert at_least(0) == 1
    assert at_least(-7) == 1


def test_keeps_values_above_the_minimum():
    assert at_least(32) == 32
    assert at_least(2, minimum=8) == 8


def test_falls_back_on_unusable_values():
    assert at_least(None) == 1
    assert at_least("") == 1
    assert at_least(None, minimum=4) == 4


def test_nproc_replica_never_drops_below_one():
    # 15 replicas on 2 nodes x 1 task: fewer tasks than groups.
    layout = resource_layout(
        n_replica=15,
        n_replica_per_group=1,
        nodes=2,
        tasks_per_node=1,
        threads_per_task=1,
    )

    assert layout["n_groups"] == 15
    assert layout["nproc_replica"] == 1


def test_n_groups_rounds_up():
    layout = resource_layout(
        n_replica=7,
        n_replica_per_group=2,
        nodes=2,
        tasks_per_node=32,
        threads_per_task=4,
    )

    assert layout["n_groups"] == 4
    assert layout["nproc_replica"] == 16
    assert layout["tasks_per_group"] == 16.0


def test_clamps_zero_and_negative_allocation():
    layout = resource_layout(
        n_replica=4,
        n_replica_per_group=1,
        nodes=0,
        tasks_per_node=-32,
        threads_per_task=0,
    )

    assert layout["nodes"] == 1
    assert layout["tasks_per_node"] == 1
    assert layout["threads_per_task"] == 1
    assert layout["nproc_replica"] == 1


def test_whole_allocation_is_assigned_to_the_concurrent_groups():
    layout = resource_layout(
        n_replica=8,
        n_replica_per_group=2,
        nodes=4,
        tasks_per_node=32,
        threads_per_task=4,
    )

    # 4 groups run concurrently over 128 tasks; the 2 replicas in a group
    # share those 32 tasks in turn rather than splitting them.
    assert layout["n_groups"] == 4
    assert layout["nproc_replica"] == 32


def helper_showing(**layout_kwargs):
    """A panel whose provider always hands back one fixed layout."""
    return NebResourceHelper(lambda: resource_layout(**layout_kwargs))


def test_reports_the_grouping_and_nproc_rep():
    text = helper_showing(
        n_replica=15,
        n_replica_per_group=3,
        nodes=5,
        tasks_per_node=32,
        threads_per_task=4,
    ).describe_layout()

    assert "5 replica group(s)" in text
    assert "32 MPI tasks" in text
    assert "NPROC_REP = 32" in text


def test_stays_quiet_when_the_tasks_divide_evenly():
    # Half a node per group, but 16 whole tasks each: nothing is idle, and
    # this is a layout the resources widget produces on purpose.
    text = helper_showing(
        n_replica=8,
        n_replica_per_group=2,
        nodes=2,
        tasks_per_node=32,
        threads_per_task=4,
    ).describe_layout()

    assert "0.5 node(s)" in text
    assert "do not divide evenly" not in text


def test_warns_when_the_tasks_do_not_divide_evenly():
    # 96 tasks over 7 groups: 13 each, leaving 5 idle.
    text = helper_showing(
        n_replica=7,
        n_replica_per_group=1,
        nodes=3,
        tasks_per_node=32,
        threads_per_task=4,
    ).describe_layout()

    assert "do not divide evenly" in text
    assert "NPROC_REP = 13" in text


class _StatefulProvider:
    """A provider whose returned layout tracks call count and node count."""

    def __init__(self):
        self.calls = 0
        self.nodes = 5

    def __call__(self):
        self.calls += 1
        return resource_layout(
            n_replica=15,
            n_replica_per_group=3,
            nodes=self.nodes,
            tasks_per_node=32,
            threads_per_task=4,
        )


@pytest.fixture
def provider():
    return _StatefulProvider()


def test_starts_empty_without_consulting_the_provider(provider):
    helper = NebResourceHelper(provider)

    assert not helper.show.value
    assert helper.text.value == ""
    assert provider.calls == 0


def test_checking_the_box_renders_the_layout(provider):
    helper = NebResourceHelper(provider)

    helper.show.value = True

    assert "NPROC_REP = 32" in helper.text.value


def test_unchecking_the_box_clears_the_panel(provider):
    helper = NebResourceHelper(provider)
    helper.show.value = True

    helper.show.value = False

    assert helper.text.value == ""


def test_update_refreshes_while_shown_and_is_inert_while_hidden(provider):
    helper = NebResourceHelper(provider)
    helper.show.value = True

    # update() is registered as an observer on the node count and friends,
    # so it is called with a change dict it ignores.
    provider.nodes = 15
    helper.update({"name": "value", "new": 15})
    assert "NPROC_REP = 96" in helper.text.value

    helper.show.value = False
    calls_while_hidden = provider.calls
    helper.update()
    assert helper.text.value == ""
    assert provider.calls == calls_while_hidden
