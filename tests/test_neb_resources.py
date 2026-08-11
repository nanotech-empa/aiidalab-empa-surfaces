import unittest

from surfaces_tools.utils.neb_resources import (
    NebResourceHelper,
    at_least,
    resource_layout,
)


class AtLeastTest(unittest.TestCase):
    def test_clamps_zero_and_negatives(self):
        self.assertEqual(at_least(0), 1)
        self.assertEqual(at_least(-7), 1)

    def test_keeps_values_above_the_minimum(self):
        self.assertEqual(at_least(32), 32)
        self.assertEqual(at_least(2, minimum=8), 8)

    def test_falls_back_on_unusable_values(self):
        self.assertEqual(at_least(None), 1)
        self.assertEqual(at_least(""), 1)
        self.assertEqual(at_least(None, minimum=4), 4)


class ResourceLayoutTest(unittest.TestCase):
    def test_nproc_replica_never_drops_below_one(self):
        # 15 replicas on 2 nodes x 1 task: fewer tasks than groups.
        layout = resource_layout(
            n_replica=15,
            n_replica_per_group=1,
            nodes=2,
            tasks_per_node=1,
            threads_per_task=1,
        )

        self.assertEqual(layout["n_groups"], 15)
        self.assertEqual(layout["nproc_replica"], 1)

    def test_n_groups_rounds_up(self):
        layout = resource_layout(
            n_replica=7,
            n_replica_per_group=2,
            nodes=2,
            tasks_per_node=32,
            threads_per_task=4,
        )

        self.assertEqual(layout["n_groups"], 4)
        self.assertEqual(layout["nproc_replica"], 16)
        self.assertEqual(layout["tasks_per_group"], 16.0)

    def test_clamps_zero_and_negative_allocation(self):
        layout = resource_layout(
            n_replica=4,
            n_replica_per_group=1,
            nodes=0,
            tasks_per_node=-32,
            threads_per_task=0,
        )

        self.assertEqual(layout["nodes"], 1)
        self.assertEqual(layout["tasks_per_node"], 1)
        self.assertEqual(layout["threads_per_task"], 1)
        self.assertEqual(layout["nproc_replica"], 1)

    def test_whole_allocation_is_assigned_to_the_concurrent_groups(self):
        layout = resource_layout(
            n_replica=8,
            n_replica_per_group=2,
            nodes=4,
            tasks_per_node=32,
            threads_per_task=4,
        )

        # 4 groups run concurrently over 128 tasks; the 2 replicas in a group
        # share those 32 tasks in turn rather than splitting them.
        self.assertEqual(layout["n_groups"], 4)
        self.assertEqual(layout["nproc_replica"], 32)


def helper_showing(**layout_kwargs):
    """A panel whose provider always hands back one fixed layout."""
    return NebResourceHelper(lambda: resource_layout(**layout_kwargs))


class DescribeLayoutTest(unittest.TestCase):
    def test_reports_the_grouping_and_nproc_rep(self):
        text = helper_showing(
            n_replica=15,
            n_replica_per_group=3,
            nodes=5,
            tasks_per_node=32,
            threads_per_task=4,
        ).describe_layout()

        self.assertIn("5 replica group(s)", text)
        self.assertIn("32 MPI tasks", text)
        self.assertIn("NPROC_REP = 32", text)

    def test_stays_quiet_when_the_tasks_divide_evenly(self):
        # Half a node per group, but 16 whole tasks each: nothing is idle, and
        # this is a layout the resources widget produces on purpose.
        text = helper_showing(
            n_replica=8,
            n_replica_per_group=2,
            nodes=2,
            tasks_per_node=32,
            threads_per_task=4,
        ).describe_layout()

        self.assertIn("0.5 node(s)", text)
        self.assertNotIn("do not divide evenly", text)

    def test_warns_when_the_tasks_do_not_divide_evenly(self):
        # 96 tasks over 7 groups: 13 each, leaving 5 idle.
        text = helper_showing(
            n_replica=7,
            n_replica_per_group=1,
            nodes=3,
            tasks_per_node=32,
            threads_per_task=4,
        ).describe_layout()

        self.assertIn("do not divide evenly", text)
        self.assertIn("NPROC_REP = 13", text)


class NebResourceHelperTest(unittest.TestCase):
    def setUp(self):
        self.calls = 0
        self.nodes = 5

    def provider(self):
        self.calls += 1
        return resource_layout(
            n_replica=15,
            n_replica_per_group=3,
            nodes=self.nodes,
            tasks_per_node=32,
            threads_per_task=4,
        )

    def test_starts_empty_without_consulting_the_provider(self):
        helper = NebResourceHelper(self.provider)

        self.assertFalse(helper.show.value)
        self.assertEqual(helper.text.value, "")
        self.assertEqual(self.calls, 0)

    def test_checking_the_box_renders_the_layout(self):
        helper = NebResourceHelper(self.provider)

        helper.show.value = True

        self.assertIn("NPROC_REP = 32", helper.text.value)

    def test_unchecking_the_box_clears_the_panel(self):
        helper = NebResourceHelper(self.provider)
        helper.show.value = True

        helper.show.value = False

        self.assertEqual(helper.text.value, "")

    def test_update_refreshes_while_shown_and_is_inert_while_hidden(self):
        helper = NebResourceHelper(self.provider)
        helper.show.value = True

        # update() is registered as an observer on the node count and friends,
        # so it is called with a change dict it ignores.
        self.nodes = 15
        helper.update({"name": "value", "new": 15})
        self.assertIn("NPROC_REP = 96", helper.text.value)

        helper.show.value = False
        calls_while_hidden = self.calls
        helper.update()
        self.assertEqual(helper.text.value, "")
        self.assertEqual(self.calls, calls_while_hidden)
