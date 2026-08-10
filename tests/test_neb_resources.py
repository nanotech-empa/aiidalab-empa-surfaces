import unittest

from surfaces_tools.utils.neb_resources import at_least, resource_layout


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
