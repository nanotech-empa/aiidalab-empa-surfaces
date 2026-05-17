import unittest

from surfaces_tools.widgets.constraints import ConstraintsWidget


class ConstraintsWidgetTest(unittest.TestCase):
    def test_single_slab_layer_default_constraint(self):
        widget = ConstraintsWidget()

        widget.details = {
            "system_type": "SlabXY",
            "bottom_H": [],
            "slab_layers": [[0, 1, 2, 3]],
        }

        self.assertEqual(
            widget.constraints.children[0].constraint_widget.value,
            "fixed xyz 1..4",
        )
