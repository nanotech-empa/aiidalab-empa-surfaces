from surfaces_tools.widgets.constraints import ConstraintsWidget


def test_single_slab_layer_default_constraint():
    widget = ConstraintsWidget()

    widget.details = {
        "system_type": "SlabXY",
        "bottom_H": [],
        "slab_layers": [[0, 1, 2, 3]],
    }

    assert widget.constraints.children[0].constraint_widget.value == "fixed xyz 1..4"
