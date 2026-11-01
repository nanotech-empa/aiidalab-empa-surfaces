import pytest
from aiida import orm

from surfaces_tools.utils.node_data import mapping_from_node

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.mark.parametrize("node_class", [orm.Dict, orm.Data])
def test_current_and_legacy_mapping_nodes(aiida_profile, node_class):
    values = {"energy": -1.5, "--eval_region": ["0", "z+5.0"], "nested": {"spin": 2}}
    node = node_class()
    node.base.attributes.set_many(values)
    node.store()
    assert mapping_from_node(node) == values
    assert node.base.attributes.all == values


def test_plain_mapping():
    value = {"energy": -1.5}
    assert mapping_from_node(value) == value
