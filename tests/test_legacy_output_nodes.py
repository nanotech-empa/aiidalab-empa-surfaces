"""Read legacy attributes without migrating or replacing scientific nodes."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiida import orm

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def readers():
    path = Path(__file__).resolve().parents[1] / "view_geometry_optimization.ipynb"
    namespace = {}
    for cell in json.loads(path.read_text())["cells"]:
        source = "".join(cell["source"])
        if not any(
            f"def {name}(" in source for name in ("_node_to_dict", "_input_dict")
        ):
            continue
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("node_type", [orm.Dict, orm.Data])
def test_current_and_legacy_nodes_preserve_output_and_spin_inputs(
    aiida_profile_clean, readers, node_type
):
    values = {
        "energy": -12.5,
        "bandgap_spin1_au": None,
        "motion_step_info": {"energy_eV": [-11.0, -12.5]},
        "magnetization_per_site": [1, 0, -1],
    }
    node = node_type()
    node.base.attributes.set_many(values)
    assert readers["_node_to_dict"](node) == values
    process = SimpleNamespace(inputs=SimpleNamespace(dft_params=node))
    assert readers["_input_dict"](process, "dft_params") == values
    assert readers["_input_dict"](process, "missing") == {}
    assert node.base.attributes.all == values


def test_missing_and_old_attribute_interfaces(readers):
    assert readers["_node_to_dict"](None) == {}
    assert readers["_node_to_dict"](SimpleNamespace(attributes={"energy": -2})) == {
        "energy": -2
    }
