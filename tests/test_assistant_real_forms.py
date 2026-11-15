"""Real scientific widgets, temporary storage, and no job submission."""

import ipywidgets as ipw
import numpy as np
import pytest
import traitlets
from aiida.tools.pytest_fixtures.configuration import (  # noqa: F401
    aiida_config_factory,
    aiida_profile_factory,
)

from surfaces_tools import assistant


@pytest.fixture(scope="module")
def profile(aiida_config_factory, aiida_profile_factory, tmp_path_factory):  # noqa: F811
    with aiida_config_factory(tmp_path_factory.mktemp("guided-forms")) as config:
        with aiida_profile_factory(config) as loaded:
            yield loaded


@pytest.fixture
def gui(profile, monkeypatch):
    import aiidalab_widgets_base as awb
    import aiidalab_widgets_empa as awe

    forms = pytest.importorskip("aiidalab_ai_assistant.forms")
    ToolRegistry = pytest.importorskip("aiidalab_ai_assistant.tools").ToolRegistry

    from surfaces_tools.widgets.dft_settings import DftSettingsWidget
    from surfaces_tools.widgets.inputs import InputDetails

    monkeypatch.setattr(forms, "panel", lambda adapter: adapter)
    selector = assistant.make_structure_manager(
        importers=[awb.StructureUploadWidget(), awe.CdxmlUploadWidget()],
        editors=[awb.BasicStructureEditor()],
        storable=False,
    )
    details = InputDetails()
    details.structure_manager = selector
    link = traitlets.dlink((selector, "structure"), (details, "structure"))
    dft = DftSettingsWidget()
    submit = ipw.Button(description="Submit test sentinel")
    called = []
    submit.on_click(lambda _: called.append(True))
    adapter = assistant.create_panel(
        "submit_geometry_optimization.ipynb",
        {
            "structure_selector": selector,
            "input_details": details,
            "dft_settings_widget": dft,
            "resources": awe.ProcessResourcesWidget(),
            "protocol": ipw.Dropdown(options=["standard", "debug"]),
            "btn_submit": submit,
            "secret": ipw.Password(value="do-not-expose"),
        },
    )
    registry = ToolRegistry()
    adapter.register(registry)
    yield selector, details, dft, adapter, registry, called
    link.unlink()
    selector.close()
    details.close()
    dft.close()


def call(registry, name, **args):
    args["expected_revision"] = registry.call("app_get_state", {})["revision"]
    result = registry.call(name, args)
    assert "error" not in result, result
    return result


def test_nested_physics_and_resources_are_real_visible_widgets(gui):
    selector, details, dft, adapter, registry, called = gui
    call(
        registry,
        "app_import_structure",
        filename="h2.xyz",
        content="2\nH2\nH 0 0 0\nH 0.74 0 0\n",
    )
    state = adapter.context()
    assert state["context"]["formula"] == "H2"
    assert "do-not-expose" not in str(state)
    functional = next(
        name
        for name, item in state["controls"].items()
        if item["label"] == "Functional:"
    )
    call(registry, "app_update_draft", fields={functional: "PBE0"})
    assert dft.xc_functional.value == "PBE0"
    # The rebuilt HFX/basis settings and bounded resource values are present.
    assert any(
        "hfx" in name.lower() or "memory" in item["label"].lower()
        for name, item in adapter.context()["controls"].items()
    )
    assert any(name.startswith("resources.") for name in state["fields"])
    assert any(name.startswith("input_details.") for name in state["fields"])
    call(registry, "app_select_atoms", indices=[2])
    assert selector.viewer.selection == [1]
    assert not called
    call(registry, "app_review_action", action="btn_submit")
    pending = call(registry, "app_execute_action", action="btn_submit")
    assert not called
    # A changed geometry must invalidate the old submission review.
    call(registry, "app_import_structure", filename="h.xyz", content="1\nH\nH 0 0 0\n")
    assert "error" in registry.approve(pending["action_id"])
    assert not called


def test_cdxml_and_database_load_use_the_existing_structure_form(gui):
    from aiida import orm

    selector, details, dft, adapter, registry, called = gui
    content = '<CDXML><page><fragment><n id="1" p="0 0" Element="6"/><n id="2" p="10 0" Element="6"/><b B="1" E="2" Order="2"/></fragment></page></CDXML>'
    call(registry, "app_import_structure", filename="drawing.cdxml", content=content)
    call(registry, "app_cdxml_action", action="create_model")
    atoms = selector.structure.copy()
    assert list(atoms.symbols).count("C") == 2
    node = orm.StructureData(ase=atoms).store()
    node.label = "guided geometry test"
    uuid = node.uuid
    del node
    found = registry.call("app_search_structures", {"query": "guided geometry test"})
    assert found["items"][0]["uuid"] == uuid
    call(registry, "app_load_structure", identifier=uuid)
    np.testing.assert_allclose(selector.structure.positions, atoms.positions)
    assert not called
