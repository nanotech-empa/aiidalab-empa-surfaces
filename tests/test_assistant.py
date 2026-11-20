import pytest

from types import SimpleNamespace

import ipywidgets as ipw

from surfaces_tools.assistant import create_panel


def test_bindings_edit_the_real_code_selector_and_ignore_unrelated_state(monkeypatch):
    forms = pytest.importorskip("aiidalab_ai_assistant.forms")

    monkeypatch.setattr(forms, "panel", lambda adapter: adapter)
    dropdown = ipw.Dropdown(options=[("None", None), ("Configured code", "code-uuid")])
    namespace = {
        "code_input_widget": SimpleNamespace(value=None, code_select_dropdown=dropdown),
        "protocol": ipw.Dropdown(options=["standard", "debug"]),
        "credentials": ipw.Password(value="never-in-context"),
        "node": object(),
    }
    adapter = create_panel("submit.ipynb", namespace)
    state = adapter.context()
    assert "never-in-context" not in str(state)
    assert state["controls"]["code_input_widget"]["options"] == [None, "code-uuid"]
    adapter.update(state["revision"], {"code_input_widget": "code-uuid"})
    assert dropdown.value == "code-uuid"
    assert not adapter._actions


def test_chat_observes_apply_and_undo_without_changing_the_editor(monkeypatch):
    from ase import Atoms
    from surfaces_tools.widgets.electrostatics import PoissonSettingsWidget

    forms = pytest.importorskip("aiidalab_ai_assistant.forms")
    monkeypatch.setattr(forms, "panel", lambda adapter: adapter)
    published = []
    widget = PoissonSettingsWidget(published.append)
    widget.set_structure(
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[10, 10, 10], pbc=False),
        {"system_type": "Molecule"},
    )
    details = ipw.VBox([widget])
    details.electrostatics = widget
    adapter = create_panel(
        "submit_geometry_optimization.ipynb", {"input_details": details}
    )
    state = adapter.context()
    boundary = state["context"]["electrostatics"]
    assert boundary["effective"] == {"periodic": "XYZ", "poisson_solver": "PERIODIC"}
    assert boundary["proposal"] == {"periodic": "NONE", "poisson_solver": "MT"}
    assert not boundary["apply_selected"] and not published

    adapter.update(state["revision"], {"electrostatics.apply_selected": True})
    assert (
        adapter.context()["context"]["electrostatics"]["effective"]
        == boundary["proposal"]
    )
    assert (
        widget.return_dict()["dft_params"]["periodic"] == "NONE" and len(published) == 1
    )
    widget.extra.value = "invalid JSON"
    assert adapter.context()["context"]["electrostatics"]["status"] == "blocked"
    widget.active.value = False
    assert (
        adapter.context()["context"]["electrostatics"]["effective"]
        == boundary["effective"]
    )
