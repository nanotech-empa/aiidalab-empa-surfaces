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
