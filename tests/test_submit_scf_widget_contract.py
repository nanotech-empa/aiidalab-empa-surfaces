import json
from pathlib import Path

from surfaces_tools.widgets.inputs import InputDetails

pytest_plugins = ["aiida.tools.pytest_fixtures"]


def test_submit_scf_uses_input_details_structure_manager():
    """Keep the SCF notebook aligned with the InputDetails widget contract."""
    notebook_path = Path(__file__).parents[1] / "submit_scf.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(
        cell_source
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for cell_source in cell["source"]
    )

    assert "structure_manager" in InputDetails.class_traits()
    assert "input_details.structure_manager = structure_selector" in source
    assert '(input_details, "structure_node")' not in source


def test_submit_scf_structure_selector_initializes(aiida_profile_tmp, monkeypatch):
    """Run the maintained startup cells through the real released widget APIs."""
    from aiida import engine
    from IPython.display import display

    def unexpected_submit(*args, **kwargs):
        raise AssertionError("Notebook initialization must not submit a calculation")

    monkeypatch.setattr(engine, "submit", unexpected_submit)
    notebook = json.loads(
        (Path(__file__).parents[1] / "submit_scf.ipynb").read_text(encoding="utf-8")
    )
    namespace = {"display": display}
    for index in (2, 3, 4):
        source = "".join(notebook["cells"][index]["source"])
        # The fixture already loaded a temporary brokerless SQLite profile.
        source = "\n".join(
            line for line in source.splitlines() if not line.startswith("%")
        )
        exec(compile(source, f"submit_scf.ipynb:cell{index}", "exec"), namespace)

    manager = namespace["structure_selector"]
    details = namespace["input_details"]
    assert details.structure_manager is manager
    assert details.structure is None
    assert manager.structure_node is None
