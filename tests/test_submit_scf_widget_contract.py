import json
from pathlib import Path

from surfaces_tools.widgets.inputs import InputDetails


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
