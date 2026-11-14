import json
from pathlib import Path
from types import SimpleNamespace
from urllib import parse

import ipywidgets as ipw
import pytest


@pytest.mark.parametrize(
    "label,run_type",
    [
        ("CP2K_SCF", "ENERGY"),
        ("CP2K_SCF_OT", "ENERGY"),
        ("CP2K_GeoOpt", "GEO_OPT"),
        ("CP2K_CellOpt", "CELL_OPT"),
    ],
)
def test_heading_structure_and_summary_share_result_context(label, run_type):
    """Exercise the maintained notebook cells without scientific execution."""
    path = Path(__file__).parents[1] / "view_geometry_optimization.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = SimpleNamespace(pk=101)
    optimized = SimpleNamespace(pk=202)
    trajectory = object()
    motion = {
        key: [value]
        for key, value in {
            "cell_a_angs": 12,
            "cell_b_angs": 12,
            "cell_c_angs": 12,
            "cell_alp_deg": 90,
            "cell_bet_deg": 90,
            "cell_gam_deg": 90,
            "pressure_bar": 0,
        }.items()
    }
    parameters = {"energy": -1.0, "run_type": run_type, "motion_step_info": motion}
    outputs = {"output_parameters": SimpleNamespace(get_dict=lambda: parameters)}
    is_scf = label.startswith("CP2K_SCF")
    if not is_scf:
        outputs.update(output_structure=optimized, output_trajectory=trajectory)
    process = SimpleNamespace(
        label=label,
        inputs=SimpleNamespace(structure=source),
        outputs=SimpleNamespace(**outputs),
    )
    displayed = []
    namespace = {
        "urlparse": parse,
        "jupyter_notebook_url": "https://example.org/view_geometry_optimization.ipynb?pk=1",
        "orm": SimpleNamespace(load_node=lambda pk: process),
        "ipw": ipw,
        "display": displayed.append,
        "helpers": SimpleNamespace(HART_2_EV=27.2114),
        "format_bandgaps": lambda *args: "",
    }
    selected_cells = [
        (index, cell)
        for index, cell in enumerate(notebook["cells"])
        if any(
            marker in "".join(cell["source"])
            for marker in (
                "workcalc = orm.load_node(",
                "## Equilibrium geometry",
                'heading = "SCF structure"',
                "def _node_to_dict(",
            )
        )
    ]
    assert len(selected_cells) == 3
    for index, cell in selected_cells:
        text = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            # Render the original static heading too, so the regression fails
            # on the unchanged notebook rather than skipping its old title.
            displayed.append(ipw.HTML(value=f"<h2>{text.removeprefix('## ')}</h2>"))
        else:
            exec(compile(text, f"{path}:cell{index}", "exec"), namespace)

    title = "SCF structure" if is_scf else "Equilibrium geometry"
    assert [widget.value for widget in displayed] == [f"<h2>{title}</h2>"]
    selected = source if is_scf else optimized
    assert namespace["opt_structure"] is selected
    assert namespace["trajectory"] is (None if is_scf else trajectory)
    assert f"<b>{title} PK:</b> {selected.pk}" in namespace["html"]
    assert "<b>Energy:</b> -1.0" in namespace["html"]
