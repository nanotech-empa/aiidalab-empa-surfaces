"""The GUI/chat use stored selections and the same confirmed common exporter."""

import ipywidgets as ipw
import numpy as np
import pytest

from surfaces_tools.widgets.series_plotter import SeriesPlotter
from surfaces_tools.widgets.stm_export import StmMicroscopyExport


@pytest.fixture
def plotter():
    plotter = SeriesPlotter(lambda: [0, 2], "stm")
    general = {"energies": [-0.5, 0, 0.5], "x_arr": np.arange(5), "y_arr": np.arange(4)}
    plotter.add_series_collection(
        general,
        [{"type": "const-height stm", "height": 5.5, "fwhm": 0.1}],
        np.ones((1, 3, 4, 3)),
        source_uuid="source",
    )
    plotter.setup_added_collections(1)
    yield plotter
    plotter.close()


def test_selection_uses_original_indices_not_plot_normalization(plotter):
    plotter.elem_list[0][3].value = True
    assert plotter.microscopy_selection() == {
        "source_uuid": "source",
        "selections": [
            {"series_index": 0, "energy_index": 0},
            {"series_index": 0, "energy_index": 2},
        ],
    }


def test_plotting_multiple_sources_still_allowed_but_export_is_explicit(plotter):
    plotter.add_series_collection(
        {"x_arr": np.arange(5), "y_arr": np.arange(4)},
        [{"type": "const-height stm", "height": 6, "fwhm": 0.1}],
        np.ones((1, 3, 4, 3)),
        source_uuid="other-source",
    )
    plotter.add_selection_row()
    plotter.elem_list[1][0].value = "ch-stm fwhm=0.10 h=6.0"
    with pytest.raises(ValueError, match="one stored"):
        plotter.microscopy_selection()
    assert len(plotter.series) == 2


@pytest.fixture
def cell_geometry(monkeypatch):
    export = pytest.importorskip("aiida_openbis.export")
    geometry = {
        "source_uuid": "source",
        "periodic_setting": "XY",
        "periodic_axes": [True, True],
        "default_image_area": {
            "width_angstrom": 4.0,
            "height_angstrom": 3.0,
            "center_angstrom": [2.0, 1.5],
        },
    }
    monkeypatch.setattr(export, "describe_stm_imaging", lambda _: geometry)
    return geometry


def test_chat_prepares_same_widget_and_execution_requires_fresh_review(
    monkeypatch, plotter, cell_geometry
):
    forms = pytest.importorskip("aiidalab_ai_assistant.forms")
    export = pytest.importorskip("aiida_openbis.export")
    from surfaces_tools.assistant import create_panel

    monkeypatch.setattr(forms, "panel", lambda adapter: adapter)
    prepared, written = [], []
    monkeypatch.setattr(
        export,
        "review_stm_imaging",
        lambda **kwargs: {
            "source_uuid": kwargs["source_uuid"],
            "sampling": kwargs["sampling"],
            "curve_groups": [],
            "entries": kwargs["selections"],
            "image_area": kwargs["image_area"],
        },
    )

    def make(source_uuid, **kwargs):
        prepared.append((source_uuid, kwargs))
        button = ipw.Button(description="Export to openBIS")
        button.on_click(lambda _: written.append(source_uuid))
        widget = ipw.VBox([ipw.Text(description="Destination", value=""), button])
        widget.save_simulations_button = button
        return widget

    monkeypatch.setattr(export, "make_export_widget", make)
    controller = StmMicroscopyExport(plotter)
    adapter = create_panel(
        "view_stm.ipynb",
        {
            "series_plotter_inst": plotter,
            "stm_microscopy_export": controller,
            "credentials": ipw.Password(value="excluded"),
        },
    )
    state = adapter.context()
    assert "excluded" not in str(state)
    assert not state["execution_actions"]
    adapter.prepare_export(state["revision"])
    assert len(prepared) == 1 and not written
    publication_plan = {
        "mode": "append_dat",
        "spm_permid": "existing-spm",
        "collection": "/SPACE/PROJECT/ORIGINAL",
        "dataset_counts": {"maps": 0, "curve_datasets": 1, "extracted_curves": 2},
    }
    controller.export_widget.microscopy_publication_plan = lambda: publication_plan
    assert controller.context()["microscopy_publication_plan"] == publication_plan
    assert (
        controller.context()["microscopy_dataset_counts"]
        == publication_plan["dataset_counts"]
    )
    state = adapter.context()
    assert "export_stm_microscopy" in state["execution_actions"]
    reviewed = adapter.review("export_stm_microscopy", state["revision"])
    # A change to the destination invalidates the reviewed execution.
    controller.export_widget.children[0].value = "different destination"
    with pytest.raises(ValueError):
        adapter.execute("export_stm_microscopy", reviewed["state"]["revision"])
    assert not written
    controller.close()


def test_comparison_gallery_keeps_bias_columns_case_rows_and_horizontal_only_scroll(
    plotter, monkeypatch
):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        plotter.add_selection_row()
        plotter.plot_series(None)
        layout = plotter.plot_output.children[-1].layout
        assert layout.overflow == "auto hidden"
        assert layout.height == "auto"
        assert layout.flex_flow == "row nowrap"
        columns = plotter.plot_output.children[-1].children
        assert len(columns) == 2
        assert all(column.layout.flex == "0 0 auto" for column in columns)
        figures = [plt.figure(number) for number in plt.get_fignums()]
        assert len(figures) == 2  # Two selected biases.
        for figure in figures:
            assert len(figure.axes) == 2  # Both selected cases remain rows.
            assert all(
                axis.get_subplotspec().get_gridspec().nrows == 2 for axis in figure.axes
            )
    finally:
        plt.close("all")


def test_area_controls_defaults_reset_and_changes_invalidate_frozen_review(
    plotter, cell_geometry
):
    controller = StmMicroscopyExport(plotter)
    controller.set_source_uuid("source")
    assert controller.image_area() == cell_geometry["default_image_area"]
    assert "XY" in controller.area_info.value
    controller.area_width.value = 8
    controller.area_center_x.value = -2
    assert controller.image_area()["width_angstrom"] == 8
    controller.review = {"entries": [], "curve_groups": []}
    controller.export_widget = ipw.VBox()
    controller.area_height.value = 9
    assert controller.review is None and controller.export_widget is None
    controller.reset_area_button.click()
    assert controller.image_area() == cell_geometry["default_image_area"]
    controller.close()
