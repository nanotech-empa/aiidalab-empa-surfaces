"""Interactive marker state and physical curve selection reach the export review."""

import numpy as np
import pytest

from surfaces_tools.widgets.series_plotter import SeriesPlotter
from surfaces_tools.widgets.stm_export import StmMicroscopyExport


@pytest.fixture
def controller(monkeypatch):
    export = pytest.importorskip("aiida_openbis.export")
    from aiida_openbis.nanonis import BOHR_ANGSTROM

    plotter = SeriesPlotter(lambda: [0], "stm")
    x, y = np.meshgrid(np.arange(8) * 0.5, np.arange(6) * 0.5, indexing="ij")
    general = {
        "energies": [-0.5, 0, 0.5],
        "x_arr": np.arange(9) * 0.5 / BOHR_ANGSTROM,
        "y_arr": np.arange(7) * 0.5 / BOHR_ANGSTROM,
    }
    series = [
        {"type": "const-height stm", "height": 4, "fwhm": 0.1},
        {"type": "const-isovalue stm", "isovalue": 1e-6, "fwhm": 0.1},
        {"type": "const-height sts", "height": 4, "fwhm": 0.1},
    ]
    values = np.array(
        [
            [2 * x + 3 * y + 4 * e + (5 if i == 1 else 0) for e in general["energies"]]
            for i in range(3)
        ]
    )
    plotter.add_series_collection(general, series, values, source_uuid="source")
    plotter.setup_added_collections(1)
    geometry = {
        "source_uuid": "source",
        "periodic_setting": "XY",
        "periodic_axes": [True, True],
        "in_plane_vectors_angstrom": [[4, 0], [0, 3]],
        "default_image_area": {
            "width_angstrom": 4,
            "height_angstrom": 3,
            "center_angstrom": [2, 1.5],
        },
    }
    monkeypatch.setattr(export, "describe_stm_imaging", lambda _: geometry)
    control = StmMicroscopyExport(plotter)
    control.set_source_uuid("source")
    yield control
    control.close()
    plotter.close()


def select(panel, part):
    label = next(
        label
        for label in panel.series.options
        if isinstance(label, tuple) and part in label[0]
    )[1]
    panel.series.value = label
    panel.load_button.click()
    assert panel._ready, panel.status.value


def test_point_spectra_keep_coordinates_colours_all_energies_and_remove_marker(
    controller,
):
    panel = controller.sampling
    select(panel, "ch-sts")
    panel.add_position([1, 1])
    panel.add_position([2, 1])
    assert len(panel.requests()[0]["points"]) == 2
    group = panel._groups[0]
    np.testing.assert_allclose(group["curves"][0]["y"], [3, 5, 7])
    np.testing.assert_allclose(group["curves"][1]["y"], [5, 7, 9])
    assert panel.map.marks[0]["color"] == group["curves"][0]["color"]
    assert group["curves"][0]["color"] != group["curves"][1]["color"]
    prior = panel.requests()
    panel.bias.value = 2
    assert (
        panel.requests() == prior
    )  # STS map display bias does not select one spectral point.
    controller.area_width.value = 8
    assert panel.requests() == prior  # Physical coordinates don't follow SXM crop.
    panel.selected.value = "P1"
    panel.remove_button.click()
    assert panel.requests()[0]["points"][0]["id"] == "P2"
    panel.clear_button.click()
    assert panel.requests() == []


def test_stm_line_updates_profile_with_bias_and_preserves_physical_path(controller):
    panel = controller.sampling
    select(panel, "cc-stm")
    panel.add_position([1, 1])
    with pytest.raises(ValueError, match="pending line"):
        panel.requests()
    assert not panel.context()["valid"]
    panel.add_position([2, 1])
    before = panel._groups[0]["curves"][0]
    assert panel.requests()[0]["kind"] == "stm_lines"
    assert panel.requests()[0]["energy_index"] == 0
    np.testing.assert_allclose(
        before["y"][[0, -1]]
        if isinstance(before["y"], np.ndarray)
        else np.array(before["y"])[[0, -1]],
        [8, 10],
    )
    panel.bias.value = 2
    after = panel._groups[0]["curves"][0]
    np.testing.assert_allclose(np.array(after["y"]) - before["y"], 4)
    np.testing.assert_array_equal(
        after["sample_positions_angstrom"], before["sample_positions_angstrom"]
    )
    assert panel.requests()[0]["energy_index"] == 2
    assert panel.map.marks[0]["color"] == after["color"]


def test_bad_line_cannot_silently_disappear_from_review_and_source_change_clears(
    controller,
):
    panel = controller.sampling
    select(panel, "ch-stm")
    panel.add_position([1, 1])
    panel.add_position([1, 1])
    assert "different endpoints" in panel.status.value
    assert panel.download_button.disabled
    with pytest.raises(ValueError, match="invalid sampling"):
        panel.requests()
    panel.clear_button.click()
    panel.add_position([1, 1])
    panel.add_position([2, 1])
    controller.review = {"entries": [], "curve_groups": []}
    panel.bias.value = 1
    assert controller.review is not None
    controller.keep_button.click()
    assert controller.review is None
    panel.configure(controller.source_geometry)
    assert panel.requests() == [] and panel.series.value == ""
