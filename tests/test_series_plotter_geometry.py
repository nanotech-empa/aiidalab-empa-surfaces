import matplotlib.pyplot as plt
import numpy as np

from surfaces_tools.widgets.series_plotter import BOHR_TO_ANG, _grid_geometry, make_plot


def test_skewed_grid_retains_physical_origin_and_corners():
    info = {
        "x_arr": np.arange(4),
        "y_arr": np.arange(3),
        "origin": [3, 2, 0],
        "dv_vectors": [[1, 0.3, 0], [0.2, 1, 0], [0, 0, 1]],
    }
    extent, _, geometry = _grid_geometry(info, (4, 3))
    expected = np.array([[3, 2], [7, 3.2], [7.6, 6.2], [3.6, 5]]) * BOHR_TO_ANG
    np.testing.assert_allclose(geometry["corners"], expected)
    fig, ax = plt.subplots()
    try:
        image = np.arange(12).reshape((4, 3))
        make_plot(fig, ax, image, extent, grid_geometry=geometry, noadd=True)
        np.testing.assert_array_equal(ax.images[0].get_array(), image.T)
        display_corners = ax.images[0].get_transform().transform([[0, 0], [4, 3]])
        physical_corners = ax.transData.inverted().transform(display_corners)
        np.testing.assert_allclose(physical_corners, expected[[0, 2]])
    finally:
        plt.close(fig)


def test_legacy_rectangular_grid_keeps_extent():
    info = {"x_arr": np.array([0, 2]), "y_arr": np.array([0, 3])}
    extent, ratio, geometry = _grid_geometry(info, (2, 2))
    np.testing.assert_allclose(extent, np.array([0, 2, 0, 3]) * BOHR_TO_ANG)
    assert np.isclose(ratio, 2 / 3)
    assert geometry is None
