from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from surfaces_tools.widgets.pdos import PdosOverlapViewerWidget


@pytest.fixture
def viewer():
    widget = PdosOverlapViewerWidget()
    yield widget
    seen = set()

    def close(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        for child in getattr(value, "children", ()):
            close(child)
        value.close()

    close(widget)


@pytest.mark.parametrize(
    "low,high,expected",
    [("", "", [None, None]), ("-2", "", [-2, None]), ("", "5", [None, 5])],
)
def test_optional_density_limits(viewer, low, high, expected):
    viewer._density_min.value, viewer._density_max.value = low, high
    assert viewer._get_density_range() == expected


@pytest.mark.parametrize("low,high", [("x", ""), ("nan", ""), ("", "inf"), ("3", "2")])
def test_invalid_density_limits_do_not_open_figures(viewer, low, high, capsys):
    viewer._density_min.value, viewer._density_max.value = low, high
    before = plt.get_fignums()
    viewer.make_plot()
    assert "Y min" in capsys.readouterr().out
    assert plt.get_fignums() == before


def test_projection_styles_preserve_exported_spin_densities(viewer, monkeypatch):
    viewer._energy_range_slider.min = -2
    viewer._energy_range_slider.max = 2
    viewer._energy_range_slider.value = (-2, 2)
    viewer._density_min.value, viewer._density_max.value = "-3", "4"
    viewer.uks, viewer.do_overlap = True, True
    energies = np.array([-0.5, 0.5])
    weights = [np.array([0.2, 0.4]), np.array([0.3, 0.1])]
    viewer._overlap = SimpleNamespace(
        data={"energies_g1": [energies, energies]},
        items=[
            SimpleNamespace(return_all_values=lambda: ("up", "red", 1, 0, weights[0])),
            SimpleNamespace(
                return_all_values=lambda: ("down", "blue", 1, 1, weights[1])
            ),
        ],
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    figures, exported = [], []
    try:
        assert viewer._orbital_projection_style.value == "filled"
        for style in ("filled", "line"):
            viewer._orbital_projection_style.value = style
            fig, (headers, data) = viewer._create_the_plot()
            figures.append(fig)
            exported.append(data)
            assert headers == ["energy [eV]", "up s0", "down s1"]
            np.testing.assert_allclose(fig.axes[0].get_ylim(), [-3, 4])
        assert len(figures[0].axes[0].collections) == 2
        assert len(figures[1].axes[0].collections) == 0
        np.testing.assert_array_equal(exported[0], exported[1])
        for spin, line in enumerate(figures[1].axes[0].lines[:2]):
            np.testing.assert_allclose(
                line.get_ydata(), exported[1][:, spin + 1] * (-2 * spin + 1)
            )
    finally:
        for fig in figures:
            plt.close(fig)
