import base64
import re

import matplotlib.pyplot as plt
import numpy as np

from surfaces_tools.widgets.pdos import PdosOverlapViewerWidget, make_image_link


def test_make_svg_image_link_contains_vector_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0], label="PDOS")
    ax.legend()

    try:
        link = make_image_link(fig, text="SVG", data_format="svg")
    finally:
        plt.close(fig)

    assert 'download="pdos.svg"' in link
    assert ">SVG</a>" in link
    match = re.search(r'href="data:image/svg\+xml;name=pdos\.svg;base64,([^"]+)"', link)
    assert match is not None

    svg = base64.b64decode(match.group(1)).decode()
    assert "<svg" in svg
    assert "PDOS" in svg


def test_plot_projections_ignores_empty_selection():
    viewer = PdosOverlapViewerWidget()
    viewer._projections.add_item(None)
    energy = np.array([0.0, 1.0])
    collected_data = energy.reshape(1, -1)
    headers = ["energy [eV]"]
    fig, ax = plt.subplots()

    try:
        result_headers, result_data = viewer._plot_projections(
            ax, [None, None], energy, collected_data, headers
        )
    finally:
        plt.close(fig)

    assert result_headers == headers
    np.testing.assert_array_equal(result_data, collected_data)
