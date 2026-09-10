import base64
import re

import matplotlib.pyplot as plt

from surfaces_tools.widgets.pdos import make_image_link


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
