import base64
import io
import math

import matplotlib.pyplot as plt
import numpy as np
from ase.visualize.plot import plot_atoms


def ase_to_thumbnail(structure, file_format=None):
    """Prepare binary information."""

    bytes_io = io.BytesIO()
    file_format = file_format if file_format else "png"
    cell = structure.cell.array
    x_extent = np.max(np.dot(cell, [1, 0, 0])) - np.min(np.dot(cell, [1, 0, 0]))
    y_extent = np.max(np.dot(cell, [0, 1, 0])) - np.min(np.dot(cell, [0, 1, 0]))

    lx = 5
    ly = lx * y_extent / x_extent
    if math.isnan(ly):
        ly = 5
    fig, ax = plt.subplots(figsize=(lx, ly))
    plot_atoms(structure, ax=ax)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        bytes_io, dpi=75, bbox_inches="tight", format=file_format, transparent=True
    )
    plt.close()
    return base64.b64encode(bytes_io.getvalue()).decode()


def display_thumbnail(thumbnail, width=400):
    return f'<img width="{width}px" src="data:image/png;base64,{thumbnail}" title="">'


def thumbnail_raw(
    nrows=1, thumbnail=None, pk=None, uuid=None, description="", tclass="tg-dark"
):
    """Returns an image with a link to structure export."""
    html = f'<td class="{tclass}" rowspan={nrows}><a target="_blank" href="./export_structure.ipynb?uuid={uuid}">'
    html += f'<img width="100px" src="data:image/png;base64,{thumbnail}" title="{description}: PK-{pk}">'
    html += "</a></td>"
    return html
