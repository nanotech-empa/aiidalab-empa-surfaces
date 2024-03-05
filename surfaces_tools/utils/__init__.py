import base64
import io


def ase_to_thumbnail(structure, file_format=None):
    """Prepare binary information."""

    bytes_io = io.BytesIO()
    file_format = file_format if file_format else "png"
    structure.write(bytes_io, format=file_format)
    return base64.b64encode(bytes_io.getvalue()).decode()


def display_thumbnail(thumbnail, width=400):
    return f'<img width="{width}px" src="data:image/png;base64,{thumbnail}" title="">'


def thumbnail_raw(
    nrows=1, thumbnail=None, pk=None, uuid=None, description="", tclass="tg-dark"
):
    """Returns an image with a link to structure export."""
    html = f'<td class="{tclass}" rowspan={nrows}><a target="_blank" href="./export_structure.ipynb?uuid={uuid}">'
    html += f'<img width="100px" src="data:image/png;base64,{thumbnail}" title="PK{pk}: {description}">'
    html += "</a></td>"
    return html
