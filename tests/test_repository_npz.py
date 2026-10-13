"""Exercise each notebook's archive read against an AiiDA repository stream."""

import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aiida import orm

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.mark.parametrize(
    "notebook,calculation,filename",
    [
        ("view_stm.ipynb", "stm_calc", "stm.npz"),
        ("view_orbitals.ipynb", "orb_calc", "orb.npz"),
    ],
)
def test_npz_read_from_repository_stream(
    aiida_profile, notebook, calculation, filename
):
    array = np.array([[1.0, 2.0], [3.0, 4.0]])
    stream = io.BytesIO()
    np.savez(stream, data=array)
    stream.seek(0)
    folder = orm.FolderData()
    folder.base.repository.put_object_from_filelike(stream, filename)
    folder.store()
    namespace = {
        "io": io,
        "np": np,
        calculation: SimpleNamespace(outputs=SimpleNamespace(retrieved=folder)),
    }
    path = Path(__file__).resolve().parents[1] / notebook
    cells = json.loads(path.read_text())["cells"]
    matches = []
    for cell in cells:
        source = "".join(cell["source"])
        if "loaded_data = np.load" not in source:
            continue
        tree = ast.parse(source)
        matches.extend(
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.With)
            and "loaded_data = np.load" in ast.get_source_segment(source, item)
        )
    assert len(matches) == 1
    exec(
        compile(ast.Module(body=matches, type_ignores=[]), str(path), "exec"), namespace
    )
    with namespace["loaded_data"] as loaded:
        np.testing.assert_array_equal(loaded["data"], array)
