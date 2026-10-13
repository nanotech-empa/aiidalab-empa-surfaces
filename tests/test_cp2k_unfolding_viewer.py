"""Run the standalone notebook on a retrieved one- or two-spin dataset."""

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from aiida import orm
from aiida.common.links import LinkType

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.mark.parametrize("projections", [False, True])
@pytest.mark.parametrize("spins", [1, 2])
def test_standalone_unfolding_viewer(
    aiida_profile, aiida_localhost, monkeypatch, spins, projections
):
    dataset = {
        "dim": 1,
        "primitive_vectors": [[2.0, 0, 0]],
        "supercell_vectors": [[4.0, 0, 0]],
        "supercell_matrix": [[2]],
        "k_frac_folded": [[0.0], [0.5]],
        "path_k_indices": [0, 1],
        "path_x": [0, np.pi / 2],
        "path_q_equiv": [[0.0], [0.5]],
        "path_labels": ["G", "X"],
        "x_ticks": [0, np.pi / 2],
        "x_tick_labels": ["G", "X"],
        "hs_point_G": [0.0],
        "hs_point_X": [0.5],
        "ref_energy_ev": 0.5,
    }
    for spin in range(spins):
        dataset[f"evals_ev_spin_{spin}"] = [-0.5 - spin, 1.5 + spin]
        dataset[f"weights_spin_{spin}"] = [[0.2, 0.3], [0.4, 0.5]]
    stream = io.BytesIO()
    np.savez(stream, **dataset)
    stream.seek(0)
    calculation = orm.CalcJobNode(computer=aiida_localhost).store()
    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_filelike(stream, "unfolding_bands.npz")
    if projections:
        projection_stream = io.BytesIO()
        np.savez(
            projection_stream,
            spin=[0, 0],
            atom_index=[0, 1],
            channel_index=[0, 1],
            mo_index=[0, 1],
            projection=[0.5, 0.8],
            channels=["s", "pz"],
        )
        projection_stream.seek(0)
        retrieved.base.repository.put_object_from_filelike(
            projection_stream, "unfolding_projections.npz"
        )
    retrieved.base.links.add_incoming(
        calculation, link_type=LinkType.CREATE, link_label="retrieved"
    )
    retrieved.store()
    workchain = orm.WorkChainNode().store()
    retrieved.base.links.add_incoming(
        workchain, link_type=LinkType.RETURN, link_label="unfolding_retrieved"
    )
    figures = []
    monkeypatch.setattr(plt, "show", lambda: figures.append(plt.gcf()))
    path = Path(__file__).resolve().parents[1] / "view_cp2k_unfolding.ipynb"
    namespace = {
        "jupyter_notebook_url": f"http://localhost/view_cp2k_unfolding.ipynb?pk={workchain.pk}"
    }
    try:
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "\n".join(
                line for line in cell["source"] if not line.lstrip().startswith("%")
            )
            exec(compile(source, str(path), "exec"), namespace)
        axes = figures[-1].axes
        assert len(axes) == spins
        for spin, axis in enumerate(axes):
            assert axis.get_title().endswith(f"spin {spin}")
            points = axis.collections[0].get_offsets()
            np.testing.assert_allclose(
                points[:, 1], np.asarray(dataset[f"evals_ev_spin_{spin}"]) - 0.5
            )
            np.testing.assert_allclose(
                axis.collections[1].get_offsets()[:, 0], np.pi / 2
            )
        if projections:
            np.testing.assert_allclose(
                namespace["projection_weights_by_mo"](
                    0, atom_indices=[0], channels=["s"]
                ),
                [0.5, 0.0],
            )
            np.testing.assert_allclose(
                namespace["projection_weights_by_mo"](
                    0, atom_indices=[1], channels=["p"]
                ),
                [0.0, 0.8],
            )
        assert "workcalc" not in namespace
        assert namespace["calculation_info"]["uuid"] == workchain.uuid
    finally:
        for value in namespace.values():
            close = getattr(value, "close", None)
            if callable(close) and value.__class__.__module__.startswith("ipywidgets"):
                close()
        for figure in figures:
            plt.close(figure)
