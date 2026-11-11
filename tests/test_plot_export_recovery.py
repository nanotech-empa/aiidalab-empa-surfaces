"""Real widget/export callbacks and exact notebook formatter; no native science."""

import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
from surfaces_tools.helpers import HART_2_EV

ROOT = Path(__file__).parent


def module(version):
    from surfaces_tools.widgets import series_plotter

    return series_plotter


def plotter(version="candidate"):
    m = module(version)
    selection = [0]
    p = m.SeriesPlotter(lambda: selection[:], "stm")
    p.fig_y = 1
    data = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5) / 1000
    general = {
        "energies": np.array([-2.6, -2.5, -2.4]),
        "x_arr": np.arange(5) / m.BOHR_TO_ANG * 0.15,
        "y_arr": np.arange(6) / m.BOHR_TO_ANG * 0.15,
    }
    info = [
        {"type": "const-height sts", "height": h, "fwhm": 0.1, "p_tip_ratio": 0.0}
        for h in (3.0, 4.0)
    ]
    p.add_series_collection(general, info, data)
    p.setup_added_collections(123)
    return m, p, selection, data


def packed(p):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        p.data_to_zip(z)
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as z:
        return {k: z.read(k) for k in z.namelist()}


def test_single_map_export_has_all_formats_correct_data_and_completed_progress():
    _, p, _, data = plotter()
    files = packed(p)
    assert len(files) == 3 and p.zip_progress.value == 1
    txt = next(v for k, v in files.items() if k.endswith(".txt"))
    np.testing.assert_array_equal(np.loadtxt(io.BytesIO(txt)), data[1, 0])
    assert b"xlim=(0.00, 0.60), ylim=(0.00, 0.75)" in txt
    assert any("e-2.60" in k for k in files)


def test_multiple_maps_preserve_numeric_planes_and_labels():
    _, p, selection, data = plotter()
    selection[:] = [0, 1]
    p.add_selection_row()
    p.elem_list[1][0].value = sorted(p.series)[0]
    files = packed(p)
    assert len(files) == 12 and p.zip_progress.value == 1
    texts = [
        np.loadtxt(io.BytesIO(value))
        for name, value in files.items()
        if name.endswith(".txt")
    ]
    expected = [data[series, index] for series in (0, 1) for index in (0, 1)]
    for value in expected:
        assert any(np.array_equal(value, plane) for plane in texts)


def test_changed_selection_plot_clear_then_second_callback_export(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    m, p, selection, _ = plotter()
    links = []
    monkeypatch.setattr(m, "display", lambda content: links.append(content))
    p.create_zip_link(None)
    with zipfile.ZipFile(tmp_path / "tmp/stm_pk123.zip") as z:
        first = set(z.namelist())
    assert len(first) == 3 and not p.zip_btn.disabled and len(links) == 1
    selection[:] = [1, 2]
    p.elem_list[0][0].value = sorted(p.series)[0]
    p.plot_series(None)
    p.full_clear(None)
    p.create_zip_link(None)
    with zipfile.ZipFile(tmp_path / "tmp/stm_pk123.zip") as z:
        second = set(z.namelist())
    assert len(second) == 6 and first.isdisjoint(second)
    assert not p.zip_btn.disabled and p.zip_progress.value == 1 and len(links) >= 2
    plt.close("all")


@pytest.mark.parametrize("failure", ["generate", "write"])
def test_error_reenables_button_clears_old_link_and_allows_retry(
    tmp_path, monkeypatch, failure
):
    monkeypatch.chdir(tmp_path)
    m, p, _, _ = plotter()
    cleared = []
    shown = []
    monkeypatch.setattr(
        p.link_out, "clear_output", lambda *a, **kw: cleared.append(True)
    )
    monkeypatch.setattr(m, "display", lambda content: shown.append(content))
    p.create_zip_link(None)
    assert len(shown) == 1
    old_bytes = (tmp_path / "tmp/stm_pk123.zip").read_bytes()
    with monkeypatch.context() as change:

        def fail(*a, **kw):
            raise OSError("synthetic export failure")

        if failure == "generate":
            change.setattr(p, "data_to_zip", fail)
        else:
            change.setattr(m, "open", fail, raising=False)
        with pytest.raises(OSError, match="synthetic"):
            p.create_zip_link(None)
    assert not p.zip_btn.disabled and len(cleared) == 2 and len(shown) == 1
    assert (tmp_path / "tmp/stm_pk123.zip").read_bytes() == old_bytes
    p.create_zip_link(None)
    assert not p.zip_btn.disabled and len(shown) == 2 and len(cleared) == 3


def formatter(version):
    nb = json.loads((ROOT.parent / "view_adsorption_energy.ipynb").read_text())
    method = next(
        n
        for c in nb["cells"]
        if "def print_results" in "".join(c.get("source", []))
        for cls in ast.parse("".join(c["source"])).body
        if isinstance(cls, ast.ClassDef)
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "print_results"
    )
    env = {"HART_2_EV": HART_2_EV}
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]),
            "<actual adsorption formatter>",
            "exec",
        ),
        env,
    )
    return env["print_results"]


@pytest.mark.parametrize(
    "energy", [-0.00051, -0.00049, -0.0001, 0.0, 0.0001, 0.00049, 0.00051, -0.123456]
)
def test_weak_binding_sign_and_precision_convert_before_formatting(energy):
    values = {
        "unrelaxed_separation_energy": energy,
        "separation_energy": 2 * energy,
        "sheet": {"unrelaxed": -5.0, "relaxed": -5.0001},
    }
    p = NS(node=NS(outputs=NS(energies=values)), fragment_labels=["sheet"])
    actual = formatter("candidate")(p)
    expected = [
        f"Adsorption energy (frozen): {energy * HART_2_EV:.3f} (eV)",
        f"Adsorption energy: {2 * energy * HART_2_EV:.3f} (eV)",
    ]
    assert actual.splitlines()[:2] == expected
