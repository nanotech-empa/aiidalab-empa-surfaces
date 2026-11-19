from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from surfaces_tools.widgets import afm


@pytest.mark.parametrize(
    ("viewer", "prefixes"),
    [(afm.ViewAfmLegacy, ("pp/pp", "pp2/pp2")), (afm.ViewAfmWidget, ("ppafm",))],
)
@pytest.mark.parametrize("height_count", [1, 2])
def test_export_afm_scan(tmp_path, monkeypatch, viewer, prefixes, height_count):
    params = SimpleNamespace(
        dict=SimpleNamespace(scanMin=[0, 0, 2], Amplitude=1, scanStep=[1, 1, 0.2])
    )
    work = SimpleNamespace(
        pk=41,
        inputs=SimpleNamespace(
            afm_pp_params=params, ppafm_params=params, structure=None
        ),
    )
    grid = np.meshgrid([0.0, 1.0], [0.0, 1.0])
    slices = np.arange(height_count * 4).reshape(height_count, 2, 2)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(afm.orm, "load_node", lambda pk: work)
    monkeypatch.setattr(afm.spm, "get_calc_by_label", lambda node, label: label)
    monkeypatch.setattr(afm.spm, "get_slab_calc_info", lambda structure: "")
    monkeypatch.setattr(afm, "load_afm_pp_data", lambda calc: (*grid, slices))
    monkeypatch.setattr(afm.plt, "show", lambda: afm.plt.close())

    widget = viewer(pk=work.pk)
    widget.create_zip_dl_link()

    assert widget.zip_progress.value == 1.0
    assert "afm_41.zip" in widget.download_zip_link.value
    with ZipFile(tmp_path / "tmp" / "afm_41.zip") as archive:
        assert set(archive.namelist()) == {
            f"{prefix}_{index:02d}.{suffix}"
            for prefix in prefixes
            for index in range(height_count)
            for suffix in ("txt", "png")
        }
        for index, expected in enumerate(slices):
            with archive.open(f"{prefixes[0]}_{index:02d}.txt") as data:
                np.testing.assert_array_equal(np.loadtxt(data), expected)
