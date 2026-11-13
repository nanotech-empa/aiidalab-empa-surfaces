"""Real widgets with invented PDOS spectra and independent AiiDA test state."""

import base64
import io
import re
from pathlib import Path
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import ProcessState
from ase import Atoms

pytest_plugins = ["aiida.tools.pytest_fixtures"]
ROOT = Path(__file__).parent


@pytest.fixture
def module():
    from surfaces_tools.widgets import pdos

    yield pdos
    plt.close("all")


@pytest.fixture
def sources(aiida_profile_clean):
    assert aiida_profile_clean.is_test_profile
    energies = np.r_[-0.3, -0.1, np.linspace(0.1, 0.8, 98)]
    files = {}
    for name, weight in [
        ("aiida-k1-1.pdos", 1.0),
        ("aiida-list1-1.pdos", 0.25),
        ("aiida-list2-1.pdos", 0.75),
        ("aiida-list3-1.pdos", 1.0),
    ]:
        stream = io.StringIO()
        stream.write("# Projected DOS for atomic kind He\n# synthetic fixture\n")
        np.savetxt(
            stream,
            np.column_stack(
                [
                    np.arange(1, 101),
                    energies,
                    np.r_[2, 2, np.zeros(98)],
                    np.full(100, weight),
                ]
            ),
            fmt="%.12g",
        )
        files[name] = stream.getvalue()
    fixture = {
        "files": files,
        "pdos_lists": [
            ["1", "left fragment"],
            ["2", "right fragment"],
            ["1..2", "all"],
        ],
        "output_parameters": {
            "eigen_spin1_au": [-0.3, -0.1],
            "eigenvalue_homo_spin1_au": -0.1,
            "eigenvalue_lumo_spin1_au": 0.1,
        },
    }

    def create(shift=0.0, two_spins=False, do_overlap=False):
        root = orm.WorkChainNode()
        inputs = {
            "structure": orm.StructureData(
                ase=Atoms(
                    "He2", positions=[[5, 6, 6], [6, 6, 6]], cell=[12] * 3, pbc=True
                )
            ),
            "pdos_lists": orm.List(list=fixture["pdos_lists"]),
            "dft_params": orm.Dict(dict={"uks": two_spins}),
            "do_overlap": orm.Bool(do_overlap),
            "overlap_params": orm.Dict(dict={"--emin1": "-2", "--emax1": "2"}),
        }
        for label, value in inputs.items():
            root.base.links.add_incoming(
                value.store(), link_type=LinkType.INPUT_WORK, link_label=label
            )
        root.store()
        retrieved = orm.FolderData()
        for name, text in fixture["files"].items():
            retrieved.base.repository.put_object_from_filelike(io.StringIO(text), name)
            if two_spins:
                lines = text.splitlines(True)
                values = np.loadtxt(io.StringIO(text))
                values[:, 1] += shift / 27.211386245988
                out = io.StringIO()
                out.write("".join(lines[:2]))
                np.savetxt(out, values, fmt="%.12g")
                out.seek(0)
                retrieved.base.repository.put_object_from_filelike(
                    out, name.replace("aiida-", "aiida-BETA-")
                )
        retrieved.store()
        retrieved.base.links.add_incoming(
            root, link_type=LinkType.RETURN, link_label="slab_retrieved"
        )
        # Real legacy reference discovery through CALL/RETURN links, no parser mock.
        ot = orm.WorkChainNode(label="slab_scf")
        ot.base.links.add_incoming(root, link_type=LinkType.CALL_WORK, link_label="ot")
        ot.store()
        output = orm.Dict(dict=fixture["output_parameters"]).store()
        output.base.links.add_incoming(
            ot, link_type=LinkType.RETURN, link_label="output_parameters"
        )
        for node in (ot, root):
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(0)
            node.seal()
        return root.uuid

    yield create
    assert orm.QueryBuilder().append(orm.CalcJobNode).count() == 0


def test_viewer_reads_overlap_from_repository_stream_without_name(
    module, sources, monkeypatch
):
    """Packed repository readers are seekable streams, not filesystem paths."""
    from types import SimpleNamespace

    identifier = sources(do_overlap=True)
    matrix = np.zeros((100, 3))
    matrix[:3] = np.eye(3)
    archive = io.BytesIO()
    np.savez(
        archive,
        metadata=np.array([{"nspin_g1": 1, "nspin_g2": 1, "homo_i_g2": [1]}]),
        overlap_matrix_s0s0=matrix,
        energies_g1_s0=np.linspace(-8, 8, 100),
        energies_g2_s0=np.array([-2.0, -1.0, 1.0]),
        orb_indexes_g2_s0=np.array([0, 1, 2]),
    )
    archive.seek(0)
    assert not hasattr(archive, "name")
    requested = []

    def open_archive(name, mode):
        requested.append((name, mode))
        return archive

    calculation = SimpleNamespace(
        outputs=SimpleNamespace(
            retrieved=SimpleNamespace(
                open=open_archive,
                base=SimpleNamespace(repository=SimpleNamespace(open=open_archive)),
            )
        )
    )

    def find_overlap(workchain, label):
        assert workchain.uuid == identifier and label == "overlap"
        return calculation

    monkeypatch.setattr(module.spm, "get_calc_by_label", find_overlap)
    widget = module.PdosOverlapViewerWidget()
    try:
        widget.load_data(identifier)
        assert requested == [("overlap.npz", "rb")]
        assert archive.closed
        assert widget._workchain_uuid == identifier
        assert not widget._overlap.add_item_button.disabled
        np.testing.assert_array_equal(widget._overlap.data["overlap_matrix"][0], matrix)
        np.testing.assert_array_equal(
            widget._overlap.data["energies_g2"][0], [-2.0, -1.0, 1.0]
        )
    finally:
        widget.close()


def choose_three(viewer):
    stack = viewer._projections
    stack.items = ()
    for series, color in [
        ("sel_3", "#000000"),
        ("sel_1", "#ff0000"),
        ("sel_2", "#0000ff"),
    ]:
        stack.add_item(None)
        item = stack.items[-1]
        item._data_selection.value = series
        item._color_picker.value = color
        item._norm_factor.value = 1


def test_actual_load_includes_native_frontiers_and_all_stored_states(module, sources):
    viewer = module.PdosOverlapViewerWidget()
    identifier = sources()
    viewer.load_data(identifier)
    energies = viewer._projections.data["tdos"][0][:, 0]
    slider = viewer._energy_range_slider
    assert energies.shape == (100,)
    assert slider.min < energies.min() < -2.435 < 2.435 < energies.max() < slider.max
    assert slider.value == (slider.min, slider.max)
    assert orm.load_node(identifier).inputs.overlap_params.get_dict() == {
        "--emin1": "-2",
        "--emax1": "2",
    }
    slider.value = (-3.5, 3.5)
    choose_three(viewer)
    fig, (headers, plotted) = viewer._create_the_plot()
    assert plotted.shape[1] == 4
    assert headers[1:] == ["all s0", "left fragment s0", "right fragment s0"]
    energy = plotted[:, 0]
    for frontier in (-0.1 * 27.211386245988, 0.1 * 27.211386245988):
        mask = abs(energy - frontier) < 0.05
        assert plotted[mask, 1].max() > 1
    # Preserve native numeric weights and the existing CSV format/precision.
    np.testing.assert_allclose(plotted[:, 1], plotted[:, 2] + plotted[:, 3], atol=2e-6)
    link = module.make_csv_link((headers, plotted))
    payload = re.search(r"base64,([^\"]+)", link)[1]
    csv = np.loadtxt(io.StringIO(base64.b64decode(payload).decode()), delimiter=",")
    np.testing.assert_allclose(csv, plotted, rtol=5e-5, atol=1e-8)
    plt.close(fig)


def test_manual_reference_preserves_three_curves_and_absolute_window(module, sources):
    viewer = module.PdosOverlapViewerWidget()
    identifier = sources()
    viewer.load_data(identifier)
    choose_three(viewer)
    viewer._energy_range_slider.value = (-3.5, 3.5)
    state = viewer._projections.selection_state()
    old = viewer._projections.data["tdos"][0].copy()
    ref = viewer._projections.data["_energy_reference"]["ev"]
    viewer.clear_plot = Mock(wraps=viewer.clear_plot)
    viewer._manual_ref_checkbox.value = True
    viewer._manual_ref_input.value = -1
    viewer._manual_ref_button.click()  # Actual linked Apply E_ref callback.
    viewer.clear_plot.assert_called_once()
    assert viewer._projections.selection_state() == state
    new = viewer._projections.data["tdos"][0]
    np.testing.assert_allclose(new[:, 0], old[:, 0] + ref + 1, atol=1e-12)
    np.testing.assert_array_equal(new[:, 1], old[:, 1])
    np.testing.assert_allclose(
        viewer._energy_range_slider.value, np.array([-3.5, 3.5]) + ref + 1
    )
    assert "-1.000 eV" in viewer._geometry_info.value
    viewer._manual_ref_checkbox.value = False
    viewer._manual_ref_button.click()
    assert viewer._projections.selection_state() == state
    np.testing.assert_allclose(viewer._projections.data["tdos"][0], old, atol=1e-12)
    np.testing.assert_allclose(viewer._energy_range_slider.value, [-3.5, 3.5])


@pytest.mark.parametrize("reference", [-100, 100])
def test_large_reference_shifts_do_not_break_slider_validation(
    module, sources, reference
):
    viewer = module.PdosOverlapViewerWidget()
    uuid = sources()
    viewer.load_data(uuid)
    viewer.load_data(uuid, manual_energy_reference_ev=reference)
    low, high = viewer._energy_range_slider.value
    assert (
        viewer._energy_range_slider.min <= low < high <= viewer._energy_range_slider.max
    )


def test_same_source_reopen_preserves_rows_but_new_source_resets(module, sources):
    viewer = module.PdosOverlapViewerWidget()
    first, second = sources(), sources()
    viewer.load_data(first)
    choose_three(viewer)
    viewer._energy_range_slider.value = (-3.5, 3.5)
    state = viewer._projections.selection_state()
    viewer.load_data(orm.load_node(first).pk)  # Same source, alternate identifier.
    assert viewer._projections.selection_state() == state
    assert viewer._energy_range_slider.value == (-3.5, 3.5)
    viewer.clear_plot = Mock(wraps=viewer.clear_plot)
    viewer.load_data(second)
    viewer.clear_plot.assert_called_once()
    assert viewer._projections.selection_state() == [(0, "tdos", "lightgray", 0.02)]
    assert viewer._overlap.items == ()
    assert viewer._overlap.add_item_button.disabled


def test_two_spin_bounds_and_selected_down_spin_are_preserved(module, sources):
    viewer = module.PdosOverlapViewerWidget()
    uuid = sources(shift=5, two_spins=True)
    viewer.load_data(uuid)
    up, down = viewer._projections.data["tdos"]
    assert viewer._energy_range_slider.min < min(up[:, 0].min(), down[:, 0].min())
    assert viewer._energy_range_slider.max > max(up[:, 0].max(), down[:, 0].max())
    viewer._projections.items[1]._color_picker.value = "green"
    state = viewer._projections.selection_state()
    viewer.load_data(uuid, manual_energy_reference_ev=-1)
    assert viewer._projections.selection_state() == state


def test_nonfinite_bound_values_are_not_used_and_empty_spectrum_is_explicit(module):
    assert module.pdos_display_bounds(
        {"tdos": [np.array([[-2, 1], [3, 1], [np.nan, 1]])]}
    ) == (-2.5, 3.5)
    with pytest.raises(ValueError, match="No finite stored PDOS"):
        module.pdos_display_bounds({"tdos": [np.array([[np.nan, 1], [np.inf, 1]])]})


def test_empty_user_selection_is_preserved_on_same_source(module, sources):
    viewer = module.PdosOverlapViewerWidget()
    uuid = sources()
    viewer.load_data(uuid)
    viewer._projections.items = ()
    viewer.load_data(uuid, manual_energy_reference_ev=-1)
    assert viewer._projections.items == ()
