from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aiida import orm
from aiida.common.links import LinkType

from surfaces_tools.helpers import HART_2_EV
from surfaces_tools.widgets import pdos

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def pdos_workchain(tmp_path):
    """A retrieved PDOS file whose energies and weights are independent of its display zero."""
    filename = "aiida-k1-1.pdos"
    (tmp_path / filename).write_text(
        "# Projected DOS for atomic kind C\n"
        "1 -0.5 2 0.2 0.3\n"
        "2 -0.2 2 0.4 0.5\n"
        "3 0.1 0 0.6 0.7\n"
    )
    retrieved = SimpleNamespace(
        list_object_names=lambda: [filename],
        open=lambda name: (tmp_path / name).open(),
    )
    return SimpleNamespace(outputs=SimpleNamespace(slab_retrieved=retrieved))


def test_no_lumo_does_not_become_zero_gap():
    reference = pdos.estimate_energy_reference_from_output_parameters(
        {"eigen_spin1_au": [-0.5, -0.2]}
    )

    assert reference["hartree"] == -0.2
    assert reference["homo_hartree"] == -0.2
    assert reference["lumo_hartree"] is None
    assert reference["spin_frontiers"][0]["lumo"] is None
    assert reference["missing_lumo_spins"] == [1]
    assert not reference["uses_legacy_lumo"]
    assert "unavailable" in reference["warning"]
    assert "Legacy" not in reference["warning"]


def test_partial_uks_frontiers_do_not_establish_a_global_lumo():
    reference = pdos.estimate_energy_reference_from_output_parameters(
        {
            "eigen_spin1_au": [-0.5, -0.2],
            "eigen_spin2_au": [-0.6, -0.3],
            "eigenvalue_lumo_spin1_au": 0.1,
        }
    )

    assert reference["hartree"] == -0.2
    assert reference["lumo_hartree"] is None
    assert reference["missing_lumo_spins"] == [2]
    assert reference["spin_frontiers"][0]["lumo"] == 0.1
    assert reference["spin_frontiers"][1]["lumo"] is None


def test_overlapping_uks_frontiers_keep_the_lower_lumo_and_homo_reference():
    reference = pdos.estimate_energy_reference_from_output_parameters(
        {
            "eigen_spin1_au": [-0.5, -0.2],
            "eigen_spin2_au": [-0.6, -0.3],
            "eigenvalue_lumo_spin1_au": 0.1,
            "eigenvalue_lumo_spin2_au": -0.25,
        }
    )

    assert reference["hartree"] == reference["homo_hartree"] == -0.2
    assert reference["lumo_hartree"] == -0.25
    assert reference["spin_frontiers"][1]["lumo"] == -0.25
    assert reference["overlapping_frontiers"]
    assert not reference["missing_lumo_spins"]
    assert "Spin frontiers overlap" in reference["warning"]


def test_legacy_gap_keeps_the_existing_homo_reference_and_warning():
    reference = pdos.estimate_energy_reference_from_output_parameters(
        {"eigen_spin1_au": [-0.5, -0.2], "bandgap_spin1_au": 0.3}
    )

    assert reference["hartree"] == -0.2
    assert reference["lumo_hartree"] == pytest.approx(0.1)
    assert reference["uses_legacy_lumo"]
    assert not reference["missing_lumo_spins"]
    assert "parsed gaps" in reference["warning"]


@pytest.mark.parametrize("manual_reference", [None, -1.0])
@pytest.mark.parametrize("computed_zero", [False, True])
def test_pdos_metadata_and_labels_distinguish_unknown_from_zero(
    pdos_workchain, monkeypatch, manual_reference, computed_zero
):
    parameters = {"eigen_spin1_au": [-0.5, -0.2]}
    if computed_zero:
        parameters.update(
            eigenvalue_homo_spin1_au=-0.2,
            eigenvalue_lumo_spin1_au=-0.2,
            bandgap_spin1_au=0.0,
        )
    reference = pdos.estimate_energy_reference_from_output_parameters(parameters)
    monkeypatch.setattr(pdos, "get_pdos_energy_reference", lambda _: reference)

    data = pdos.process_pdos_files(
        pdos_workchain, manual_energy_reference_ev=manual_reference
    )
    metadata = data["_energy_reference"]
    expected_reference = (
        -0.2 if manual_reference is None else manual_reference / HART_2_EV
    )
    np.testing.assert_allclose(
        data["tdos"][0][:, 0],
        (np.array([-0.5, -0.2, 0.1]) - expected_reference) * HART_2_EV,
    )
    np.testing.assert_allclose(data["tdos"][0][:, 1], [0.5, 0.9, 1.3])

    html = pdos.PdosOverlapViewerWidget._format_energy_reference_info(None, metadata)
    if computed_zero:
        assert metadata["global_gap_ev"] == 0.0
        assert metadata["lumo_ev"] == pytest.approx(-0.2 * HART_2_EV)
        assert "unavailable" not in html
        assert "Frontier LUMO" in html
    else:
        assert metadata["lumo_ev"] is None
        assert metadata["lumo_rel_ev"] is None
        assert metadata["global_gap_ev"] is None
        assert "unavailable" in html
        assert "Legacy" not in html
        assert "Frontier LUMO relative" not in html


def test_empty_orbital_data_still_requires_a_usable_reference():
    with pytest.raises(ValueError, match="No spin-resolved eigenvalue"):
        pdos.estimate_energy_reference_from_output_parameters({"eigen_spin1_au": []})


@pytest.mark.parametrize("second_lumo,expected_gap", [(0.2, 0.3), (-0.2, 0.0)])
def test_nonoverlapping_frontiers_keep_the_common_midpoint(second_lumo, expected_gap):
    reference = pdos.estimate_energy_reference_from_output_parameters(
        {
            "eigen_spin1_au": [-0.5, -0.2],
            "eigen_spin2_au": [-0.6, -0.3],
            "eigenvalue_lumo_spin1_au": 0.1,
            "eigenvalue_lumo_spin2_au": second_lumo,
        }
    )
    assert reference["hartree"] == pytest.approx(-0.2 + expected_gap / 2)
    assert not reference["overlapping_frontiers"]
    assert "warning" not in reference


@pytest.mark.parametrize("manual_reference", [None, -1.0])
def test_overlapping_spin_frontiers_remain_distinct_from_missing_lumos(
    pdos_workchain, monkeypatch, manual_reference
):
    retrieved = pdos_workchain.outputs.slab_retrieved
    with retrieved.open("aiida-k1-1.pdos") as handle:
        original = Path(handle.name)
    alpha = original.with_name("aiida-ALPHA_k1-1.pdos")
    beta = original.with_name("aiida-BETA_k1-1.pdos")
    original.rename(alpha)
    beta.write_text(alpha.read_text())
    retrieved.list_object_names = lambda: [alpha.name, beta.name]
    reference = pdos.estimate_energy_reference_from_output_parameters(
        {
            "eigen_spin1_au": [-0.5, -0.2],
            "eigen_spin2_au": [-0.6, -0.3],
            "eigenvalue_lumo_spin1_au": 0.1,
            "eigenvalue_lumo_spin2_au": -0.25,
        }
    )
    monkeypatch.setattr(pdos, "get_pdos_energy_reference", lambda _: reference)
    data = pdos.process_pdos_files(
        pdos_workchain, manual_energy_reference_ev=manual_reference
    )
    metadata = data["_energy_reference"]
    assert metadata["overlapping_frontiers"]
    assert metadata["global_gap_ev"] == 0.0
    assert metadata["frontier_separation_ev"] == pytest.approx(-0.05 * HART_2_EV)
    assert metadata["lumo_ev"] == pytest.approx(-0.25 * HART_2_EV)
    assert not metadata["missing_lumo_spins"]
    expected_zero = -0.2 if manual_reference is None else manual_reference / HART_2_EV
    for spin in range(2):
        np.testing.assert_allclose(
            data["tdos"][spin][:, 0],
            (np.array([-0.5, -0.2, 0.1]) - expected_zero) * HART_2_EV,
        )
        np.testing.assert_allclose(data["tdos"][spin][:, 1], [0.5, 0.9, 1.3])
    html = pdos.PdosOverlapViewerWidget._format_energy_reference_info(None, metadata)
    assert "Spin frontiers overlap" in html
    assert "Signed LUMO-HOMO frontier separation" in html
    assert "no positive global frontier gap" in html
    assert "midpoint" not in html


def test_pdos_widget_retains_only_uuid_and_reloads_nodes(aiida_profile, monkeypatch):
    workchain = orm.WorkChainNode()
    workchain.base.links.add_incoming(
        orm.List([]).store(), LinkType.INPUT_WORK, "pdos_lists"
    )
    workchain.store()
    loaded = []
    processed = []
    original_load = orm.load_node

    def load(identifier):
        loaded.append(identifier)
        return original_load(identifier)

    def process(node, **kwargs):
        processed.append((node.uuid, kwargs.get("manual_energy_reference_ev")))
        return {"tdos": [np.array([[-1.0, 0.5], [1.0, 0.7]])]}

    monkeypatch.setattr(pdos.orm, "load_node", load)
    monkeypatch.setattr(pdos, "process_pdos_files", process)
    widget = pdos.PdosStackWidget(item_class=pdos.PdosSelectionWidget)
    try:
        widget.workchain = workchain
        assert widget.workchain_uuid == workchain.uuid
        assert len(loaded) == 1
        assert all(
            not isinstance(value, orm.Node) for value in widget._trait_values.values()
        )
        widget.manual_energy_reference_ev = -1.0
        widget.load_workchain()
        assert len(loaded) == 2
        assert processed[-1] == (workchain.uuid, -1.0)
        assert widget.workchain.uuid == workchain.uuid
        assert len(loaded) == 3
        widget.workchain = None
        assert widget.workchain_uuid is None
        assert widget.data == {}
    finally:
        # Close the full widget tree constructed by this regression.
        seen = set()

        def close(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            for child in getattr(value, "children", ()):
                close(child)
            value.close()

        close(widget)


@pytest.mark.parametrize("legacy", [False, True])
def test_reference_comes_from_full_system_ot_not_molecule_or_diagonalization(legacy):
    def calculation(pk, ctime, homo, label=""):
        return SimpleNamespace(
            pk=pk,
            ctime=ctime,
            label=label,
            process_label="Cp2kBaseWorkChain",
            outputs=SimpleNamespace(
                output_parameters={
                    "eigen_spin1_au": [homo],
                    "eigenvalue_lumo_spin1_au": homo + 0.4,
                }
            ),
        )

    slab_ot = calculation(10, 1, -0.5, "slab_scf")
    slab_diag = calculation(11, 2, 5.0)
    molecule_ot = calculation(12, 3, 20.0, "molecule_scf")
    slab = SimpleNamespace(
        process_label="Cp2kDiagWorkChain",
        ctime=1,
        pk=1,
        called_descendants=[slab_diag, slab_ot],
    )
    molecule = SimpleNamespace(
        process_label="Cp2kDiagWorkChain",
        ctime=2,
        pk=2,
        called_descendants=[molecule_ot],
    )
    descendants = (
        [molecule_ot, slab_diag, slab_ot]
        if legacy
        else [molecule, slab_diag, slab, molecule_ot, slab_ot]
    )
    reference = pdos.get_pdos_energy_reference(
        SimpleNamespace(called_descendants=descendants)
    )
    assert reference["source_pk"] == slab_ot.pk
    assert reference["hartree"] == pytest.approx(-0.3)
