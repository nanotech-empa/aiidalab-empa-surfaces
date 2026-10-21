from types import SimpleNamespace

import numpy as np
import pytest

from surfaces_tools.helpers import HART_2_EV
from surfaces_tools.widgets import pdos


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


def test_complete_uks_frontiers_keep_one_common_reference():
    reference = pdos.estimate_energy_reference_from_output_parameters(
        {
            "eigen_spin1_au": [-0.5, -0.2],
            "eigen_spin2_au": [-0.6, -0.3],
            "eigenvalue_lumo_spin1_au": 0.1,
            "eigenvalue_lumo_spin2_au": -0.25,
        }
    )

    assert reference["hartree"] == pytest.approx(-0.05)
    assert reference["homo_hartree"] == -0.2
    assert reference["lumo_hartree"] == 0.1
    assert reference["spin_frontiers"][1]["lumo"] == -0.25
    assert not reference["missing_lumo_spins"]
    assert "warning" not in reference


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
