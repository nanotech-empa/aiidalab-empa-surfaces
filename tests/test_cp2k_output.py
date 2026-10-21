import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from surfaces_tools import helpers
from surfaces_tools.helpers import HART_2_EV
from surfaces_tools.utils.cp2k_output import format_bandgaps

HEADER = "  Eigenvalues of the occupied subspace spin            1\n"


def printed(*gaps):
    return HEADER + "".join(f" HOMO - LUMO gap [eV] : {gap}\n" for gap in gaps)


def test_corrected_parser_has_priority_over_stdout_workaround():
    parameters = {"bandgap_spin1_au": 2 / HART_2_EV, "eigenvalue_homo_spin1_au": -0.2}
    html = format_bandgaps(parameters, "RKS", printed(8) + printed(9))
    assert "2.000 (eV)" in html
    assert "8.000" not in html
    assert "9.000" not in html


@pytest.mark.parametrize(
    "dft_type,first,last", [("RKS", (8,), (2,)), ("UKS", (8, 9), (2, 3))]
)
def test_legacy_stdout_uses_last_orbital_group(dft_type, first, last):
    html = format_bandgaps(
        {"bandgap_spin1_au": 0.7}, dft_type, printed(*first) + printed(*last)
    )
    for gap in last:
        assert f"{gap:.3f} (eV)" in html
    for gap in first:
        assert f"{gap:.3f} (eV)" not in html


def test_incomplete_final_group_never_borrows_previous_gaps():
    html = format_bandgaps({"bandgap_spin1_au": 0.7}, "UKS", printed(8, 9) + printed(2))
    assert "Gap spin1: </b>2.000" in html
    assert "Gap spin2: </b>unavailable" in html
    assert "9.000" not in html
    html = format_bandgaps({"bandgap_spin1_au": 0.7}, "RKS", printed(8) + HEADER)
    assert "Gap: </b>unavailable" in html
    assert "8.000" not in html


def test_missing_lumos_stay_unavailable_in_current_parser():
    html = format_bandgaps({"eigenvalue_homo_spin1_au": -0.2}, "RKS", printed(8))
    assert "Gap: </b>unavailable" in html
    assert "does not establish a zero gap" in html
    assert "8.000" not in html


@pytest.mark.parametrize("gap", [0.0, -0.1])
def test_actual_zero_and_negative_gaps_are_not_replaced(gap):
    html = format_bandgaps(
        {"bandgap_spin1_au": gap / HART_2_EV, "eigenvalue_homo_spin1_au": -0.2}, "RKS"
    )
    assert f"{gap:.3f} (eV)" in html
    assert "unavailable" not in html


def test_partial_uks_keeps_spin_labels():
    html = format_bandgaps(
        {"bandgap_spin2_au": 2 / HART_2_EV, "printed_bandgap_spin2_ev": 2}, "UKS"
    )
    assert "Gap spin1: </b>unavailable" in html
    assert "Gap spin2: </b>2.000" in html


def test_legacy_parameters_without_stdout_have_a_warning():
    html = format_bandgaps({"bandgap_spin1_au": 2 / HART_2_EV}, "RKS")
    assert "2.000 (eV)" in html
    assert "Legacy stored gap values" in html


def test_gap_only_stdout_keeps_final_pair():
    stdout = printed(8, 9).replace(HEADER, "") + printed(2, 3).replace(HEADER, "")
    html = format_bandgaps({}, "UKS", stdout)
    assert "2.000 (eV)" in html
    assert "3.000 (eV)" in html
    assert "8.000" not in html


@pytest.mark.parametrize("gap", [None, 0.0, 2.0])
def test_geometry_notebook_displays_parser_gap_or_unavailable(gap):
    path = Path(__file__).resolve().parents[1] / "view_geometry_optimization.ipynb"
    notebook = json.loads(path.read_text())
    parameters = {"eigenvalue_homo_spin1_au": -0.2}
    if gap is not None:
        parameters["bandgap_spin1_au"] = gap / HART_2_EV
    namespace = {
        "workcalc": SimpleNamespace(
            label="CP2K_SCF",
            outputs=SimpleNamespace(
                output_parameters=SimpleNamespace(get_dict=lambda: parameters),
                retrieved=SimpleNamespace(open=lambda _: StringIO(printed(8))),
            ),
        ),
        "opt_structure": SimpleNamespace(pk=1),
        "helpers": helpers,
        "format_bandgaps": format_bandgaps,
    }
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "html += format_bandgaps(" in "".join(cell["source"])
    )
    exec(compile(source, str(path), "exec"), namespace)
    assert "SCF structure" in namespace["html"]
    expected = "unavailable" if gap is None else f"{gap:.3f} (eV)"
    assert f"Gap: </b>{expected}" in namespace["html"]
    assert "8.000" not in namespace["html"]
