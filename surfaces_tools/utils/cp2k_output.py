"""Display final CP2K gaps without treating missing unoccupied levels as zero."""

import re

from ..helpers import HART_2_EV


def format_bandgaps(output_parameters, dft_type, stdout=""):
    """Prefer corrected parser values; recover final printed gaps for old nodes.

    Older parsers could calculate an occupied--occupied spacing as a band gap.
    Their stdout workaround must use the final orbital group, including an
    incomplete group, rather than silently borrowing gaps from an earlier step.
    """
    spins = (1, 2) if dft_type == "UKS" else (1,)
    gaps = {
        spin: output_parameters[f"bandgap_spin{spin}_au"] * HART_2_EV
        for spin in spins
        if output_parameters.get(f"bandgap_spin{spin}_au") is not None
    }
    current_parser = any(
        key.startswith(("eigenvalue_", "unoccupied_eigen_", "printed_bandgap_"))
        for key in output_parameters
    )
    warning = ""
    if not current_parser:
        printed_gaps = {}
        orbital_group_seen = False
        next_spin = 1
        gap_seen = False
        for line in stdout.splitlines():
            if re.search(r"Eigenvalues of the occupied subspace\s+spin\s+1\b", line):
                printed_gaps = {}
                next_spin = 1
                orbital_group_seen = True
            if line.strip().startswith("HOMO - LUMO gap [eV]"):
                if next_spin > len(spins):
                    printed_gaps = {}
                    next_spin = 1
                printed_gaps[next_spin] = float(line.split()[-1])
                next_spin += 1
                gap_seen = True
        if orbital_group_seen or gap_seen:
            gaps = printed_gaps
        elif gaps:
            warning = (
                "<br><span style='color:#b45309'>Legacy stored gap values: "
                "no final printed orbital group is available for verification.</span>"
            )

    html = ""
    for spin in spins:
        label = f"Gap spin{spin}" if len(spins) == 2 else "Gap"
        value = f"{gaps[spin]:.3f} (eV)" if spin in gaps else "unavailable"
        html += f"<b>{label}: </b>{value}<br>"
    if len(gaps) < len(spins):
        html += (
            "No LUMO/gap was available for the missing channel(s); "
            "this does not establish a zero gap.<br>"
        )
    return html + warning
