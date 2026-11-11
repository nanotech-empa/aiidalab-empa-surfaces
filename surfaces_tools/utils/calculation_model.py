"""Calculation-aware legacy display/export; never modify stored structures."""

import html
import re

from aiida import orm
from aiida_nanotech_empa.utils.cp2k_results import (
    calculation_boundary,
    parse_neb_status,
)


def structure_for_display(structure):
    """Return a transient ASE copy and an explicit source/effective-model notice."""
    atoms = structure.get_ase()
    stored_axes = (
        "".join(axis for axis, enabled in zip("XYZ", atoms.pbc) if enabled) or "NONE"
    )
    creator = structure.creator
    if (
        not isinstance(creator, orm.CalcJobNode)
        or "cp2k" not in (creator.process_type or "").lower()
    ):
        return (
            atoms,
            f"Stored structure PBC: {stored_axes}. No CP2K calculation boundary model was inferred.",
        )
    try:
        boundary = calculation_boundary(creator.inputs.parameters.get_dict())
    except (AttributeError, KeyError, ValueError) as exc:
        return (
            atoms,
            f"Calculation boundary model is unavailable or inconsistent ({exc}). View/export retains stored PBC {stored_axes}; it is not a verified calculation model.",
        )
    axes = boundary["periodic"]
    nonperiodic = "".join(axis for axis in "XYZ" if axis not in axes) or "NONE"
    model = "0D nonperiodic" if axes == "NONE" else f"{boundary['dimension']}D periodic"
    message = f"Effective calculation model: {model}; periodic axes {axes}; nonperiodic axes {nonperiodic}; Poisson solver {boundary['poisson_solver']}."
    if tuple(atoms.pbc) != boundary["pbc"]:
        atoms.pbc = boundary["pbc"]
        atoms.info.update(
            source_structure_uuid=structure.uuid,
            calculation_uuid=creator.uuid,
            stored_periodicity=stored_axes,
            effective_periodicity=axes,
        )
        message += f" Legacy stored PBC is {stored_axes}. This view and its downloads use the calculation model on a temporary copy; the original stored node is unchanged."
    else:
        message += " Stored structure PBC agrees."
    return atoms, message


def neb_status_for_view(workchain):
    """Read current status or reconstruct legacy status from retained native logs."""
    if "output_parameters" in workchain.outputs:
        status = workchain.outputs.output_parameters.get_dict()
        if status.get("neb_status_version") == 1:
            return status
    calculations = []
    for node in workchain.called_descendants:
        if not isinstance(node, orm.CalcJobNode) or "parameters" not in node.inputs:
            continue
        params = node.inputs.parameters.get_dict()
        if (
            params.get("GLOBAL", {}).get("RUN_TYPE") == "BAND"
            and "retrieved" in node.outputs
        ):
            calculations.append(node)
    if not calculations:
        return parse_neb_status({})
    native = max(calculations, key=lambda node: node.ctime)
    repository = native.outputs.retrieved.base.repository
    texts = {
        name: repository.get_object_content(name)
        for name in repository.list_object_names()
        if name == "aiida.out" or re.fullmatch(r"aiida-BAND\d+\.out", name)
    }
    expected = (
        native.inputs.parameters.get_dict()
        .get("MOTION", {})
        .get("BAND", {})
        .get("NUMBER_OF_REPLICA")
    )
    return parse_neb_status(texts, expected)


def neb_status_html(status):
    def verdict(value):
        return (
            "converged"
            if value is True
            else "not converged"
            if value is False
            else "not established"
        )

    return (
        "<p><b>NEB result status:</b> "
        + html.escape(status["termination_reason"])
        + ". "
        "Timed out: "
        + (
            "yes"
            if status["exceeded_walltime"]
            else "no timeout found in retained logs"
        )
        + "; "
        "electronic convergence across recorded SCF calls: "
        + verdict(status["electronic_converged"])
        + "; "
        "band convergence: " + verdict(status["band_converged"]) + ". "
        "Plots show accepted band steps. Partial results remain available; a barrier trend alone is not proof of convergence.</p>"
    )
