"""Opt-in boundary conditions, explicit solver settings and reversible cell edits."""

import html
import json

import ipywidgets as ipw
import numpy as np

from aiida_nanotech_empa.workflows.cp2k.electrostatics import (
    PERIODICITIES,
    SOLVER_DIMENSIONS,
    poisson_settings,
    prepare_cells,
    validate_cell,
)


class PoissonSettingsWidget(ipw.VBox):
    def __init__(self, on_structure=None, get_geometries=None):
        self.on_structure = on_structure
        self.get_geometries = get_geometries
        self.prepared_geometries = []
        self.original = None
        self.applied = None
        self.publishing = False
        self._updating = False
        self.active = ipw.ToggleButton(
            value=False,
            description="Apply electrostatics and cell",
            icon="check",
            layout=ipw.Layout(width="280px"),
        )
        self.periodic = ipw.Dropdown(
            description="Periodic axes:",
            options=[
                ("Choose periodic cell axes", None),
                *[(p, p) for p in PERIODICITIES],
            ],
            value="NONE",
            style={"description_width": "120px"},
        )
        self.solver = ipw.Dropdown(
            description="Poisson solver:",
            options=list(SOLVER_DIMENSIONS),
            value="MT",
            style={"description_width": "120px"},
        )
        self.extra = ipw.Textarea(
            value="{}",
            description="POISSON settings:",
            style={"description_width": "140px"},
            layout=ipw.Layout(width="90%", height="90px"),
        )
        self.guidance = ipw.HTML()
        self.preview = ipw.HTML()
        advanced = ipw.Accordion(children=[ipw.VBox([self.extra])], selected_index=None)
        advanced.set_title(0, "Additional CP2K POISSON settings (JSON)")
        super().__init__(
            children=[
                self.guidance,
                self.periodic,
                self.solver,
                advanced,
                self.preview,
                self.active,
            ]
        )
        self.active.observe(self._changed, "value")
        for control in (self.periodic, self.solver, self.extra):
            control.observe(self._changed, "value")
        self.set_structure(None, {})

    def set_structure(self, atoms, details):
        if self.publishing:
            return
        self._updating = True
        try:
            self.original = None if atoms is None else atoms.copy()
            self.applied = None
            self.active.value = False
            self.active.description = "Apply electrostatics and cell"
            self.active.button_style = ""
            kind = details.get("system_type", "")
            periodic = details.get("periodic_axes")
            if periodic is None and kind != "Wire" and not kind.startswith("Slab"):
                periodic = {
                    "Molecule": "NONE",
                    "SlabXY": "XY",
                    "SlabYZ": "YZ",
                    "SlabXZ": "XZ",
                    "Bulk": "XYZ",
                }.get(kind, "XYZ")
            self.periodic.value = periodic
            self.solver.value = (
                "ANALYTIC"
                if kind == "Wire" or periodic in ("X", "Y", "Z")
                else "PERIODIC"
                if periodic == "XYZ"
                else "MT"
            )
            self.extra.value = "{}"
            dimension = {
                "Molecule": "0D molecule",
                "Wire": "1D wire",
                "Bulk": "3D system",
            }.get(kind, "2D slab" if kind.startswith("Slab") else "system")
            self.guidance.value = (
                f"<b>Detected: {html.escape(dimension)}.</b> Changes are off by default. "
                "Review the solver and cell preview before applying them. Deactivate to restore "
                "the original cell and positions.<br>"
                "<b>MT supports 0D and 2D, and is not implemented for 1D.</b> "
                "For 1D, select the periodic cell axis explicitly and a supported solver. "
                "ANALYTIC supports 0D/1D/2D but may converge slowly. "
                "WAVELET needs density to vanish at isolated cell faces; availability and "
                "supported orientations depend on the CP2K version/build. "
                "IMPLICIT and MULTIPOLE require method-specific settings and validation. "
                '<a href="https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/POISSON.html" '
                'target="_blank">CP2K solver guidance</a>.'
            )
        finally:
            self._updating = False
        self._refresh()

    def settings(self):
        if self.periodic.value is None:
            raise ValueError(
                "Choose the periodic cell axes explicitly; their orientation could not be identified"
            )
        try:
            extra = json.loads(self.extra.value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Additional POISSON settings must be a JSON object"
            ) from exc
        result = dict(
            periodic=self.periodic.value,
            poisson_solver=self.solver.value,
            poisson_settings=extra,
        )
        poisson_settings(result)
        return result

    def _proposal(self):
        if self.original is None:
            raise ValueError("Select a structure first")
        params = self.settings()
        geometries = self.get_geometries() if self.get_geometries is not None else None
        geometries = geometries or [self.original]
        self.prepared_geometries = prepare_cells(geometries, params)
        prepared = self.prepared_geometries[0]
        translation = prepared.positions[0] - geometries[0].positions[0]
        proposal = self.original.copy()
        proposal.cell = prepared.cell
        proposal.positions += translation
        proposal.pbc = prepared.pbc
        validate_cell(proposal, params)
        return proposal

    def _refresh(self):
        try:
            proposal = self._proposal()
        except (ValueError, KeyError) as exc:
            self.active.disabled = not self.active.value
            self.preview.value = f'<p style="color:#a00">{html.escape(str(exc))}</p>'
            return None
        self.active.disabled = False

        def lengths(atoms):
            return " × ".join(f"{x:.3f}" for x in atoms.cell.lengths())

        original_volume = abs(np.linalg.det(self.original.cell))
        new_volume = abs(np.linalg.det(proposal.cell))
        volume = (
            f"volume ×{new_volume / original_volume:.2f}"
            if original_volume > 1e-10
            else f"new volume {new_volume:.2f} Å³"
        )
        self.preview.value = (
            f"<p>Cell lengths: <b>{lengths(self.original)} Å → {lengths(proposal)} Å</b>; "
            f"{volume}. Periodic lattice vectors are retained. "
            "Only isolated directions may grow; the atoms are translated rigidly to the centre "
            "in those directions.<br>MT preparation uses twice the atomic span plus 15 Å; "
            "other solvers use the atomic span plus 15 Å. Already larger cells are retained. "
            "This is a geometric starting point: check density tails and vacuum convergence. "
            "Changes take effect only while the button is active.</p>"
        )
        return proposal

    def _publish(self, atoms):
        self.publishing = True
        try:
            self.applied = atoms.copy() if self.active.value else None
            if self.on_structure is not None:
                self.on_structure(atoms)
        finally:
            self.publishing = False

    def _changed(self, change):
        if self._updating:
            return
        self.active.description = (
            "Active — click to restore"
            if self.active.value
            else "Apply electrostatics and cell"
        )
        self.active.button_style = "success" if self.active.value else ""
        proposal = self._refresh()
        if self.active.value:
            if proposal is not None:
                self._publish(proposal)
        elif change["owner"] is self.active and self.original is not None:
            self._publish(self.original.copy())

    def return_dict(self):
        if not self.active.value:
            if self.original is not None:
                validate_cell(self.original, {"periodic": "XYZ"})
            return {"dft_params": {"periodic": "XYZ"}}
        if self._refresh() is None or self.applied is None:
            raise ValueError("Review and apply valid Poisson solver and cell settings")
        return {"dft_params": self.settings()}
