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
            description="Apply selected settings",
            icon="check",
            layout=ipw.Layout(width="310px"),
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
            description="Optional overrides:",
            style={"description_width": "140px"},
            layout=ipw.Layout(width="90%", height="90px"),
        )
        self.guidance = ipw.HTML()
        self.current_settings = ipw.HTML()
        self.preview = ipw.HTML()
        extra_help = ipw.HTML(
            "<p><b>Usually leave <code>{}</code>.</b> This uses CP2K's default "
            "parameters for the solver selected above. You do not need to enter JSON "
            "to choose a solver or its periodic axes.</p>"
            "<p>For example, with <b>MT</b> selected, this explicitly retains its "
            "default ALPHA value of 7.0:</p>"
            '<pre>{\n  "MT": {\n    "ALPHA": 7.0\n  }\n}</pre>'
            "<p>This MT example applies only to the MT solver. For a basic "
            "<b>ANALYTIC</b> wire setup, leave <code>{}</code> and choose its "
            "periodic axis above. Set <code>PERIODIC</code> and "
            "<code>POISSON_SOLVER</code> using the dropdowns, not in this field. "
            '<a href="https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/POISSON/MT.html" '
            'target="_blank">CP2K MT parameter reference</a>.</p>'
        )
        advanced = ipw.Accordion(
            children=[ipw.VBox([extra_help, self.extra])], selected_index=None
        )
        advanced.set_title(0, "Advanced: additional CP2K POISSON settings (JSON)")
        examples = ipw.HTML(
            '<table style="border-collapse:collapse;text-align:left">'
            '<tr><th style="padding:5px 12px">Example system</th>'
            '<th style="padding:5px 12px">Periodic axes</th>'
            '<th style="padding:5px 12px">Solver</th>'
            '<th style="padding:5px 12px">When you apply</th></tr>'
            '<tr><td style="padding:5px 12px">Isolated molecule</td><td>NONE</td><td>MT</td>'
            '<td style="padding:5px 12px">Prepare space around the molecule in all directions.</td></tr>'
            '<tr><td style="padding:5px 12px">Slab in the xy plane</td><td>XY</td><td>MT</td>'
            '<td style="padding:5px 12px">Keep the in-plane cell vectors; prepare vacuum along z.</td></tr>'
            '<tr><td style="padding:5px 12px">Wire along z</td><td>Z</td><td>ANALYTIC</td>'
            '<td style="padding:5px 12px">Keep the wire cell vector; prepare space in x and y. '
            "MT is not implemented for 1D.</td></tr></table>"
            "<p>For a wire along x or y, choose X or Y instead. For a slab in "
            "another cell plane, choose XZ or YZ. ANALYTIC may converge slowly. "
            "WAVELET needs the density to vanish at isolated cell faces; build and "
            "orientation restrictions may apply. IMPLICIT and MULTIPOLE need "
            "method-specific settings. "
            '<a href="https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/POISSON.html" '
            'target="_blank">CP2K solver guidance</a>.</p>'
        )
        guide = ipw.Accordion(children=[examples], selected_index=None)
        guide.set_title(0, "Examples: molecule, slab and wire")
        super().__init__(
            children=[
                self.guidance,
                self.current_settings,
                guide,
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
            self.active.description = "Apply selected settings"
            self.active.button_style = ""
            kind = details.get("system_type", "")
            periodic = details.get("periodic_axes")
            if periodic is None and kind != "Wire":
                periodic = {
                    "Molecule": "NONE",
                    "SlabXY": "XY",
                    "SlabYZ": "YZ",
                    "SlabXZ": "XZ",
                    "Bulk": "XYZ",
                }.get(kind, None if kind.startswith("Slab") else "XYZ")
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
                f"<p><b>Detected: {html.escape(dimension)}.</b> "
                "Review the suggested solver and cell changes below. "
                "<b>MT supports molecules (0D) and slabs (2D), and is not implemented for 1D.</b> "
                "For a wire, choose its periodic cell axis and a supported solver; "
                "ANALYTIC is offered but may converge slowly.</p>"
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

    def _show_current_settings(self, error=None):
        if self.original is None:
            self.current_settings.value = "<p>Select a structure above to review its cell and suggested settings.</p>"
        elif self.active.value and error:
            self.current_settings.value = (
                '<p style="padding:10px;background:#fff3cd"><b>Submission is blocked:</b> '
                "the selected settings are invalid. Correct them, or click "
                "<b>Undo: restore original settings</b> to return to the original cell "
                "and XYZ/PERIODIC electrostatics.</p>"
            )
        elif self.active.value:
            p = self.settings()
            self.current_settings.value = (
                '<p style="padding:10px;background:#e9f5e9"><b>Currently used for submission:</b> '
                f"periodic axes <b>{html.escape(p['periodic'])}</b>; Poisson solver "
                f"<b>{html.escape(p['poisson_solver'])}</b>; the prepared cell shown below. "
                "The selected settings are applied. Changes to these controls now update "
                "the applied settings. Click <b>Undo: restore original settings</b> to "
                "restore the original cell, atomic positions and XYZ/PERIODIC electrostatics.</p>"
            )
        else:
            self.current_settings.value = (
                '<p style="padding:10px;background:#eef4fb"><b>Currently used for submission:</b> '
                "periodic axes <b>XYZ</b> (periodic along all three cell axes); Poisson solver "
                "<b>PERIODIC</b>; the <b>original cell and atomic positions</b>. "
                "The suggested settings below are <b>not applied</b>. Click "
                "<b>Apply selected settings</b> to use them and the previewed cell. "
                "Until you click that button, changing the fields below does not change "
                "the settings used for submission.</p>"
            )

    def _refresh(self):
        try:
            proposal = self._proposal()
        except (ValueError, KeyError) as exc:
            self._show_current_settings(error=str(exc))
            self.active.disabled = not self.active.value
            self.preview.value = f'<p style="color:#a00">{html.escape(str(exc))}</p>'
            return None
        self.active.disabled = False
        self._show_current_settings()

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
            "<p><b>"
            + (
                "Applied cell"
                if self.active.value
                else "Cell preview — applied only after clicking Apply selected settings"
            )
            + "</b><br>"
            f"Original → prepared cell lengths: <b>{lengths(self.original)} Å → {lengths(proposal)} Å</b>; "
            f"{volume}. Periodic lattice vectors are retained. "
            "Only isolated directions may grow; the atoms are translated rigidly to the centre "
            "in those directions.<br>MT preparation uses twice the atomic span plus 15 Å; "
            "other solvers use the atomic span plus 15 Å. Already larger cells are retained. "
            "This is a geometric starting point: check density tails and vacuum convergence. "
            "Click the button to apply these changes, or undo them to restore the original geometry.</p>"
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
            "Undo: restore original settings"
            if self.active.value
            else "Apply selected settings"
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
