import json

import ipywidgets as ipw


DEFAULTS = {
    "PBE": {
        "basis_set_file_names": "BASIS_MOLOPT",
        "potential_file_name": "POTENTIAL",
        "basis_set_key": "basis_set",
        "potential_key": "pseudopotential",
        "aux_basis_set_key": "",
    },
    "PBE0": {
        "basis_set_file_names": "BASIS_MOLOPT_UZH, BASIS_ADMM_UZH",
        "potential_file_name": "POTENTIAL_UZH",
        "basis_set_key": "pbe0_basis_set",
        "potential_key": "pbe0_pseudopotential",
        "aux_basis_set_key": "pbe0_aux_basis_set",
    },
    "xc_functional": "PBE",
    "hfx_fraction": 0.25,
    "hfx_cutoff_radius": 10.0,
    "hfx_max_memory": 80000,
    "ot_minimizer": "CG",
    "admm_purification_method": "NONE",
}


def _parse_mapping(text, label):
    text = text.strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'{label} must be valid JSON, for example {{"C": "TZVP..."}}'
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


class DftSettingsWidget(ipw.VBox):
    def __init__(self):
        self.xc_functional = ipw.Dropdown(
            value=DEFAULTS["xc_functional"],
            options=[("PBE", "PBE"), ("PBE0", "PBE0")],
            description="Functional:",
            style={"description_width": "initial"},
        )
        self.basis_set_file_names = ipw.Text(
            value=DEFAULTS["PBE"]["basis_set_file_names"],
            description="Basis files:",
            placeholder="BASIS_MOLOPT or BASIS_MOLOPT_UZH, BASIS_ADMM_UZH",
            style={"description_width": "initial"},
            layout={"width": "70%"},
        )
        self.potential_file_name = ipw.Text(
            value=DEFAULTS["PBE"]["potential_file_name"],
            description="Potential file:",
            style={"description_width": "initial"},
            layout={"width": "320px"},
        )
        self.basis_set_key = ipw.Text(
            value=DEFAULTS["PBE"]["basis_set_key"],
            description="Basis map:",
            style={"description_width": "initial"},
            layout={"width": "320px"},
        )
        self.potential_key = ipw.Text(
            value=DEFAULTS["PBE"]["potential_key"],
            description="Potential map:",
            style={"description_width": "initial"},
            layout={"width": "320px"},
        )
        self.aux_basis_set_key = ipw.Text(
            value=DEFAULTS["PBE"]["aux_basis_set_key"],
            description="Aux basis map:",
            placeholder="Optional atomic_kinds.yml key for BASIS_SET RI_AUX",
            style={"description_width": "initial"},
            layout={"width": "420px"},
        )
        self.basis_set_overrides = ipw.Textarea(
            value="",
            description="Basis overrides:",
            placeholder='Optional JSON, e.g. {"B": "...", "N": "..."}',
            rows=2,
            style={"description_width": "initial"},
            layout={"width": "70%"},
        )
        self.aux_basis_set_overrides = ipw.Textarea(
            value="",
            description="Aux overrides:",
            placeholder='Optional JSON, e.g. {"B": "...", "N": "..."}',
            rows=2,
            style={"description_width": "initial"},
            layout={"width": "70%"},
        )
        self.potential_overrides = ipw.Textarea(
            value="",
            description="Potential overrides:",
            placeholder='Optional JSON, e.g. {"B": "GTH-PBE-q3"}',
            rows=2,
            style={"description_width": "initial"},
            layout={"width": "70%"},
        )
        self.hfx_fraction = ipw.FloatText(
            value=DEFAULTS["hfx_fraction"],
            description="HF fraction:",
            style={"description_width": "initial"},
            layout={"width": "220px"},
        )
        self.hfx_cutoff_radius = ipw.FloatText(
            value=DEFAULTS["hfx_cutoff_radius"],
            description="HF cutoff radius:",
            style={"description_width": "initial"},
            layout={"width": "260px"},
        )
        self.hfx_max_memory = ipw.IntText(
            value=DEFAULTS["hfx_max_memory"],
            description="HF max memory [MiB/rank]:",
            style={"description_width": "initial"},
            layout={"width": "320px"},
        )
        self.ot_minimizer = ipw.Dropdown(
            value=DEFAULTS["ot_minimizer"],
            options=[("CG", "CG"), ("DIIS", "DIIS")],
            description="OT minimizer:",
            style={"description_width": "initial"},
            layout={"width": "220px"},
        )
        self.admm_purification_method = ipw.Dropdown(
            value=DEFAULTS["admm_purification_method"],
            options=[("NONE", "NONE"), ("MO_DIAG", "MO_DIAG")],
            description="ADMM purification:",
            style={"description_width": "initial"},
            layout={"width": "320px"},
        )
        self.show_info = ipw.Checkbox(
            value=False,
            description="Show DFT field help",
            indent=False,
            style={"description_width": "initial"},
        )
        self.info_box = ipw.HTML(layout={"width": "90%"})
        self.hfx_box = ipw.VBox([])
        self._previous_functional = self.xc_functional.value
        self.xc_functional.observe(self._update_functional_options, "value")
        self.show_info.observe(self._update_info_box, "value")
        super().__init__(
            [
                ipw.HBox([self.xc_functional]),
                self.basis_set_file_names,
                ipw.HBox(
                    [self.potential_file_name, self.basis_set_key, self.potential_key]
                ),
                self.aux_basis_set_key,
                self.basis_set_overrides,
                self.aux_basis_set_overrides,
                self.potential_overrides,
                self.show_info,
                self.info_box,
                ipw.HBox([self.ot_minimizer]),
                self.hfx_box,
            ]
        )
        self._update_functional_options()
        self._update_info_box()

    def _update_info_box(self, _=None):
        if not self.show_info.value:
            self.info_box.value = ""
            return
        self.info_box.value = """
        <div style='line-height:1.45; max-width: 1000px;'>
          <strong>Basis and potential fields.</strong>
          <code>Basis files</code> is a comma-separated list of CP2K basis files.
          <code>Potential file</code> is the CP2K pseudopotential file.
          <code>Basis map</code>, <code>Potential map</code>, and <code>Aux basis map</code>
          are keys in <code>atomic_kinds.yml</code> used to select the default value for each element.<br>
          <strong>Override fields.</strong>
          The basis, aux-basis, and potential override boxes expect JSON objects mapping element symbols to CP2K names,
          for example <code>{&quot;B&quot;: &quot;DZVP-MOLOPT-SR-GTH&quot;, &quot;N&quot;: &quot;DZVP-MOLOPT-SR-GTH&quot;}</code>.
          These values override the map-derived defaults only for the listed elements.<br>
          <strong>PBE0 fields.</strong>
          <code>HF fraction</code> is the exact-exchange fraction.
          <code>HF cutoff radius</code> is the truncated Coulomb cutoff radius in Angstrom.
          <code>HF max memory</code> is CP2K HF memory in MiB per MPI process/rank.
          <code>ADMM purification</code> controls the CP2K ADMM purification method; <code>NONE</code> is the default for diagonalization workflows with added MOs.
        </div>
        """

    def _apply_functional_defaults(self, functional):
        defaults = DEFAULTS[functional]
        self.basis_set_file_names.value = defaults["basis_set_file_names"]
        self.potential_file_name.value = defaults["potential_file_name"]
        self.basis_set_key.value = defaults["basis_set_key"]
        self.potential_key.value = defaults["potential_key"]
        self.aux_basis_set_key.value = defaults["aux_basis_set_key"]

    def _update_functional_options(self, change=None):
        if change is not None:
            old_defaults = DEFAULTS[self._previous_functional]
            current_values = {
                "basis_set_file_names": self.basis_set_file_names.value,
                "potential_file_name": self.potential_file_name.value,
                "basis_set_key": self.basis_set_key.value,
                "potential_key": self.potential_key.value,
                "aux_basis_set_key": self.aux_basis_set_key.value,
            }
            if all(current_values[key] == old_defaults[key] for key in old_defaults):
                self._apply_functional_defaults(self.xc_functional.value)
            self._previous_functional = self.xc_functional.value

        self.hfx_box.children = (
            [
                ipw.HBox(
                    [self.hfx_fraction, self.hfx_cutoff_radius, self.hfx_max_memory]
                ),
                ipw.HBox([self.admm_purification_method]),
            ]
            if self.xc_functional.value == "PBE0"
            else []
        )

    def get_dft_params(self):
        params = {}
        if self.xc_functional.value != DEFAULTS["xc_functional"]:
            params["xc_functional"] = self.xc_functional.value

        basis_files = [
            item.strip()
            for item in self.basis_set_file_names.value.replace(";", ",").split(",")
            if item.strip()
        ]
        defaults = DEFAULTS[self.xc_functional.value]
        if ", ".join(basis_files) != defaults["basis_set_file_names"]:
            params["basis_set_file_names"] = basis_files
        if self.potential_file_name.value.strip() != defaults["potential_file_name"]:
            params["potential_file_name"] = self.potential_file_name.value.strip()
        if self.basis_set_key.value.strip() != defaults["basis_set_key"]:
            params["basis_set_key"] = self.basis_set_key.value.strip()
        if self.potential_key.value.strip() != defaults["potential_key"]:
            params["potential_key"] = self.potential_key.value.strip()
        if self.aux_basis_set_key.value.strip() != defaults["aux_basis_set_key"]:
            params["aux_basis_set_key"] = self.aux_basis_set_key.value.strip()

        for widget, key, label in (
            (self.basis_set_overrides, "basis_set_overrides", "Basis overrides"),
            (self.aux_basis_set_overrides, "aux_basis_set_overrides", "Aux overrides"),
            (self.potential_overrides, "potential_overrides", "Potential overrides"),
        ):
            parsed = _parse_mapping(widget.value, label)
            if parsed:
                params[key] = parsed

        if self.ot_minimizer.value != DEFAULTS["ot_minimizer"]:
            params["ot_minimizer"] = self.ot_minimizer.value

        if self.xc_functional.value == "PBE0":
            params["hfx_fraction"] = self.hfx_fraction.value
            params["hfx_cutoff_radius"] = self.hfx_cutoff_radius.value
            params["hfx_max_memory"] = self.hfx_max_memory.value
            params["admm_purification_method"] = self.admm_purification_method.value
        return params

    def update_dft_params(self, dft_params):
        dft_params.update(self.get_dft_params())
        return dft_params
