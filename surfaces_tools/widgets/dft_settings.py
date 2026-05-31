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
            description="HF max memory:",
            style={"description_width": "initial"},
            layout={"width": "260px"},
        )
        self.hfx_box = ipw.VBox([])
        self._previous_functional = self.xc_functional.value
        self.xc_functional.observe(self._update_functional_options, "value")
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
                self.hfx_box,
            ]
        )
        self._update_functional_options()

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
                )
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

        if self.xc_functional.value == "PBE0":
            params["hfx_fraction"] = self.hfx_fraction.value
            params["hfx_cutoff_radius"] = self.hfx_cutoff_radius.value
            params["hfx_max_memory"] = self.hfx_max_memory.value
        return params

    def update_dft_params(self, dft_params):
        dft_params.update(self.get_dft_params())
        return dft_params
