import ipywidgets as ipw
import traitlets as tr
from aiida_nanotech_empa.workflows.cp2k import cp2k_utils
from ase import Atoms

from .analyze_structure import mol_ids_range


class OneColvar(ipw.HBox):
    def __init__(self, cvtype="distance"):
        style = {"description_width": "initial"}
        units = {
            "distance": ["A", "eV/A^2", "40"],
            "angle": ["deg", "ev/deg^2", "2"],
            "bond": ["deg", "ev/deg^2", "2"],
            "torsion": ["deg", "ev/deg^2", "2"],
        }
        self.target_widget = ipw.Text(
            description="", style=style, layout={"width": "140px"}
        )
        self.restraint_widget = ipw.Text(
            description="", style=style, layout={"width": "140px"}
        )

        self.target_html = ipw.HTML(value="Target " + units[cvtype][0], style=style)
        self.restraint_html = ipw.HTML(value="K " + units[cvtype][1], style=style)
        self.restraint_widget.value = units[cvtype][2]

        super().__init__(
            [
                self.target_html,
                self.target_widget,
                self.restraint_html,
                self.restraint_widget,
            ]
        )


class OneConstraint(ipw.HBox):
    def __init__(self, ase_atoms=None):
        self.constraint_widget = ipw.Text(
            description="Constraint",
            style={"description_width": "initial"},
        )
        self.constraint_widget.observe(self._observe_constraint, names="value")
        self.ase_atoms = ase_atoms

        super().__init__([self.constraint_widget])

    def _observe_constraint(self, change):
        try_guess = False
        if "distance" in change["new"]:
            self.children = self.children[:1]
            self.children += (OneColvar(cvtype="distance"),)
            try_guess = True
        elif "angle" in change["new"]:
            self.children = self.children[:1]
            self.children += (OneColvar(cvtype="angle"),)
            try_guess = True
        elif "bond" in change["new"]:
            self.children = self.children[:1]
            self.children += (OneColvar(cvtype="bond"),)
            try_guess = True
        elif "torsion" in change["new"]:
            self.children = self.children[:1]
            self.children += (OneColvar(cvtype="torsion"),)
            try_guess = True
        elif "fixed" in change["new"] and len(self.children) > 1:
            self.children = self.children[:1]
            try_guess = False
        if try_guess and self.ase_atoms is not None:
            try:
                cv_value = cp2k_utils.compute_colvars(change["new"], self.ase_atoms)
                self.children[1].target_widget.value = str(round(cv_value[0][1], 3))
            except IndexError:
                pass


class ConstraintsWidget(ipw.VBox):
    details = tr.Dict()
    ase_atoms = tr.Instance(Atoms, allow_none=True)

    def __init__(self):
        self.constraints = ipw.VBox()

        # Add constraint button.
        self.add_constraint_button = ipw.Button(
            description="Add constraint",
            layout={"width": "initial"},
            button_style="success",
        )
        self.add_constraint_button.on_click(self.add_constraint)

        # Remove constraint button.
        self.remove_constraint_button = ipw.Button(
            description="Remove constraint",
            layout={"width": "initial"},
            button_style="danger",
        )
        self.remove_constraint_button.on_click(self.remove_constraint)

        self.help_checkbox = ipw.Checkbox(
            value=False,
            description="Show help",
            indent=False,
            layout={"width": "initial"},
        )
        self.help_checkbox.observe(self._observe_help, names="value")

        self.help_html = ipw.HTML()

        super().__init__(
            [
                self.constraints,
                ipw.HBox(
                    [
                        self.add_constraint_button,
                        self.remove_constraint_button,
                        self.help_checkbox,
                    ]
                ),
                self.help_html,
            ]
        )

    @tr.observe("details")
    def _observe_manager(self, _=None):
        if self.details and "Slab" in self.details["system_type"]:
            self.add_constraint()

    def add_constraint(self, b=None):
        self.constraints.children += (OneConstraint(ase_atoms=self.ase_atoms),)
        if (
            len(self.constraints.children) == 1
            and self.details
            and "Slab" in self.details["system_type"]
        ):
            to_fix = list(
                self.details["bottom_H"]
                + self.details["slab_layers"][0]
                + self.details["slab_layers"][1]
            )
            self.constraints.children[
                0
            ].constraint_widget.value = "fixed xyz " + mol_ids_range(to_fix)

    def _observe_help(self, change):
        if change["new"]:
            self.help_html.value = """
            <p>
            Example of <b>distance</b> colvar:<br>
            <ul>
                <li>distance atoms 1 3</li>
                <li>distance point fix_point 1.1 2.2 3.3 point atoms 1..6   axis xy</li>
            </ul>
            </p>
            <p>
            Example of <b>angle</b> colvar:<br>
            <ul>
                <li>angle point atoms 2 3  point fix_point 8.36 6.78 5.0 point atoms 3 4</li>
            </ul>
            </p>

            <p>
            Example of <b>torsion</b> colvar:<br>
            four points are mandatory, the dihedral is defined by points 1 2 3 4<br>
            <ul>
                <li>torsion point atoms 8 point atoms 7 point atoms 16   point atoms 25</li>
            </ul>
            </p>

            <p>
            Example of <b>angle_plane_plan</b> colvar:<br>
            <ul>
                <li>angle_plane_plane point fix_point 12.1 7.5  5. point atoms 1..6 point fix_point 7.1 7.5   7. plane atoms 1 2 3 plane vector 0 0 1</li>
            </ul>
            </p>
            <p>
            Example of <b>bond_rotation</b> colvar:<br>
            <ul>
                <li>bond_rotation point fix_point 8.36 6.78 5.8 point fix_point 0 0 1 point atoms 3 point atoms 9</li>
            </ul>
            </p>
            """
        else:
            self.help_html.value = ""

    def remove_constraint(self, b=None):
        self.constraints.children = self.constraints.children[:-1]

    def return_dict(self):
        # 'fixed z 3..18 , collective 1 [ev/angstrom^2] 40 [angstrom] 0.75'
        # distance atoms 1 2
        units = {
            "distance": ["[angstrom]", "[eV/angstrom^2]"],
            "angle": ["[deg]", "[ev/deg^2]"],
            "bond": ["[deg]", "[ev/deg^2]"],
            "torsion": ["[deg]", "[ev/deg^2]"],
        }
        constraints = ""
        colvars = ""
        ncolvars = 0
        for c in self.constraints.children:
            if len(c.children) == 1:
                constraints += " " + c.constraint_widget.value + " ,"
            else:
                ncolvars += 1
                cvtype = c.constraint_widget.value.split()[0].lower().split("_")[0]
                constraints += (
                    " "
                    + "collective "
                    + str(ncolvars)
                    + " "
                    + units[cvtype][1]
                    + " "
                    + c.children[1].restraint_widget.value
                    + " "
                    + units[cvtype][0]
                    + " "
                    + c.children[1].target_widget.value
                    + " ,"
                )
                colvars += " " + c.constraint_widget.value + " ,"
        return {
            "sys_params": {"constraints": constraints[:-1], "colvars": colvars[:-1]}
        }

    def traits_to_link(self):
        return ["details", "ase_atoms"]
