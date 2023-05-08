import ipywidgets as ipw
import numpy as np
from IPython.display import clear_output, display

style = {"description_width": "120px"}
layout = {"width": "70%"}


class DistanceCV:
    def __init__(self, no_widget=False):

        self.spring_unit = "eV/angstrom^2"
        self.target_unit = "angstrom"

        self.text_colvar_atoms = None
        self.widget = None

        if not no_widget:
            self._create_widget()

        self.input_received = False

        # Atom list (cp2k index convention: starts from 1).
        self.a_list = None

    @classmethod
    def from_cp2k_subsys(cls, cp2k_subsys):
        atoms_list = [int(x) for x in cp2k_subsys["DISTANCE"]["ATOMS"].split()]
        cv = cls(no_widget=True)
        cv.a_list = atoms_list
        cv.input_received = True
        return cv

    def _create_widget(self):
        self.text_colvar_atoms = ipw.Text(
            placeholder="1 2",
            description="Colvar Atoms",
            style=style,
            layout={"width": "60%"},
        )
        self.widget = ipw.VBox([self.text_colvar_atoms])

    def read_and_validate_inputs(self):
        try:
            a_list = [int(x) for x in self.text_colvar_atoms.value.split()]
        except Exception:
            raise OSError("Error: wrong input for distance cv.")
        if len(a_list) != 2:
            raise OSError("Error: distance cv not two atoms.")
        self.a_list = a_list

        self.input_received = True

    def eval_cv(self, atoms):
        return np.linalg.norm(
            atoms[self.a_list[0] - 1].position - atoms[self.a_list[1] - 1].position
        )

    def print_cv(self, atoms):
        print("Distance between the atoms:")
        print(self.eval_cv(atoms))

    def visualization_list(self, atoms=None):
        if self.a_list is None:
            return []
        # index starts from 0
        return [i_a - 1 for i_a in self.a_list]

    def cp2k_subsys_inp(self):
        cp2k_subsys = {"DISTANCE": {"ATOMS": self.text_colvar_atoms.value}}
        return cp2k_subsys


class AnglePlanePlaneCV:
    def __init__(self, no_widget=False):

        self.spring_unit = "eV/deg^2"
        self.target_unit = "deg"

        self.text_plane1_def = None
        self.toggle_plane2_type = None
        self.text_plane2_def = None
        self.widget = None

        if not no_widget:
            self._create_widget()

        self.input_received = False

        # Definition lists for plane 1 and 2.
        # Includes either 3 atom indexes or vector xyz coordinates.
        self.p1_def = None
        self.p2_def_type = None
        self.p2_def = None

    @classmethod
    def from_cp2k_subsys(cls, cp2k_subsys):

        cv = cls(no_widget=True)
        subsys_app = cp2k_subsys["ANGLE_PLANE_PLANE"]
        cv.p1_def = np.array([int(x) for x in subsys_app["PLANE"]["ATOMS"].split()])
        cv.p2_def_type = subsys_app["PLANE  "]["DEF_TYPE"]
        if cv.p2_def_type == "ATOMS":
            cv.p2_def = np.array(
                [int(x) for x in subsys_app["PLANE  "]["ATOMS"].split()]
            )
        else:
            cv.p2_def = np.array(
                [float(x) for x in subsys_app["PLANE  "]["NORMAL_VECTOR"].split()]
            )

        return cv

    def _create_widget(self):
        self.text_plane1_def = ipw.Text(
            placeholder="1 2 3", description="Plane 1 atoms", style=style, layout=layout
        )

        def on_plane2_type(c):
            self.text_plane2_def.description = self.toggle_plane2_type.value

        self.toggle_plane2_type = ipw.ToggleButtons(
            options=["ATOMS", "VECTOR"],
            description="Plane 2 definition",
            style=style,
            layout=layout,
        )
        self.toggle_plane2_type.observe(on_plane2_type, "value")

        self.text_plane2_def = ipw.Text(
            placeholder="1 2 3", description="Atoms", style=style, layout=layout
        )

        self.widget = ipw.VBox(
            [self.text_plane1_def, self.toggle_plane2_type, self.text_plane2_def]
        )

    def read_and_validate_inputs(self):
        try:
            self.p1_def = np.array([int(x) for x in self.text_plane1_def.value.split()])
        except Exception:
            raise OSError("Error: wrong input for plane 1 definition.")
        if len(self.p1_def) != 3:
            raise OSError("Error: plane 1 needs 3 atoms.")

        self.p2_def_type = self.toggle_plane2_type.value

        if self.p2_def_type == "ATOMS":
            try:
                self.p2_def = np.array(
                    [int(x) for x in self.text_plane2_def.value.split()]
                )
            except Exception:
                raise OSError("Error: wrong input for plane 2 definition.")
            if len(self.p2_def) != 3:
                raise OSError("Error: plane 2 needs 3 atoms.")
        else:
            try:
                self.p2_def = np.array(
                    [float(x) for x in self.text_plane2_def.value.split()]
                )
            except Exception:
                raise OSError("Error: wrong input for plane 2 definition.")
            if len(self.p2_def) != 3:
                raise OSError("Error: plane 2 normal needs 3 coordinates.")

        self.input_received = True

    def _cp2k_plane_normal(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        vec = np.cross(v1, v2)
        return vec / np.linalg.norm(vec)

    def _p1_normal(self, atoms):
        # NB until here, the indexes use cp2k convention (starts from 1)
        return self._cp2k_plane_normal(
            atoms[self.p1_def[0] - 1].position,
            atoms[self.p1_def[1] - 1].position,
            atoms[self.p1_def[2] - 1].position,
        )

    def _p2_normal(self, atoms):
        # NB until here, the indexes use cp2k convention (starts from 1)
        if self.p2_def_type == "ATOMS":
            return self._cp2k_plane_normal(
                atoms[self.p2_def[0] - 1].position,
                atoms[self.p2_def[1] - 1].position,
                atoms[self.p2_def[2] - 1].position,
            )
        else:
            return self.p2_def / np.linalg.norm(self.p2_def)

    def eval_cv(self, atoms):
        """
        Evaluates the ANGLE PLANE PLANE collective variable according to cp2k conventions
        In case of defining 3 atoms a1, a2 and a3, the plane normal is defined as
        (a1 - a2) x (a3 - a2)
        """
        cosine = np.dot(self._p1_normal(atoms), self._p2_normal(atoms))
        angle = np.arccos(cosine) * 180.0 / np.pi
        return angle

    def print_cv(self, atoms):
        print("Angle between the planes:")
        print(self.eval_cv(atoms))

    def visualization_list(self, atoms):

        if self.p1_def is None:
            return []

        # atom indexes start from 0
        vis_list = list(self.p1_def - 1)

        # add middle point of p1 and a point along the normal
        p1_middle = np.mean(atoms[list(self.p1_def - 1)].positions, axis=0)
        vis_list.append(p1_middle)
        vis_list.append(p1_middle + 3.0 * self._p1_normal(atoms))

        if self.p2_def_type == "ATOMS":
            vis_list.extend(list(self.p2_def - 1))
            p2_middle = np.mean(atoms[list(self.p2_def - 1)].positions, axis=0)
            vis_list.append(p2_middle)
            vis_list.append(p2_middle + 3.0 * self._p2_normal(atoms))
        else:
            vis_list.append(p1_middle + 3.0 * self._p2_normal(atoms))

        return vis_list

    def cp2k_subsys_inp(self):

        repl = {"ATOMS": "ATOMS", "VECTOR": "NORMAL_VECTOR"}

        cp2k_subsys = {
            "ANGLE_PLANE_PLANE": {
                "PLANE": {
                    "DEF_TYPE": "ATOMS",
                    "ATOMS": " ".join([str(x) for x in self.p1_def]),
                },
                "PLANE  ": {
                    "DEF_TYPE": self.p2_def_type,
                    repl[self.p2_def_type]: " ".join([str(x) for x in self.p2_def]),
                },
            }
        }

        return cp2k_subsys


class BondRotationCV:
    def __init__(self, no_widget=False):

        self.spring_unit = "eV/deg^2"
        self.target_unit = "deg"

        self.bond_point_texts = None
        self.bond_point_btns = None
        self.bond_point_textbs = None
        self.widget = None

        if not no_widget:
            self._create_widget()

        self.input_received = False

        self.types_list = None
        self.data_txt_list = None
        self.data_list = None  # atom indexes start from 0

    @classmethod
    def from_cp2k_subsys(cls, cp2k_subsys):
        cv = cls(no_widget=True)
        point_list = cp2k_subsys["BOND_ROTATION"]["POINT"]

        cv.types_list = []
        cv.data_list = []

        for p in point_list:
            cv.types_list.append(p["TYPE"])
            if p["TYPE"] == "GEO_CENTER":
                cv.data_list.append([int(x) - 1 for x in p["ATOMS"].split()])
            else:
                cv.data_list.append([float(x) for x in p["XYZ"].split()])

        cv.input_received = True
        return cv

    def _create_widget(self):

        self.bond_point_texts = [
            "1st point 1st line",
            "2nd point 1st line",
            "1st point 2nd line",
            "2nd point 2nd line",
        ]

        self.bond_point_btns = []
        self.bond_point_textbs = []

        self.textbox_defaults = {
            "GEO_CENTER": ("1 2 3", "atom index(es)"),
            "FIX_POINT": ("18.20 22.15 20.10", "position in Angstrom"),
        }

        def on_bond_point_type_toggle(c, tb):
            tb.placeholder = self.textbox_defaults[c.new][0]
            tb.description = self.textbox_defaults[c.new][1]

        for bond_point_text in self.bond_point_texts:

            toggle_button = ipw.ToggleButtons(
                options=["GEO_CENTER", "FIX_POINT"],
                description=bond_point_text,
                style=style,
                layout=layout,
            )

            textbox = ipw.Text(
                placeholder=self.textbox_defaults[toggle_button.value][0],
                description=self.textbox_defaults[toggle_button.value][1],
                style=style,
                layout=layout,
            )

            toggle_button.observe(
                lambda c, tb=textbox: on_bond_point_type_toggle(c, tb), names="value"
            )

            self.bond_point_btns.append(toggle_button)
            self.bond_point_textbs.append(textbox)

        self.widget = ipw.VBox(
            [x for ab in zip(self.bond_point_btns, self.bond_point_textbs) for x in ab]
        )

    def read_and_validate_inputs(self):

        self.types_list = []
        self.data_txt_list = []
        self.data_list = []

        for i_p, btn in enumerate(self.bond_point_btns):
            typ = btn.value
            self.types_list.append(typ)

            if typ == "GEO_CENTER":
                try:
                    dl = np.array(
                        [int(x) - 1 for x in self.bond_point_textbs[i_p].value.split()]
                    )
                except Exception:
                    raise OSError(
                        f"Error: wrong input for '{self.bond_point_texts[i_p]}'"
                    )
            else:
                try:
                    dl = np.array(
                        [float(x) for x in self.bond_point_textbs[i_p].value.split()]
                    )
                except Exception:
                    raise OSError(
                        f"Error: wrong input for '{self.bond_point_texts[i_p]}'"
                    )
                if len(dl) != 3:
                    raise OSError(f"Error: '{self.bond_point_texts[i_p]}' needs x,y,z")
            self.data_list.append(dl)
            self.data_txt_list.append(self.bond_point_textbs[i_p].value)

        self.input_received = True

    def _point_list(self, atoms):
        p_list = []
        for typ, d in zip(self.types_list, self.data_list):
            if typ == "GEO_CENTER":
                p_list.append(np.mean(atoms[d].positions, axis=0))
            else:
                p_list.append(d)
        return p_list

    def eval_cv(self, atoms):
        p_list = np.array(self._point_list(atoms))
        v1 = p_list[1] - p_list[0]
        v2 = p_list[3] - p_list[2]

        cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cosine) * 180.0 / np.pi
        return angle

    def print_cv(self, atoms):
        print("Angle between the two lines:")
        print(self.eval_cv(atoms))

    def visualization_list(self, atoms):
        return self._point_list(atoms)

    def cp2k_subsys_inp(self):

        repl = {"GEO_CENTER": "ATOMS", "FIX_POINT": "XYZ"}

        point_list = [
            {"TYPE": typ, repl[typ]: text}
            for typ, text in zip(self.types_list, self.data_txt_list)
        ]

        cp2k_subsys = {
            "BOND_ROTATION": {
                "POINT": point_list,
                "P1_BOND1": "1",
                "P2_BOND1": "2",
                "P1_BOND2": "3",
                "P2_BOND2": "4",
            }
        }
        return cp2k_subsys


class CollectiveVariableWidget(ipw.VBox):
    def __init__(self, viewer_widget=None, **kwargs):

        self.current_cv_instance = None

        def on_choose_colvar(c):
            self.current_cv_instance = self.drop_colvar_type.value()
            with self.out_colvar:
                clear_output()
                display(self.current_cv_instance.widget)

        self.drop_colvar_type = ipw.Select(
            options=COLVARS, description="Colvar Type", style=style, layout=layout
        )

        self.drop_colvar_type.observe(on_choose_colvar, "value")

        self.out_colvar = ipw.Output(layout={"border": "1px solid #ccc"})

        self.text_colvar_targets = ipw.Text(
            placeholder="0.9 1.3 1.7 2.4",
            description="Colvar Targets",
            style=style,
            layout=layout,
        )

        self.visualize_colvar_btn = ipw.Button(
            description="Visualize CV", style=style, layout=layout
        )

        self.float_spring = ipw.FloatText(
            description="Spring constant", value=30.0, style=style, layout=layout
        )

        self.error_out = ipw.Output()

        on_choose_colvar("")

        children = [
            self.drop_colvar_type,
            self.out_colvar,
            self.visualize_colvar_btn,
            self.text_colvar_targets,
            self.float_spring,
            self.error_out,
        ]
        super().__init__(children=children, **kwargs)

    def validation_check(self):
        try:
            self.current_cv_instance.read_and_validate_inputs()
        except Exception as e:
            with self.error_out:
                print(e.message)
            return False
        return True

    def set_job_details(self, job_details):
        if not self.validation_check():
            return False

        # TODO also check the inputs below

        job_details["colvar_targets"] = self.text_colvar_targets.value
        job_details["spring"] = self.float_spring.value
        job_details["spring_unit"] = self.current_cv_instance.spring_unit
        job_details["target_unit"] = self.current_cv_instance.target_unit
        job_details["subsys_colvar"] = self.current_cv_instance.cp2k_subsys_inp()

        return True

    @property
    def colvar_targets(self):
        return self.text_colvar_targets.value

    @property
    def target_unit(self):
        return self.current_cv_instance.target_unit

    @property
    def spring_constant(self):
        return self.float_spring.value

    @property
    def spring_unit(self):
        return self.current_cv_instance.spring_unit

    @property
    def subsys_colvar(self):
        return self.current_cv_instance.cp2k_subsys_inp()


COLVARS = {
    "DISTANCE": DistanceCV,
    "ANGLE_PLANE_PLANE": AnglePlanePlaneCV,
    "BOND_ROTATION": BondRotationCV,
}
