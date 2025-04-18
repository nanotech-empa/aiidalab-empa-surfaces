from functools import reduce

import aiidalab_widgets_base as awb
import ipywidgets as ipw
import numpy as np
import traitlets as tr
from aiida import orm
from ase import Atoms
from IPython.display import clear_output, display

from ..utils.cp2k_input_validity import validate_input
from . import analyze_structure, constraints, stack


def cp2k_bool(value):
    return ".TRUE." if value else ".FALSE."


STYLE = {"description_width": "120px"}
LAYOUT = {"width": "70%"}
LAYOUT2 = {"width": "35%"}


class InputDetails(ipw.VBox):
    structure = tr.Instance(Atoms, allow_none=True)  # needed for colvars
    selected_code = tr.Union([tr.Unicode(), tr.Instance(orm.Code)], allow_none=True)
    details = tr.Dict()
    protocol = tr.Unicode()
    to_fix = tr.List()
    do_cell_opt = tr.Bool()
    uks = tr.Bool()
    net_charge = tr.Int()
    neb = tr.Bool()  # Set by app in case of neb calculation, to be linked to resources.
    replica = tr.Bool()  # Set by app in case of replica chain calculation.
    phonons = tr.Bool()  # Set by app in case of phonons calculation
    n_replica_trait = (
        tr.Int()
    )  # To be linked to resources used only if neb = True or phonons = True
    nproc_replica_trait = (
        tr.Int()
    )  # To be linked from resources to input_details  used only if neb = True or phonons = True
    n_replica_per_group_trait = (
        tr.Int()
    )  # To be linked to resources used only if neb = True

    def __init__(
        self,
    ):
        """
        Arguments:
            sections(list): list of tuples each containing the displayed name of an input section and the
                section object. Each object should containt 'structure' trait pointing to the imported
                structure. The trait will be linked to 'structure' trait of this class.
        """
        # Displaying input sections.
        self.output = ipw.Output()
        self.displayed_sections = []
        # self.neb = False

        self.strusture_analyzer = analyze_structure.StructureAnalyzer()
        tr.dlink((self, "structure"), (self.strusture_analyzer, "structure"))
        tr.dlink((self.strusture_analyzer, "details"), (self, "details"))

        super().__init__(children=[self.output])

    @tr.default("neb")
    def _default_neb(self):
        return False

    @tr.default("phonons")
    def _default_phonons(self):
        return False

    @tr.default("n_replica_trait")
    def _default_n_proc_replica(self):
        if self.neb:
            return 15
        return 1

    @tr.default("n_replica_per_group_trait")
    def _default_n_replica_per_group_trait(self):
        return 1

    @tr.observe("details", "neb", "replica", "phonons")
    def _observe_details(self, _=None):
        self.to_fix = []
        self.net_charge = 0
        self.do_cell_opt = False
        self.uks = False
        with self.output:
            clear_output()
            if self.details:
                sys_type = self.details["system_type"]
                if self.neb:
                    sys_type = "Neb"
                if self.replica:
                    sys_type = "Replica"
                if self.phonons:
                    sys_type = "Phonons"
            else:
                sys_type = "None"

            self.displayed_sections = []
            add_children = []
            for sec in SECTIONS_TO_DISPLAY[sys_type]:
                section = sec()
                if hasattr(section, "traits_to_link"):
                    for trait in section.traits_to_link():
                        tr.link((self, trait), (section, trait))
                self.displayed_sections.append(section)
            display(ipw.VBox(add_children + self.displayed_sections))

    def return_final_dictionary(self):
        final_dictionary = {
            "elements": self.details["all_elements"],
            "system_type": self.details["system_type"],
        }

        # Retrieve all widgets values.
        for section in self.displayed_sections:
            to_add = section.return_dict()
            if to_add:
                for key in to_add.keys():
                    if key in final_dictionary.keys():
                        final_dictionary[key].update(to_add[key])
                    else:
                        final_dictionary[key] = to_add[key]

        # If its a molecule, make non-periodic.
        if (
            self.details["system_type"] == "Molecule"
            and "forceperiodic" not in final_dictionary["dft_params"].keys()
        ):
            final_dictionary["dft_params"]["periodic"] = "NONE"

        # Check input validity.
        can_submit, error_msg = validate_input(self.details, final_dictionary)

        # print(self.final_dictionary)
        return can_submit, error_msg, final_dictionary


class DescriptionWidget(ipw.Text):
    def __init__(self):
        super().__init__(
            description="Process description: ",
            value="",
            placeholder="Type the name here.",
            style={"description_width": "120px"},
            layout={"width": "70%"},
        )

    def return_dict(self):
        return {"description": self.value}


class StructureInfoWidget(ipw.Accordion):
    details = tr.Dict()

    def __init__(self):
        self.info = ipw.Output()

        self.set_title(0, "Structure details")
        super().__init__(selected_index=None)

    @tr.observe("details")
    def _observe_details(self, _=None):
        if self.details is None:
            return
        else:
            self.children = [ipw.VBox([self.info])]
            with self.info:
                clear_output()
                print(self.details["summary"])

    def traits_to_link(self):
        return ["details"]

    def return_dict(self):
        return {}


class VdwSelectorWidget(ipw.Checkbox):
    def __init__(self):
        super().__init__(
            value=True,
            description="Dispersion Corrections",
            tooltip="VDW_POTENTIAL",
            style={"description_width": "120px"},
        )

    def return_dict(self):
        return {"dft_params": {"vdw": self.value}}

    def traits_to_link(self):
        return []


class ForcePeriodicWidget(ipw.Checkbox):
    def __init__(self):
        super().__init__(
            value=False,
            description="Keep periodic",
            # style={"description_width": "120px"},
        )

    def return_dict(self):
        if self.value:
            return {"dft_params": {"forceperiodic": self.value}}
        else:
            return {}

    def traits_to_link(self):
        return []


class ReplicaWidget(ipw.VBox):
    def __init__(self):
        self.restart_from = ipw.Text(
            description="Restart from PK:",
            value="",
            style={"description_width": "120px"},
            layout={"width": "340px"},
        )
        info1 = ipw.HTML(
            value="""If you want to restart from a previous Replica calculation, please enter the PK of the replica calculation.<br>
            Define all CVs using <strong style='color: green;'>'Add constraint'</strong> at least one has to be defined. Check that the automatically provided value for the CVs matches the selected geometry<br>
            Define the final value that each CV should reach<br>
            Define the increment for each CV (e.g. 0.05 to increase a bond length)<br>
            If you do not want a CV to evolve set 0.0 as increment, still adjustments in the CV value may occurr<br>""",
            layout={"width": "70%"},
        )
        self.CVs_targets = ipw.Text(
            description="CVs targets",
            value="",
            style={"description_width": "120px"},
            layout={"width": "70%"},
        )
        self.CVs_increments = ipw.Text(
            description="CVs increments",
            value="",
            style={"description_width": "120px"},
            layout={"width": "70%"},
        )
        super().__init__(
            children=[info1, self.CVs_targets, self.CVs_increments, self.restart_from],
        )

    def return_dict(self):
        the_dict = {}
        if self.restart_from.value:
            the_dict["restart_from"] = orm.load_node(self.restart_from.value).uuid
        the_dict["sys_params"] = {
            "colvars_targets": [float(i) for i in self.CVs_targets.value.split()],
            "colvars_increments": [float(i) for i in self.CVs_increments.value.split()],
        }
        return the_dict

    def traits_to_link(self):
        return []


class NebWidget(ipw.VBox):
    n_replica_trait = tr.Int()
    nproc_replica_trait = tr.Int()
    n_replica_per_group_trait = tr.Int()

    def __init__(self):
        self.restart_from = ipw.Text(
            description="Restart from PK:",
            value="",
            style={"description_width": "150px"},
            layout={"width": "90%"},
        )
        info_restart = ipw.HTML(
            value="""If you want to restart from a previous NEB calculation, please enter the PK of the neb calculation.<br>
            You should then leave empty the replica PKs field.<br>
            In the field replica PKs, please enter the PKs of the replicas you want to use for the NEB calculation.<br>
            You have to <strong>OMIT</strong> the first one that is defined by the selected structure<br>
            <strong>At leat one PK is needed</strong>, defining the final replica of te NEB.<br>""",
            layout={"width": "90%"},
        )
        self.replica_pks = ipw.Text(
            description="Replica PKs:",
            value="",
            style={"description_width": "150px"},
            layout={"width": "90%"},
        )
        self.align_frames = ipw.Checkbox(
            description="Align Frames",
            value=False,
            style={"description_width": "initial"},
            layout={"width": "25%"},
        )
        self.rotate_frames = ipw.Checkbox(
            description="Rotate Frames",
            value=False,
            style={"description_width": "initial"},
            layout={"width": "25%"},
        )
        self.optimize_endpoints = ipw.Checkbox(
            description="Optimize Endpoints",
            value=False,
            style={"description_width": "initial"},
            layout={"width": "25%"},
        )
        self.band_type = ipw.Dropdown(
            options=["CI-NEB"],
            description="Band Type",
            value="CI-NEB",
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )
        self.k_spring = ipw.Text(
            description="Spring constant",
            value="0.05",
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )
        self.nproc_rep = ipw.HTML(
            description="# processors / rep",
            value="324",
            style={"description_width": "initial"},
            layout={"width": "150px"},
        )
        self.n_replica = ipw.IntText(
            description="# of replica",
            value="15",
            style={"description_width": "initial"},
            layout={"width": "150px"},
        )
        self.n_replica_per_group = ipw.Dropdown(
            description="# rep / group",
            options=[1, 3, 5],
            value=1,
            style={"description_width": "initial"},
            layout={"width": "150px"},
        )
        self.nsteps_it = ipw.Text(
            description="Steps before CI",
            value="5",
            style={"description_width": "initial"},
            layout={"width": "150px"},
        )

        self.n_replica.observe(self.on_n_replica_change, "value")
        self.n_replica_per_group.observe(self.on_n_replica_per_group_change, "value")

        super().__init__(
            children=[
                self.restart_from,
                info_restart,
                self.replica_pks,
                ipw.HBox(
                    [self.optimize_endpoints, self.align_frames, self.rotate_frames]
                ),
                ipw.HBox(
                    [
                        ipw.VBox(
                            [
                                self.band_type,
                                self.k_spring,
                            ],
                            layout={"width": "50%"},
                        ),
                        ipw.VBox(
                            [
                                self.nproc_rep,
                                self.n_replica,
                                self.n_replica_per_group,
                                self.nsteps_it,
                            ],
                            layout={"width": "50%"},
                        ),
                    ]
                ),
            ],
        )

    def return_dict(self):
        the_dict = {}
        if self.restart_from.value != "":
            the_dict["restart_from"] = orm.load_node(self.restart_from.value).uuid
        the_dict["neb_params"] = {
            "align_frames": cp2k_bool(self.align_frames.value),
            "rotate_frames": cp2k_bool(self.rotate_frames.value),
            "band_type": self.band_type.value,
            "k_spring": self.k_spring.value,
            "nproc_rep": int(self.nproc_rep.value),
            "number_of_replica": int(self.n_replica.value),
            "nsteps_it": int(self.nsteps_it.value),
            "optimize_end_points": cp2k_bool(self.optimize_endpoints.value),
        }

        the_dict["replica_uuids"] = [
            orm.load_node(int(pk)).uuid for pk in self.replica_pks.value.split()
        ]

        return the_dict

    @tr.observe("nproc_replica_trait")
    def _observe_nproc_replica_trait(self, _=None):
        self.nproc_rep.value = str(self.nproc_replica_trait)

    def on_n_replica_per_group_change(self, _=None):
        self.n_replica_per_group_trait = self.n_replica_per_group.value

    def on_n_replica_change(self, _=None):
        self.n_replica_trait = int(self.n_replica.value)
        nrep = int(self.n_replica.value)
        self.n_replica_per_group.value = 1
        # factors of n_replica
        self.n_replica_per_group.options = set(
            reduce(
                list.__add__,
                ([i, nrep // i] for i in range(1, int(nrep**0.5) + 1) if nrep % i == 0),
            )
        )

    def traits_to_link(self):
        return ["n_replica_trait", "nproc_replica_trait", "n_replica_per_group_trait"]


class PhononsWidget(ipw.VBox):
    details = tr.Dict()
    n_replica_trait = tr.Int()
    nproc_replica_trait = tr.Int()

    def __init__(self):
        self.nproc_rep = ipw.HTML(
            description="# processors / rep",
            value="324",
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )
        self.n_replica = ipw.Dropdown(
            description="# of replica",
            value=3,
            options=[1, 3],
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )

        self.n_replica.observe(self.on_n_replica_change, "value")

        super().__init__(
            children=[
                self.nproc_rep,
                self.n_replica,
            ],
        )

    def return_dict(self):
        return {
            "phonons_params": {
                "nproc_rep": int(self.nproc_rep.value),
            }
        }

    @tr.observe("nproc_replica_trait")
    def _observe_nproc_replica_trait(self, _=None):
        self.nproc_rep.value = str(self.nproc_replica_trait)

    @tr.observe("details")
    def _observe_details(self, _=None):
        self.n_replica.value = 3
        three_times_natoms = self.details["numatoms"] * 3
        self.n_replica.options = set(
            reduce(
                list.__add__,
                (
                    [i, three_times_natoms // i]
                    for i in range(1, int(three_times_natoms**0.5) + 1)
                    if three_times_natoms % i == 0
                ),
            )
        )

    def on_n_replica_change(self, _=None):
        self.n_replica_trait = int(self.n_replica.value)

    def traits_to_link(self):
        return ["n_replica_trait", "nproc_replica_trait", "details"]


class UksSectionWidget(ipw.Accordion):
    details = tr.Dict()
    uks = tr.Bool()
    net_charge = tr.Int()

    def __init__(self, charge_visibility="visible", multiplicity_visibility="visible"):
        self.uks_toggle = ipw.Checkbox(
            value=False,
            description="UKS",
            tooltip="Activate UKS",
            style={"description_width": "initial"},
            layout={"width": "60px"},
        )
        tr.link((self, "uks"), (self.uks_toggle, "value"))

        # Spins
        class OneSpinWidget(stack.HorizontalItemWidget):
            def __init__(self):
                self.selection = ipw.Text(
                    description="Atoms indices:",
                    style={"description_width": "initial"},
                )
                self.starting_magnetization = ipw.IntText(
                    description="Magnetization value:",
                    style={"description_width": "initial"},
                )
                super().__init__(children=[self.selection, self.starting_magnetization])

        self.spins = stack.VerticalStackWidget(
            item_class=OneSpinWidget, add_button_text="Add spin set"
        )

        self.charge = ipw.IntText(
            value=0,
            description="Net charge:",
            style={"description_width": "initial"},
            layout={"width": "120px", "visibility": charge_visibility},
        )
        tr.link((self, "net_charge"), (self.charge, "value"))

        self.multiplicity = ipw.IntText(
            value=1,
            description="Multiplicity:",
            style={"description_width": "initial"},
            layout={"width": "140px", "visibility": multiplicity_visibility},
        )

        self.set_title(0, "Spin-polarized calculation")

        self.uks_box = [
            ipw.VBox(
                [
                    ipw.HBox(
                        [
                            self.uks_toggle,
                            self.charge,
                            self.multiplicity,
                        ]
                    ),
                    self.spins,
                ]
            )
        ]
        self.no_uks_box = [ipw.HBox([self.uks_toggle, self.charge])]

        super().__init__(selected_index=None)

        self.children = self.no_uks_box

    def return_dict(self):
        to_return = {
            "uks": self.uks,
            "charge": self.net_charge,
        }

        if self.uks:
            magnetization_per_site = np.zeros(self.details["numatoms"])
            for spinset in self.spins.items:
                magnetization_per_site[
                    awb.utils.string_range_to_list(spinset.selection.value)[0]
                ] = spinset.starting_magnetization.value
            to_return.update(
                {
                    "multiplicity": self.multiplicity.value,
                    "magnetization_per_site": magnetization_per_site.astype(
                        np.int32
                    ).tolist(),
                }
            )
        return {"dft_params": to_return}

    @tr.observe("uks")
    def _observe_uks(self, value=None):
        self.children = self.uks_box if value["new"] else self.no_uks_box

    def traits_to_link(self):
        return ["details", "uks", "net_charge"]


class CellSectionWidget(ipw.Accordion):
    details = tr.Dict()
    do_cell_opt = tr.Bool()

    def __init__(self):
        self.cell_symmetry = ipw.Dropdown(
            description="Cell symmetry:",
            options=[
                "CUBIC",
                "HEXAGONL",
                "MONOCLINIC",
                "NONE",
                "ORTHORHOMBIC",
                "RHOMBOHEDRAL",
                "TETRAGONAL_AB",
                "TETRAGONAL_AC",
                "TETRAGONAL_BC",
                "TRICLINIC",
            ],
            value="ORTHORHOMBIC",
            style=STYLE,
        )

        self.cell_constraint = ipw.Dropdown(
            description="Cell constraints:",
            options=["XYZ", "NONE", "X", "XY", "XZ", "Y", "YZ", "Z"],
            value="NONE",
            style=STYLE,
        )

        self.cell_freedom = ipw.Dropdown(
            options=["FREE", "KEEP_SYMMETRY", "KEEP_ANGLES", "KEEP_SPACE_GROUP"],
            description="Cell freedom",
            value="KEEP_SYMMETRY",
            style=STYLE,
        )

        self.opt_cell = ipw.Checkbox(
            value=False,
            description="Optimize cell",
        )
        tr.link((self, "do_cell_opt"), (self.opt_cell, "value"))

        self.set_title(0, "Cell optimization")

        super().__init__(
            selected_index=None,
            children=[
                ipw.VBox(
                    [
                        self.cell_symmetry,
                        self.cell_freedom,
                        self.cell_constraint,
                        self.opt_cell,
                    ]
                )
            ],
        )

    def return_dict(self):
        sys_params = {"symmetry": self.cell_symmetry.value}
        if self.opt_cell.value:
            sys_params["cell_opt"] = ""
        if self.cell_constraint.value != "NONE":
            sys_params["cell_opt_constraint"] = self.cell_constraint.value
        if self.cell_freedom.value != "FREE":
            sys_params[self.cell_freedom.value.lower()] = ""

        return {"sys_params": sys_params}

    def traits_to_link(self):
        return ["details", "do_cell_opt"]


class DiagonalisationSmearingWidget(ipw.HBox):
    def __init__(self, **kwargs):
        self.enable_diagonalisation = ipw.Checkbox(
            value=False,
            description="Self-consistent diagonalisation",
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )
        self.enable_diagonalisation.observe(
            self._observe_enable_diagonalisation, "value"
        )
        self.enable_diagonalisation.observe(self.enable_or_disable_widgets, "value")

        self.enable_smearing = ipw.ToggleButton(
            value=False,
            description="Enable Fermi-Dirac smearing",
            style={"description_width": "initial"},
            layout={"width": "450px"},
        )
        self.enable_smearing.observe(self.enable_or_disable_widgets, "value")

        self.smearing_temperature = ipw.FloatText(
            value=150.0,
            description="Temperature [K]",
            disabled=True,
            style={"description_width": "initial"},
            layout={"width": "200px"},
        )
        self.force_multiplicity = ipw.Checkbox(
            value=True, description="Force multiplicity", disabled=True
        )
        self.smearing_box = ipw.VBox(
            children=[
                self.enable_smearing,
                ipw.HBox(children=[self.smearing_temperature, self.force_multiplicity]),
            ]
        )

        super().__init__(
            children=[
                self.enable_diagonalisation,
            ],
            **kwargs,
        )

    def _observe_enable_diagonalisation(self, _=None):
        if self.enable_diagonalisation.value:
            self.children = [
                self.enable_diagonalisation,
                self.smearing_box,
            ]
        else:
            self.children = [self.enable_diagonalisation]

    def enable_or_disable_widgets(self, _=None):
        self.enable_smearing.disabled = not self.enable_diagonalisation.value
        self.smearing_temperature.disabled = not self.enable_smearing.value
        self.force_multiplicity.disabled = not self.enable_smearing.value

    @property
    def smearing_enabled(self):
        return self.enable_diagonalisation and self.enable_smearing.value


SECTIONS_TO_DISPLAY = {
    "None": [],
    "Wire": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
    ],
    "Bulk": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
        CellSectionWidget,
    ],
    "SlabXY": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
    ],
    "SlabYZ": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
    ],
    "SlabXZ": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
    ],
    "Molecule": [
        StructureInfoWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        constraints.ConstraintsWidget,
        ForcePeriodicWidget,
    ],
    "Replica": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
        ReplicaWidget,
    ],
    "Neb": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
        NebWidget,
    ],
    "Phonons": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
        PhononsWidget,
    ],
}
