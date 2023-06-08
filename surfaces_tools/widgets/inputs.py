from functools import reduce

import aiidalab_widgets_base as awb
import ipywidgets as ipw
import numpy as np
import traitlets as tr
from aiida import orm
from ase import Atoms
from IPython.display import clear_output, display

from ..utils.cp2k_input_validity import validate_input
from .constraints import ConstraintsWidget
from .spins import SpinsWidget

STYLE = {"description_width": "120px"}
LAYOUT = {"width": "70%"}
LAYOUT2 = {"width": "35%"}


class InputDetails(ipw.VBox):
    selected_code = tr.Union([tr.Unicode(), tr.Instance(orm.Code)], allow_none=True)
    details = tr.Dict()
    protocol = tr.Unicode()
    final_dictionary = tr.Dict()
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

    ase_atoms = tr.Instance(Atoms, allow_none=True)  # needed for colvars

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
        tmp_dict = {}

        # PUT LIST OF ELEMENTS IN DICTIONARY
        tmp_dict["elements"] = self.details["all_elements"]

        # RETRIEVE ALL WIDGET VALUES
        for section in self.displayed_sections:
            to_add = section.return_dict()
            if to_add:
                for key in to_add.keys():
                    if key in tmp_dict.keys():
                        tmp_dict[key].update(to_add[key])
                    else:
                        tmp_dict[key] = to_add[key]

        # System type

        tmp_dict["system_type"] = self.details["system_type"]

        # Molecule
        if self.details["system_type"] == "Molecule":
            tmp_dict["dft_params"]["periodic"] = "NONE"

        # CHECK input validity
        can_submit, error_msg = validate_input(self.details, tmp_dict)

        # CREATE PLAIN INPUT
        if can_submit:
            self.final_dictionary = tmp_dict

        # print(self.final_dictionary)
        # RETURN DICT of widgets details
        return can_submit, error_msg, self.final_dictionary


class DescriptionWidget(ipw.Text):
    # DESCRIPTION OF CALCULATION
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


class VdwSelectorWidget(ipw.ToggleButton):
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


class ReplicaWidget(ipw.VBox):
    def __init__(self):
        self.restart_from = ipw.Text(
            description="Restart from PK:",
            value="",
            style={"description_width": "initial"},
            layout={"width": "340px"},
        )
        self.CVs_targets = ipw.Text(
            description="CVs targets",
            value="",
            style={"description_width": "initial"},
            layout={"width": "540px"},
        )
        self.CVs_increments = ipw.Text(
            description="CVs increments",
            value="",
            style={"description_width": "initial"},
            layout={"width": "540px"},
        )
        super().__init__(
            children=[self.restart_from, self.CVs_targets, self.CVs_increments],
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
            style={"description_width": "initial"},
            layout={"width": "540px"},
        )
        self.replica_pks = ipw.Text(
            description="Replica PKs:",
            value="",
            style={"description_width": "initial"},
            layout={"width": "540px"},
        )
        self.align_frames = ipw.ToggleButton(
            description="Align Frames",
            value=False,
            style={"description_width": "initial"},
            layout={"width": "140px"},
        )
        self.optimize_endpoints = ipw.ToggleButton(
            description="Optimize Endpoints",
            value=False,
            style={"description_width": "initial"},
            layout={"width": "140px"},
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
            layout={"width": "240px"},
        )
        self.n_replica = ipw.IntText(
            description="# of replica",
            value="15",
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )
        self.n_replica_per_group = ipw.Dropdown(
            description="# rep / group",
            options=[1, 3, 5],
            value=1,
            style={"description_width": "initial"},
        )
        self.nsteps_it = ipw.Text(
            description="Steps before CI",
            value="5",
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )

        self.n_replica.observe(self.on_n_replica_change, "value")
        self.n_replica_per_group.observe(self.on_n_replica_per_group_change, "value")

        super().__init__(
            children=[
                self.restart_from,
                self.replica_pks,
                self.align_frames,
                self.optimize_endpoints,
                self.band_type,
                self.k_spring,
                self.nproc_rep,
                self.n_replica,
                self.n_replica_per_group,
                self.nsteps_it,
            ],
        )

    def return_dict(self):
        align_frames = ".FALSE."
        optimize_endpoints = ".FALSE."
        if self.align_frames.value:
            align_frames = ".TRUE."
        if self.optimize_endpoints.value:
            optimize_endpoints = ".TRUE."
        the_dict = {}
        if self.restart_from.value != "":
            the_dict["restart_from"] = orm.load_node(self.restart_from.value).uuid
        the_dict["neb_params"] = {
            "align_frames": align_frames,
            "band_type": self.band_type.value,
            "k_spring": self.k_spring.value,
            "nproc_rep": int(self.nproc_rep.value),
            "number_of_replica": int(self.n_replica.value),
            "nsteps_it": int(self.nsteps_it.value),
            "optimize_end_points": optimize_endpoints,
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
                (
                    [i, nrep // i]
                    for i in range(1, int(nrep**0.5) + 1)
                    if nrep % i == 0
                ),
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
        the_dict = {}
        the_dict["phonons_params"] = {"nproc_rep": int(self.nproc_rep.value)}

        return the_dict

    @tr.observe("nproc_replica_trait")
    def _observe_nproc_replica_trait(self, _=None):
        self.nproc_rep.value = str(self.nproc_replica_trait)

    @tr.observe("details")
    def _observe_details(self, _=None):
        self.n_replica.value = 3
        three_times_natoms = self.details["numatoms"] * 3
        # factors of 3 * Natoms
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

    def __init__(self):
        # UKS
        self.uks_toggle = ipw.ToggleButton(
            value=False,
            description="UKS",
            tooltip="Activate UKS",
            style={"description_width": "80px"},
        )

        tr.link((self, "uks"), (self.uks_toggle, "value"))

        self.spins = SpinsWidget()
        self.multiplicity = ipw.IntText(
            value=0,
            description="MULTIPLICITY",
            style={"description_width": "initial"},
            layout={"width": "140px"},
        )

        self.charge = ipw.IntText(
            value=0,
            description="net charge",
            style={"description_width": "initial"},
            layout={"width": "120px"},
        )

        super().__init__(selected_index=None)

    def return_dict(self):
        if self.uks:
            magnetization_per_site = np.zeros(self.details["numatoms"])
            for spinset in self.spins.spinsets.children:
                magnetization_per_site[
                    awb.utils.string_range_to_list(spinset.selection.value)[0]
                ] = spinset.starting_magnetization.value
            return {
                "dft_params": {
                    "uks": True,
                    "multiplicity": self.multiplicity.value,
                    "magnetization_per_site": magnetization_per_site.astype(
                        np.int32
                    ).tolist(),
                    "charge": self.charge.value,
                }
            }
        else:
            return {"dft_params": {"charge": self.charge.value}}

    @tr.observe("details")
    def _observe_details(self, _=None):
        self._widgets_to_show()

    @tr.observe("uks")
    def _observe_uks(self, _=None):
        self._widgets_to_show()

    def _widgets_to_show(self):
        self.set_title(0, "RKS/UKS")
        if self.uks:
            self.children = [
                ipw.VBox(
                    [
                        ipw.HBox(
                            [
                                self.uks_toggle,
                                self.multiplicity,
                            ]
                        ),
                        self.spins,
                        self.charge,
                    ]
                )
            ]
        else:
            self.children = [ipw.VBox([self.uks_toggle, self.charge])]

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


SECTIONS_TO_DISPLAY = {
    "None": [],
    "Wire": [],
    "Bulk": [
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        ConstraintsWidget,
        CellSectionWidget,
    ],
    "SlabXY": [
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        ConstraintsWidget,
    ],
    "Molecule": [
        StructureInfoWidget,
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        ConstraintsWidget,
    ],
    "Replica": [
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        ConstraintsWidget,
        ReplicaWidget,
    ],
    "Neb": [
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        ConstraintsWidget,
        NebWidget,
    ],
    "Phonons": [
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        ConstraintsWidget,
        PhononsWidget,
    ],
}
