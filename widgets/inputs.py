import ipywidgets as ipw
from aiida.orm import Code
from aiidalab_widgets_base.utils import string_range_to_list
from IPython.display import clear_output, display
from traitlets import Bool, Dict, Instance, Int, List, Unicode, Union, link, observe

import aiidalab_widgets_base as awb
import numpy as np

from .cp2k_input_validity import validate_input
from .constraints import ConstraintsWidget
from .spins import SpinsWidget

# from aiida_cp2k.workchains.base import Cp2kBaseWorkChain


STYLE = {"description_width": "120px"}
LAYOUT = {"width": "70%"}
LAYOUT2 = {"width": "35%"}


class InputDetails(ipw.VBox):
    selected_code = Union([Unicode(), Instance(Code)], allow_none=True)
    details = Dict()
    final_dictionary = Dict()
    to_fix = List()
    do_cell_opt = Bool()
    uks = Bool()
    net_charge = Int()

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

        super().__init__(children=[self.output])

    @observe("details")
    def _observe_details(self, _=None):
        self.to_fix = []
        self.net_charge = 0
        self.do_cell_opt = False
        self.uks = False
        with self.output:
            clear_output()

            if self.details:
                sys_type = self.details["system_type"]
            else:
                sys_type = "None"

            self.displayed_sections = []
            add_children = []

            for sec in SECTIONS_TO_DISPLAY[sys_type]:
                section = sec()
                if hasattr(section, "traits_to_link"):
                    for trait in section.traits_to_link():
                        link((self, trait), (section, trait))
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
                tmp_dict.update(to_add)

        # DECIDE WHICH KIND OF WORKCHAIN

        # SLAB
        if self.details["system_type"] == "SlabXY":
            tmp_dict.update({"workchain": "Cp2kSlabOptWorkChain"})

        # MOLECULE
        elif self.details["system_type"] == "Molecule":
            tmp_dict.update({"workchain": "Cp2kMoleculeOptWorkChain"})

        # BULK
        elif self.details["system_type"] == "Bulk":
            tmp_dict.update({"workchain": "Cp2kBulkOptWorkChain"})

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
    details = Dict()

    def __init__(self):

        self.info = ipw.Output()

        self.set_title(0, "Structure details")
        super().__init__(selected_index=None)

    @observe("details")
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


class ProtocolSelectionWidget(ipw.Dropdown):
    def __init__(self):
        options = [
            ("Standard", "standard"),
            ("Low accuracy", "low_accuracy"),
            ("Debug", "debug"),
        ]
        super().__init__(
            value="standard",
            options=options,
            description="Protocol:",
            style={"description_width": "120px"},
        )

    def return_dict(self):
        return {"protocol": self.value}


class VdwSelectorWidget(ipw.ToggleButton):
    def __init__(self):
        super().__init__(
            value=True,
            description="Dispersion Corrections",
            tooltip="VDW_POTENTIAL",
            style={"description_width": "120px"},
        )

    def return_dict(self):
        return {"vdw": self.value}

    def traits_to_link(self):
        return []


class UksSectionWidget(ipw.Accordion):
    details = Dict()
    uks = Bool()
    net_charge = Int()

    def __init__(self):
        # UKS
        self.uks_toggle = ipw.ToggleButton(
            value=False,
            description="UKS",
            tooltip="Activate UKS",
            style={"description_width": "80px"},
        )

        link((self, "uks"), (self.uks_toggle, "value"))

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

        # guess multiplicity
        # def multiplicity_guess(c=None):
        #    self.net_charge = self.charge.value
        #    system_charge = self.details["total_charge"] - self.net_charge
        #    # check if same atom entered in two different spins
        #    if bool(setu & setd):
        #        self.multiplicity.value = 1
        #        self.spin_u.value = ""
        #        self.spin_d.value = ""

        #    if not system_charge % 2:
        #        self.multiplicity.value = min(abs(nu - nd) * 2 + 1, 3)
        #    else:
        #        self.multiplicity.value = 2

        # self.charge.observe(multiplicity_guess, "value")

        super().__init__(selected_index=None)

    def return_dict(self):
        if self.uks:
            magnetization_per_site = np.zeros(self.details["numatoms"])
            for spinset in self.spins.spinsets.children:
                magnetization_per_site[
                    awb.utils.string_range_to_list(spinset.selection.value)[0]
                ] = spinset.starting_magnetization.value
            return {
                "uks": True,
                "multiplicity": self.multiplicity.value,
                "magnetisation_per_site": magnetization_per_site.astype(
                    np.int32
                ).tolist(),
                "charge": self.charge.value,
            }
        else:
            return {"charge": self.charge.value}

    @observe("details")
    def _observe_details(self, _=None):
        self._widgets_to_show()

    @observe("uks")
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
    details = Dict()
    do_cell_opt = Bool()

    def __init__(self):

        self.cell_constraint = ipw.Dropdown(
            description="Cell constr.",
            options=["XYZ", "NONE", "X", "XY", "XZ", "Y", "YZ", "Z"],
            value="NONE",
            style=STYLE,
            layout=LAYOUT2,
        )

        self.cell_sym = ipw.Dropdown(
            description="symmetry",
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
            layout=LAYOUT,
        )

        self.opt_cell = ipw.ToggleButton(
            value=False,
            description="Optimize cell",
            style={"description_width": "120px"},
        )

        def on_cell_opt(c=None):
            self.do_cell_opt = self.opt_cell.value

        self.opt_cell.observe(on_cell_opt, "value")

        self.cell_free = ipw.ToggleButtons(
            options=["FREE", "KEEP_SYMMETRY", "KEEP_ANGLES", "KEEP_SPACE_GROUP"],
            description="Cell freedom",
            value="KEEP_SYMMETRY",
            style=STYLE,
            layout=LAYOUT,
        )

        super().__init__(selected_index=None)

    def return_dict(self):
        sys_params = {"symmetry": self.cell_sym.value}
        if self.opt_cell.value:
            sys_params["cell_opt"] = ""
        if self.cell_constraint.value != "NONE":
            sys_params["cell_opt_constraint"] = self.cell_constraint.value
        if self.cell.free.value != "FREE":
            sys_params[self.cell.free.value.lower()] = ""

        return sys_params

    @observe("details")
    def _observe_details(self, _=None):
        self._widgets_to_show()

    @observe("do_cell_opt")
    def _observe_do_cell_opt(self, _=None):
        self._widgets_to_show()

    def _widgets_to_show(self):
        if self.opt_cell.value:
            self.set_title(0, "CELL details")
            self.children = [
                ipw.VBox([self.cell_sym, self.cell_free, self.cell_constraint])
            ]
        else:
            self.set_title(0, "CELL details")
            self.children = [self.cell_sym]

        if self.net_charge and self.details["system_type"] == "Molecule":
            self.periodic.value = "NONE"

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
        ProtocolSelectionWidget,
    ],
    "SlabXY": [
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        ConstraintsWidget,
        ProtocolSelectionWidget,
    ],
    "Molecule": [
        StructureInfoWidget,
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        ConstraintsWidget,
        ProtocolSelectionWidget,
    ],
}
