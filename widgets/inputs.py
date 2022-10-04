import ipywidgets as ipw
from aiida.orm import Code
from aiidalab_widgets_base.utils import string_range_to_list
from IPython.display import clear_output, display
from traitlets import Bool, Dict, Instance, Int, List, Unicode, Union, link, observe

from .ANALYZE_structure import mol_ids_range
from .cp2k_input_validity import validate_input

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
        return {"vdw_switch": self.value}

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

        self.multiplicity = ipw.IntText(
            value=0,
            description="MULTIPLICITY",
            style={"description_width": "initial"},
            layout={"width": "140px"},
        )
        self.spin_u = ipw.Text(
            placeholder="1..10 15",
            description="IDs atoms spin U",
            style={"description_width": "initial"},
        )

        self.spin_d = ipw.Text(
            placeholder="1..10 15",
            description="IDs atoms spin D",
            style={"description_width": "initial"},
        )

        self.charge = ipw.IntText(
            value=0,
            description="net charge",
            style={"description_width": "initial"},
            layout={"width": "120px"},
        )

        # guess multiplicity
        def multiplicity_guess(c=None):
            self.net_charge = self.charge.value
            system_charge = self.details["total_charge"] - self.net_charge
            setu = set(string_range_to_list(self.spin_u.value)[0])
            setd = set(string_range_to_list(self.spin_d.value)[0])
            # check if same atom entered in two different spins
            if bool(setu & setd):
                self.multiplicity.value = 1
                self.spin_u.value = ""
                self.spin_d.value = ""

            nu = len(string_range_to_list(self.spin_u.value)[0])
            nd = len(string_range_to_list(self.spin_d.value)[0])
            if not system_charge % 2:
                self.multiplicity.value = min(abs(nu - nd) * 2 + 1, 3)
            else:
                self.multiplicity.value = 2

        self.spin_u.observe(multiplicity_guess, "value")
        self.spin_d.observe(multiplicity_guess, "value")
        self.charge.observe(multiplicity_guess, "value")

        super().__init__(selected_index=None)

    def return_dict(self):
        if self.uks:
            return {
                "multiplicity": self.multiplicity.value,
                "spin_u": self.spin_u.value,
                "spin_d": self.spin_d.value,
                "charge": self.charge.value,
            }
        else:
            return {
                "multiplicity": 0,
                "spin_u": "",
                "spin_d": "",
                "charge": self.charge.value,
            }

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
                                self.spin_u,
                                self.spin_d,
                            ]
                        ),
                        self.charge,
                    ]
                )
            ]
        else:
            self.children = [ipw.VBox([self.uks_toggle, self.charge])]

    def traits_to_link(self):
        return ["details", "uks", "net_charge"]


class FixedAtomsWidget(ipw.Text):
    to_fix = List()

    def __init__(
        self,
    ):
        super().__init__(
            placeholder="1..10",
            value=mol_ids_range(self.to_fix),
            description="Fixed Atoms",
            style=STYLE,
            layout={"width": "60%"},
        )

    def return_dict(self):
        return {"fixed_atoms": self.value}

    @observe("to_fix")
    def _observe_to_fix(self, _=None):
        self.value = mol_ids_range(self.to_fix)

    def traits_to_link(self):
        return ["to_fix"]


class CellSectionWidget(ipw.Accordion):
    details = Dict()
    do_cell_opt = Bool()
    net_charge = Int()

    def __init__(self):

        self.periodic = ipw.Dropdown(
            description="PBC",
            options=["XYZ", "NONE", "X", "XY", "XZ", "Y", "YZ", "Z"],
            value="XYZ",
            style=STYLE,
            layout=LAYOUT2,
        )

        self.poisson_solver = ipw.Dropdown(
            description="Poisson solver",
            options=["PERIODIC"],
            value="PERIODIC",
            style=STYLE,
            layout=LAYOUT2,
        )

        def observe_periodic(c=None):
            if self.periodic.value == "NONE":
                self.poisson_solver.options = [
                    "MT",
                    "ANALYTIC",
                    "IMPLICIT",
                    "MULTIPOLE",
                    "WAVELET",
                ]
                self.poisson_solver.value = "MT"
            elif self.periodic.value == "XYZ":
                self.poisson_solver.options = ["PERIODIC"]
                self.poisson_solver.value = "PERIODIC"
                if self.net_charge and self.details["system_type"] == "Molecule":
                    self.periodic.value = "NONE"

        self.periodic.observe(observe_periodic)

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

        self.cell = ipw.Text(
            description="cell size", style=STYLE, layout={"width": "60%"}
        )

        def observe_poisson(c=None):
            if self.poisson_solver.value == "MT":
                cell = self.details["sys_size"] * 2 + 15
                self.cell.value = " ".join(map(str, [int(i) for i in cell.tolist()]))
            elif self.poisson_solver.value == "PERIODIC":
                self.cell.value = self.details["cell"]

        self.poisson_solver.observe(observe_poisson)

        self.center_coordinates = ipw.RadioButtons(
            description="center coordinates",
            options=["False", "True"],
            value="True",
            disabled=False,
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
            options=["FREE", "KEEP_ANGLES", "KEEP_SYMMETRY"],
            description="Cell freedom",
            value="KEEP_SYMMETRY",
            style=STYLE,
            layout=LAYOUT,
        )

        # 'cell_free'
        self.cell_cases = {
            "Cell_true": [
                ("cell", self.cell),
                ("cell_sym", self.cell_sym),
                ("cell_free", self.cell_free),
                ("opt_cell", self.opt_cell),
            ],
            "Bulk": [
                #  ('cell', self.cell),
                ("opt_cell", self.opt_cell)
            ],
            "SlabXY": [
                #  ('periodic', self.periodic),
                #  ('poisson_solver', self.poisson_solver),
                #  ('cell', self.cell)
            ],
            "Molecule": [
                #  ('periodic', self.periodic),
                #  ('poisson_solver', self.poisson_solver),
                #  ('cell', self.cell),
                #  ('center_coordinates', self.center_coordinates)
            ],
        }

        super().__init__(selected_index=None)

    def return_dict(self):
        to_return = {}
        if self.opt_cell.value:
            cases = self.cell_cases["Cell_true"]
        else:
            cases = self.cell_cases[self.details["system_type"]]

        for i in cases:
            to_return.update({i[0]: i[1].value})
        return to_return

    @observe("details")
    def _observe_details(self, _=None):
        self._widgets_to_show()

    @observe("net_charge")
    def _observe_net_charge(self, _=None):
        self._widgets_to_show()

    @observe("do_cell_opt")
    def _observe_do_cell_opt(self, _=None):
        self._widgets_to_show()

    def _widgets_to_show(self):
        if self.opt_cell.value:
            self.set_title(0, "CELL/PBC details")
            self.children = [ipw.VBox([i[1] for i in self.cell_cases["Cell_true"]])]
        else:
            self.set_title(0, "CELL/PBC details")
            self.children = [
                ipw.VBox([i[1] for i in self.cell_cases[self.details["system_type"]]])
            ]
        self.cell.value = self.details["cell"]

        if self.net_charge and self.details["system_type"] == "Molecule":
            self.periodic.value = "NONE"

    def traits_to_link(self):
        return ["details", "do_cell_opt", "net_charge"]


class MetadataWidget(ipw.VBox):
    """Setup metadata for an AiiDA process."""

    details = Dict()
    uks = Bool()
    selected_code = Union([Unicode(), Instance(Code)], allow_none=True)

    def __init__(self):
        """Metadata widget to generate metadata"""

        self.walltime_s = ipw.IntText(
            value=86400,
            description="seconds:",
            style={"description_width": "initial"},
            layout={"width": "initial"},
        )

        self.nodes = ipw.IntText(
            value=48, description="# Nodes", style=STYLE, layout=LAYOUT2
        )
        self.tasks_per_node = ipw.IntText(
            value=12, description="tasks/node", style=STYLE
        )
        self.threads_per_task = ipw.IntText(
            value=1, description="threads/task", style=STYLE
        )

        children = [
            ipw.HBox([self.nodes, self.tasks_per_node, self.threads_per_task]),
            ipw.HBox([ipw.HTML("walltime, "), self.walltime_s]),
        ]

        super().__init__(children=children)
        # ---------------------------------------------------------

    def return_dict(self):

        return {
            "nodes": self.nodes.value,
            "tasks_per_node": self.tasks_per_node.value,
            "threads_per_task": self.threads_per_task.value,
            "walltime": self.walltime_s.value,
        }

    @observe("details")
    def _observe_details(self, _=None):
        self._suggest_resources()

    @observe("selected_code")
    def _observe_selected_code(self, _=None):
        self._suggest_resources()

    @observe("uks")
    def _observe_uks(self, _=None):
        self._suggest_resources()

    def _compute_cost(self):
        """Compute cost of the calculation."""

        cost = {
            "H": 1,
            "C": 4,
            "Si": 4,
            "N": 5,
            "O": 6,
            "Au": 11,
            "Cu": 11,
            "Ag": 11,
            "Pt": 18,
            "Tb": 19,
            "Co": 11,
            "Zn": 10,
            "Pd": 18,
            "Ga": 10,
        }
        the_cost = 0
        for element in self.details["all_elements"]:
            s = "".join(i for i in element if not i.isdigit())
            if isinstance(s[-1], type(1)):
                s = s[:-1]
            if s in cost.keys():
                the_cost += cost[s]
            else:
                the_cost += 4
        if (
            "Slab" in self.details["system_type"]
            or "Bulk" in self.details["system_type"]
        ):
            the_cost = int(the_cost / 11)
        else:
            the_cost = int(the_cost / 4)
        if self.uks:
            the_cost = the_cost * 1.26
        return the_cost

    def _suggest_resources(self):
        """ "Determine the resources needed for the calculation."""
        threads = 1
        try:
            max_tasks_per_node = (
                self.selected_code.computer.get_default_mpiprocs_per_machine()
            )
        except AttributeError:
            max_tasks_per_node = None
        if max_tasks_per_node is None:
            max_tasks_per_node = 1

        if not self.details["all_elements"] or self.selected_code is None:
            return 1, 1, 1

        resources = {
            "Slab": {
                50: {"nodes": 4, "tasks_per_node": max_tasks_per_node, "threads": 1},
                200: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                1400: {"nodes": 27, "tasks_per_node": max_tasks_per_node, "threads": 1},
                3000: {"nodes": 48, "tasks_per_node": max_tasks_per_node, "threads": 1},
                4000: {"nodes": 75, "tasks_per_node": max_tasks_per_node, "threads": 1},
                10000: {
                    "nodes": 108,
                    "tasks_per_node": max_tasks_per_node,
                    "threads": 1,
                },
            },
            "Bulk": {
                50: {"nodes": 4, "tasks_per_node": max_tasks_per_node, "threads": 1},
                200: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                1400: {"nodes": 27, "tasks_per_node": max_tasks_per_node, "threads": 1},
                3000: {"nodes": 48, "tasks_per_node": max_tasks_per_node, "threads": 1},
                4000: {"nodes": 75, "tasks_per_node": max_tasks_per_node, "threads": 1},
                10000: {
                    "nodes": 108,
                    "tasks_per_node": max_tasks_per_node,
                    "threads": 1,
                },
            },
            "Molecule": {
                50: {"nodes": 4, "tasks_per_node": max_tasks_per_node, "threads": 1},
                100: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                180: {"nodes": 27, "tasks_per_node": max_tasks_per_node, "threads": 1},
                400: {"nodes": 48, "tasks_per_node": max_tasks_per_node, "threads": 1},
            },
            "Default": {
                50: {"nodes": 4, "tasks_per_node": max_tasks_per_node, "threads": 1},
                100: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                180: {"nodes": 27, "tasks_per_node": max_tasks_per_node, "threads": 1},
                400: {"nodes": 48, "tasks_per_node": max_tasks_per_node, "threads": 1},
            },
        }
        cost = self._compute_cost()
        calctype = self.details["system_type"]
        # Slab_XY,....,Bulk,Molecule,Wire
        if "Slab" in self.details["system_type"]:
            calctype = "Slab"

        theone = min(resources[calctype], key=lambda x: abs(x - cost))
        nodes = resources[calctype][theone]["nodes"]
        tasks_per_node = resources[calctype][theone]["tasks_per_node"]
        threads = resources[calctype][theone]["threads"]
        self.nodes.value = nodes
        self.tasks_per_node.value = tasks_per_node
        self.threads_per_task.value = threads

    def traits_to_link(self):
        return ["details", "uks", "selected_code"]


SECTIONS_TO_DISPLAY = {
    "None": [],
    "Wire": [],
    "Bulk": [
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        FixedAtomsWidget,
        CellSectionWidget,
        ProtocolSelectionWidget,
        MetadataWidget,
    ],
    "SlabXY": [
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        FixedAtomsWidget,
        ProtocolSelectionWidget,
        MetadataWidget,
    ],
    "Molecule": [
        StructureInfoWidget,
        DescriptionWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        ProtocolSelectionWidget,
        MetadataWidget,
    ],
}
