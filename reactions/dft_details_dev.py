from __future__ import absolute_import, print_function

from collections import OrderedDict

import ipywidgets as ipw
from IPython.display import clear_output, display

###The widgets defined here assign value to the following input keywords
###stored in job_details:
#  'max_force'
#  'calc_type'    : 'Mixed DFT'
#  'mgrid_cutoff'
#  'vdw_switch'
#  'center_switch'
#  'charge'
#  'multiplicity'
#  'uks_switch'   : 'UKS' or 'RKS'
#      'spin_guess' e.e. C1 -1 1 1 C2 -1 1 1

#  'cell_free'  :'FREE','KEEP_ANGLES', 'KEEP_SYMMETRY'
#  'cell_sym'   : 'CUBIC' 'ORTHOROMBIC'...
#  'functional' : 'PBE' 'B3LYP' B3LYP not implemented
#  'gw_type'    : 'RI_RPA_GPW','GW-IC','GW-LS'
#      'group_size' 'max_memory' 'size_freq_integ_group' 'ev_sc_iter' 'corr_occ' 'corr_virt'
#      'eps_filter' 'eps_grid' 'rpa_num_quad_points' 'eps_schwarz' 'eps_pgf_orb_s'
#       'group_size_3c' 'gorup_size_p' 'memory_cut'

#  'cell'       : '30 30 30' not yet implemented


style = {"description_width": "120px"}
layout = {"width": "70%"}
layout2 = {"width": "35%"}
FUNCTION_TYPE = type(lambda c: c)
WIDGETS_DISABLE_DEFAULT = {
    "fixed_atoms": False,
    "max_force": False,
    "calc_type": False,
    "mgrid_cutoff": False,
    "vdw_switch": False,
    "periodic": True,
    "center_switch": True,
    "charge": True,
    "multiplicity": True,
    "uks_switch": True,
    "ot_switch": True,
    "added_mos": True,
    "cell_free": True,
    "cell_sym": True,
    "functional": True,
    "gw_type": True,
    "cell": True,
}


class DFTDetails(ipw.VBox):
    def __init__(self, job_details={}, widgets_disabled={}, **kwargs):

        self.widgets_disabled = WIDGETS_DISABLE_DEFAULT
        for wd in widgets_disabled:
            self.widgets_disabled[wd] = widgets_disabled[wd]

        ### ---------------------------------------------------------
        ### Define all child widgets contained in this composite widget
        self.job_details = job_details

        self.fixed_atoms = ipw.Text(
            placeholder="1..10",
            description="Fixed Atoms",
            style=style,
            layout={"width": "60%"},
        )

        self.btn_fixed_atoms = ipw.Button(description="show", layout={"width": "10%"})
        self.btn_fixed_pressed = False

        self.max_force = ipw.BoundedFloatText(
            description="MAX_FORCE:",
            value=1e-4,
            min=1e-4,
            max=1e-3,
            step=1e-4,
            style=style,
            layout=layout,
        )

        self.calc_type = ipw.ToggleButtons(
            options=["Mixed DFTB", "Mixed DFT", "Full DFT"],
            description="Calculation Type",
            value="Full DFT",
            tooltip="Active: DFT, Inactive: DFTB",
            style=style,
            layout=layout,
        )

        self.mgrid_cutoff = ipw.IntSlider(
            description="MGRID_CUTOFF:",
            value=600,
            step=100,
            min=200,
            max=1200,
            style=style,
            layout=layout,
        )

        self.vdw_switch = ipw.ToggleButton(
            value=True,
            description="Dispersion Corrections",
            tooltip="VDW_POTENTIAL",
            style=style,
            layout=layout,
        )

        self.center_switch = ipw.ToggleButton(
            value=False,
            disabled=False,
            description="Center Coordinates",
            tooltip="Center Coordinates",
            style=style,
            layout={"width": "60%"},
        )

        self.periodic = ipw.Dropdown(
            options=["XYZ", "NONE", "X", "XY", "XZ", "Y", "YZ", "Z"],
            description="PBC",
            value="NONE",
            style=style,
            layout=layout2,
        )

        self.psolver = ipw.Dropdown(
            options=["MT", "PERIODIC", "ANALYTIC", "IMPLICIT", "MULTIPOLE", "WAVELET"],
            description="Poisson solver",
            value="MT",
            style=style,
            layout=layout2,
        )

        def get_periodic_values():
            self.job_details["periodic"] = self.periodic.value
            self.job_details["psolver"] = self.psolver.value

        self.dft_out = ipw.Output()
        self.message_output = ipw.Output()
        ###for CELL and UKS
        self.charge = ipw.IntText(
            value=0, description="net charge", style=style, layout=layout
        )

        self.multiplicity = ipw.IntText(
            value=0, description="MULTIPLICITY", style=style, layout=layout
        )

        self.ot_switch = ipw.ToggleButtons(
            options=["OT", "DIAG"],
            description="OT",
            value="OT",
            style=style,
            layout=layout,
        )

        self.added_mos = ipw.IntText(
            value=10, description="ADDED_MOS", style=style, layout=layout
        )

        self.uks_switch = ipw.ToggleButtons(
            options=["UKS", "RKS"],
            description="UKS",
            value="RKS",
            style=style,
            layout=layout,
        )
        self.spin_guess = ipw.VBox()
        self.spin_guess_string = None

        def get_spin_string():
            self.job_details["spin_guess"] = self.spin_guess_string

        self.create_spin_guess_boxes()

        self.cell_free = ipw.ToggleButtons(
            options=["FREE", "KEEP_ANGLES", "KEEP_SYMMETRY"],
            description="Cell freedom",
            value="KEEP_SYMMETRY",
            style=style,
            layout=layout,
        )

        self.cell_sym = ipw.Dropdown(
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
            description="Cell symmetry",
            value="ORTHORHOMBIC",
            style=style,
            layout=layout,
        )

        self.functional = ipw.Dropdown(
            options=["PBE", "B3LYP"],
            description="XC Functional",
            value="PBE",
            style=style,
            layout=layout,
        )

        self.cell = ipw.Text(
            placeholder=" 30 30 20",
            description="cell",
            style=style,
            layout={"width": "60%"},
        )

        ###GW SECTION
        self.gw_type = ipw.Dropdown(
            options=["GW", "GW-IC", "GW-LS"],
            description="GW type",
            value="GW",
            style=style,
            layout=layout,
        )

        # self.gw_w_dict={'gw_type' :self.gw_type}
        self.gw_w_dict = {"GW": {}, "GW-IC": {}, "GW-LS": {}}
        self.gw_children = ipw.VBox()
        ###GROUP_SIZE 12
        self.group_size = ipw.IntText(
            value=12, description="Group Size", style=style, layout=layout
        )
        self.gw_w_dict["GW"]["group_size"] = self.group_size
        self.gw_w_dict["GW-IC"]["group_size"] = self.group_size
        ###GROUP_SIZE_3C 32
        self.group_size_3c = ipw.IntText(
            value=32, description="Group Size 3C", style=style, layout=layout
        )
        self.gw_w_dict["GW-LS"]["group_size_3c"] = self.group_size_3c
        ###GROUP_SIZE_P 4
        self.group_size_p = ipw.IntText(
            value=4, description="Group Size P", style=style, layout=layout
        )
        self.gw_w_dict["GW-LS"]["group_size_p"] = self.group_size_p
        ###MAX_MEMORY 0
        # self.max_memory = ipw.IntText(value=0,
        #                   description='Max Memory',
        #                   style=style, layout=layout)
        # self.gw_w_dict['GW']['max_memory']=self.max_memory
        # self.gw_w_dict['GW-IC']['max_memory']=self.max_memory
        ###MEMORY_CUT 12
        self.memory_cut = ipw.IntText(
            value=12, description="Memory cut", style=style, layout=layout
        )
        self.gw_w_dict["GW-LS"]["memory_cut"] = self.memory_cut
        ###SIZE_FREQ_INTEG_GROUP  1200
        self.size_freq_integ_group = ipw.IntText(
            value=-1, description="Size freq integ group", style=style, layout=layout
        )
        self.gw_w_dict["GW"]["size_freq_integ_group"] = self.size_freq_integ_group
        # self.gw_w_dict['GW-IC']['size_freq_integ_group']=self.size_freq_integ_group
        ###EV_SC_ITER 10
        self.ev_sc_iter = ipw.IntText(
            value=10, description="# EV SC iter", style=style, layout=layout
        )
        self.gw_w_dict["GW"]["ev_sc_iter"] = self.ev_sc_iter
        self.gw_w_dict["GW-LS"]["ev_sc_iter"] = self.ev_sc_iter
        ###CORR_OCC 15
        self.corr_occ = ipw.IntText(
            value=10, description="# KS occ", style=style, layout=layout
        )
        self.gw_w_dict["GW"]["corr_occ"] = self.corr_occ
        self.gw_w_dict["GW-IC"]["corr_occ"] = self.corr_occ
        self.gw_w_dict["GW-LS"]["corr_occ"] = self.corr_occ
        ###CORR_VIRT 15
        self.corr_virt = ipw.IntText(
            value=10, description="# KS virt", style=style, layout=layout
        )
        self.gw_w_dict["GW"]["corr_virt"] = self.corr_virt
        self.gw_w_dict["GW-IC"]["corr_virt"] = self.corr_virt
        self.gw_w_dict["GW-LS"]["corr_virt"] = self.corr_virt
        ###EPS_DEFAULT  1.0E-15
        self.eps_default = ipw.IntSlider(
            value=-15,
            min=-30,
            max=-12,
            step=1,
            description="EPS_DEFAULT 10^-",
            style=style,
            layout=layout,
        )
        self.gw_w_dict["GW"]["eps_default"] = self.eps_default
        self.gw_w_dict["GW-IC"]["eps_default"] = self.eps_default
        self.gw_w_dict["GW-LS"]["eps_default"] = self.eps_default
        ###EPS_FILTER  1.0E-12
        self.eps_filter = ipw.IntSlider(
            value=-12,
            min=-20,
            max=-12,
            step=1,
            description="EPS_FILTER 10^-",
            style=style,
            layout=layout,
        )
        self.gw_w_dict["GW"]["eps_filter"] = self.eps_filter
        self.gw_w_dict["GW-IC"]["eps_filter"] = self.eps_filter
        self.gw_w_dict["GW-LS"]["eps_filter"] = self.eps_filter
        ###EPS_GRID 1.0E-12
        self.eps_grid = ipw.IntSlider(
            value=-12,
            min=-30,
            max=-12,
            step=1,
            description="EPS_GRID 10^-",
            style=style,
            layout=layout,
        )
        self.gw_w_dict["GW"]["eps_grid"] = self.eps_grid
        self.gw_w_dict["GW-IC"]["eps_grid"] = self.eps_grid
        self.gw_w_dict["GW-LS"]["eps_grid"] = self.eps_grid
        ###EPS_PGF_ORB_S 1.0E-30
        self.eps_pgf_orb_s = ipw.IntSlider(
            value=-12,
            min=-30,
            max=-12,
            step=1,
            description="EPS_PGF_ORB_S 10^-",
            style=style,
            layout=layout,
        )
        self.gw_w_dict["GW-LS"]["eps_pgf_orb_s"] = self.eps_pgf_orb_s
        ###RPA_NUM_QUAD_POINTS 200  NOT IN IC
        self.rpa_num_quad_points = ipw.IntText(
            value=200,
            description="# rpa quad pt. use 12 (max 20) for  LS",
            style=style,
            layout=layout,
        )
        self.gw_w_dict["GW"]["rpa_num_quad_points"] = self.rpa_num_quad_points
        self.gw_w_dict["GW-LS"]["rpa_num_quad_points"] = self.rpa_num_quad_points
        ###EPS_SCHWARZ   1.0E-13  NOT IN IC
        self.eps_schwarz = ipw.IntSlider(
            value=-13,
            min=-20,
            max=-13,
            step=1,
            description="EPS_SCHWARZ 10^-",
            style=style,
            layout=layout,
        )
        self.gw_w_dict["GW"]["eps_schwarz"] = self.eps_schwarz

        ###EPS_FILTER_IM_TIME 1.0E-12 IC
        self.eps_filter_im_time = ipw.IntSlider(
            value=-12,
            min=-20,
            max=-12,
            step=1,
            description="EPS_FILTER_IM_TIME 10^-",
            style=style,
            layout=layout,
        )
        self.gw_w_dict["GW-IC"]["eps_filter_im_time"] = self.eps_filter_im_time
        self.gw_w_dict["GW-LS"]["eps_filter_im_time"] = self.eps_filter_im_time

        ###
        self.ads_height = ipw.FloatText(
            value=3.0,
            step=0.1,
            description="Ads. height (wrt geom. center)",
            style=style,
            layout=layout,
        )
        self.gw_w_dict["GW-IC"]["ads_height"] = self.ads_height

        def get_gw_values():
            gw_type = self.gw_type.value
            for key in self.gw_w_dict[gw_type].keys():
                self.job_details[key] = self.gw_w_dict[gw_type][key].value

        self.create_gw_boxes()
        ##END GW

        #### -------------------------------------------------------------------------------------------
        #### -------------------------------------------------------------------------------------------
        #### -------------------------------------------------------------------------------------------
        #### Methods to observe

        # Define the methods you want the widgets to observe

        def on_fixed_atoms_btn_press(b):
            self.btn_fixed_pressed = not self.btn_fixed_pressed
            self.btn_fixed_atoms.description = (
                "hide" if self.btn_fixed_pressed else "show"
            )

        self.btn_fixed_atoms.on_click(on_fixed_atoms_btn_press)

        ### DFT toggle button
        def on_dft_toggle(v):
            with self.dft_out:
                clear_output()
                if self.calc_type.value in ["Mixed DFT", "Full DFT"]:
                    display(self.mgrid_cutoff)

        self.calc_type.observe(on_dft_toggle, "value")

        def check_charge():
            with self.message_output:
                clear_output()
                total_charge = (
                    self.job_details["slab_analyzed"]["total_charge"]
                    + self.charge.value
                )
                if total_charge % 2 > 0:
                    print("odd charge: UKS NEEDED if CHARGE  is odd")
                    self.uks_switch.value = "UKS"
                    self.multiplicity.value = 2
                else:
                    self.uks_switch.value = "RKS"
                    self.multiplicity.value = 0

        # self.charge.observe(lambda c: check_charge(), 'value')

        self.uks_switch.observe(lambda c: self.create_spin_guess_boxes(), "value")
        self.gw_type.observe(lambda c: self.create_gw_boxes(), "value")

        #### -------------------------------------------------------------------------------------------
        #### -------------------------------------------------------------------------------------------
        #### -------------------------------------------------------------------------------------------
        ####

        self.independent_widgets = OrderedDict(
            [
                ("fixed_atoms", self.fixed_atoms),
                ("max_force", self.max_force),
                ("calc_type", self.calc_type),
                ("ot_switch", self.ot_switch),
                ("added_mos", self.added_mos),
                ("vdw_switch", self.vdw_switch),
                ("center_switch", self.center_switch),
                ("periodic", self.periodic),
                ("gw_type", self.gw_type),
                ("functional", self.functional),
                ("mgrid_cutoff", self.mgrid_cutoff),
                ("charge", self.charge),
                ("multiplicity", self.multiplicity),
                ("uks_switch", self.uks_switch),
                ("cell_free", self.cell_free),
                ("cell_sym", self.cell_sym),
                ("cell", self.cell),
            ]
        )

        ####some widgets do not have themselves to be visualized
        ####here are listed the exceptions

        self.independent_widget_children = OrderedDict(
            [
                ("fixed_atoms", [ipw.HBox([self.fixed_atoms, self.btn_fixed_atoms])]),
                ("mgrid_cutoff", [self.dft_out]),
                ("periodic", [ipw.HBox([self.periodic, self.psolver])]),
                ("uks_switch", [self.uks_switch, self.spin_guess]),
                ("gw_type", [self.gw_type, self.gw_children]),
            ]
        )

        ####some widgets follow special rules to update the job_details e.g. uks_switch
        #### needs to store both 'UKS' and spin_guess
        self.independent_widget_jd = OrderedDict(
            [
                ("uks_switch", get_spin_string),
                (
                    "gw_type",
                    get_gw_values,
                ),  ##TO DO switching gw_type the old job_entries remain defined
                ("periodic", get_periodic_values),
            ]
        )

        ####list widgets to be visualized and link to observe functions
        visualize_widgets = []
        for wk in self.independent_widgets:
            enabled = not self.widgets_disabled[wk]
            if enabled:
                if wk in self.independent_widget_children:
                    visualize_widgets.extend(self.independent_widget_children[wk])
                else:
                    visualize_widgets.append(self.independent_widgets[wk])

        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX

        ##################SUPER

        super(DFTDetails, self).__init__(children=visualize_widgets, **kwargs)

        with self.dft_out:
            display(self.mgrid_cutoff)

    def read_widgets_and_update_job_details(self):
        for w in self.independent_widgets.keys():
            if not self.widgets_disabled[w]:
                self.job_details[w] = self.independent_widgets[w].value
                if w in self.independent_widget_jd:
                    self.independent_widget_jd[w]()

    def reset(
        self,
        fixed_atoms="",
        btn_fixed_pressed=False,
        btn_fixed_atoms="show",
        vdw_switch=True,
        calc_type="Full DFT",
        center_switch=False,
        uks_switch="RKS",
        cell="",
    ):
        self.fixed_atoms.value = fixed_atoms
        self.btn_fixed_pressed = btn_fixed_pressed
        self.btn_fixed_atoms.description = btn_fixed_atoms
        self.calc_type.value = calc_type
        self.vdw_switch.value = vdw_switch
        self.center_switch.value = center_switch
        self.uks_switch.value = uks_switch
        self.cell.value = cell
        self.spin_guess_string = None

    def generate_spin_guess(self, int_net, guess_kinds):
        spin_guess = [
            [
                str(guess_kinds[i]),
                str(int_net[i][0].value),
                str(int_net[i][1].value),
                str(int_net[i][2].value),
            ]
            for i in range(len(guess_kinds))
        ]
        self.spin_guess_string = " ".join([x for xs in spin_guess for x in xs])

    def create_gw_boxes(self):
        gw_type = self.gw_type.value
        self.gw_children.children = tuple(
            [self.gw_w_dict[gw_type][key] for key in self.gw_w_dict[gw_type].keys()]
        )

    def create_spin_guess_boxes(self):
        if self.uks_switch.value == "UKS":
            spins_up = list(self.job_details["slab_analyzed"]["spins_up"])
            spins_down = list(self.job_details["slab_analyzed"]["spins_down"])

            self.int_net = []
            guess_kinds = spins_up + spins_down
            if len(guess_kinds) > 0:
                for k in guess_kinds:

                    self.int_net.append(
                        [
                            ipw.IntText(
                                value=0,
                                description="NEL " + k,
                                style={"description_width": "60px"},
                                layout={"width": "15%"},
                            ),
                            ipw.IntText(
                                value=0,
                                description="L " + k,
                                style={"description_width": "60px"},
                                layout={"width": "15%"},
                            ),
                            ipw.IntText(
                                value=0,
                                description="N " + k,
                                style={"description_width": "60px"},
                                layout={"width": "15%"},
                            ),
                        ]
                    )

                    for i_n in self.int_net[-1]:
                        i_n.observe(
                            lambda c, int_n=self.int_net, g_kinds=guess_kinds: self.generate_spin_guess(
                                int_n, g_kinds
                            ),
                            "value",
                        )

            self.spin_guess.children = tuple(
                [ipw.HBox([wn, wL, wN]) for wn, wL, wN, in self.int_net]
            )
            self.generate_spin_guess(self.int_net, guess_kinds)
        else:
            self.spin_guess_string = None
            self.spin_guess.children = tuple()
