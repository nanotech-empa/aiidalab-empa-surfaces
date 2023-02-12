import ipywidgets as ipw
import pandas as pd
import traitlets as trt
from aiida import orm

STYLE = {
    "description_width": "120px",
}
LAYOUT = {"width": "200px"}


class ProcessResourcesWidget(ipw.VBox):
    """Setup metadata for an AiiDA process."""

    def __init__(self):
        """Metadata widget to generate metadata"""

        self.walltime_widget = ipw.Text(
            description="Walltime:",
            placeholder="10:30:00",
            style=STYLE,
            layout=LAYOUT,
        )

        self.wrong_syntax = ipw.HTML(
            value="""<i class="fa fa-times" style="color:red;font-size:2em;" ></i> wrong syntax""",
            layout={"visibility": "hidden"},
        )
        self.time_info = ipw.HTML(layout=LAYOUT)
        self.walltime_widget.observe(self.parse_time_string, "value")

        self.walltime_widget.value = "24:00:00"

        self.nodes_widget = ipw.IntText(
            value=48, description="# Nodes", style=STYLE, layout=LAYOUT
        )
        self.tasks_per_node_widget = ipw.IntText(
            value=12, description="# Tasks per node", style=STYLE, layout=LAYOUT
        )
        self.threads_per_task_widget = ipw.IntText(
            value=1, description="# Threads per task", style=STYLE, layout=LAYOUT
        )

        children = [
            self.nodes_widget,
            self.tasks_per_node_widget,
            self.threads_per_task_widget,
            ipw.HBox([self.walltime_widget, self.wrong_syntax]),
            self.time_info,
        ]

        super().__init__(children=children)
        # ---------------------------------------------------------

    @property
    def nodes(self):
        return int(self.nodes_widget.value)

    @property
    def tasks_per_node(self):
        return int(self.tasks_per_node_widget.value)

    @property
    def threads_per_task(self):
        return int(self.threads_per_task_widget.value)

    @property
    def walltime_seconds(self):
        return int(pd.Timedelta(self.walltime_widget.value).total_seconds())

    def parse_time_string(self, _=None):
        """Parse the time string and set the time in seconds"""
        self.wrong_syntax.layout.visibility = "hidden"
        self.time_info.value = ""
        try:
            dtime = pd.Timedelta(self.walltime_widget.value)
            self.time_info.value = f"Walltime will be: {dtime}"
        except Exception:
            self.wrong_syntax.layout.visibility = "visible"
            self.time_info.value = ""


class ResourcesEstimatorWidget(ipw.VBox):

    details = trt.Dict()
    uks = trt.Bool()
    selected_code = trt.Union([trt.Unicode(), trt.Instance(orm.Code)], allow_none=True)

    def __init__(self, calculation_type="dft"):
        """Resources estimator widget to generate metadata"""

        self.max_tasks_per_node = 1
        self.calculation_type = calculation_type
        self.estimate_resources_button = ipw.Button(
            description="Estimate resources", button_style="warning"
        )
        self.estimate_resources_button.on_click(self.estimate_resources)

        super().__init__([self.estimate_resources_button])

    def link_to_resources_widget(self, resources_widget):
        self.resources = resources_widget

    @trt.observe("details")
    def _observe_details(self, _=None):
        try:
            self.system_type = (
                "Slab"
                if "Slab" in self.details["system_type"]
                else self.details["system_type"]
            )
            self.element_list = self.details["all_elements"]
        except KeyError:
            self.system_type = "Other"
            self.element_list = []

    @trt.observe("selected_code")
    def _observe_code(self, _=None):
        try:
            self.max_tasks_per_node = (
                orm.load_code(self.selected_code).computer.get_default_mpiprocs_per_machine()
                )
        except (ValueError,AttributeError):
            print("Code not recognized setting tasks per node to 1")
            self.max_tasks_per_node = 1

    def _compute_cost(self, element_list=None, system_type="Slab", uks=False):
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
        if element_list is not None:
            for element in element_list:
                s = "".join(i for i in element if not i.isdigit())
                if isinstance(s[-1], int):
                    s = s[:-1]
                if s in cost.keys():
                    the_cost += cost[s]
                else:
                    the_cost += 4
            if system_type == "Slab" or system_type == "Bulk":
                the_cost = int(the_cost / 11)
            else:
                the_cost = int(the_cost / 4)
            if uks:
                the_cost = the_cost * 1.26
        return the_cost

    def estimate_resources(self, _=None):
        """Determine the resources needed for the calculation."""

        if self.calculation_type == "dft":
            resources = self._estimate_resources_dft()
        elif self.calculation_type == "gw":
            resources = self._estimate_resources_gw()
        elif self.calculation_type == "gw_ic":
            resources = self._estimate_resources_gw_ic()

        cost = self._compute_cost(
            element_list=self.element_list, system_type=self.system_type, uks=self.uks
        )

        theone = min(resources, key=lambda x: abs(x - cost))

        self.resources.nodes_widget.value = resources[theone]["nodes"]
        self.resources.tasks_per_node_widget.value = resources[theone]["tasks_per_node"]
        self.resources.threads_per_task_widget.value = resources[theone]["threads"]

    def _estimate_resources_dft(self):
        """Determine the resources needed for the DFT calculation."""

        resources = {
            "Slab": {
                50: {
                    "nodes": 4,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                200: {
                    "nodes": 12,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                1400: {
                    "nodes": 27,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                3000: {
                    "nodes": 48,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                4000: {
                    "nodes": 75,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                10000: {
                    "nodes": 108,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
            },
            "Bulk": {
                50: {
                    "nodes": 4,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                200: {
                    "nodes": 12,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                1400: {
                    "nodes": 27,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                3000: {
                    "nodes": 48,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                4000: {
                    "nodes": 75,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                10000: {
                    "nodes": 108,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
            },
            "Molecule": {
                50: {
                    "nodes": 4,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                100: {
                    "nodes": 12,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                180: {
                    "nodes": 27,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                400: {
                    "nodes": 48,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
            },
            "Other": {
                50: {
                    "nodes": 4,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                100: {
                    "nodes": 12,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                180: {
                    "nodes": 27,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
                400: {
                    "nodes": 48,
                    "tasks_per_node": self.max_tasks_per_node,
                    "threads": 1,
                },
            },
        }

        return resources[self.system_type]

    def _estimate_resources_gw(self):
        resources = {
            10: {
                "nodes": 2,
                "tasks_per_node": max(self.max_tasks_per_node / 4, 1),
                "threads": 1,
            },
            20: {
                "nodes": 6,
                "tasks_per_node": max(self.max_tasks_per_node / 4, 1),
                "threads": 1,
            },
            50: {
                "nodes": 12,
                "tasks_per_node": self.max_tasks_per_node,
                "threads": 1,
            },
            100: {
                "nodes": 256,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
            180: {
                "nodes": 512,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
            400: {
                "nodes": 1024,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
        }
        return resources

    def _estimate_resources_gw_ic(self):
        resources = {
            10: {
                "nodes": 2,
                "tasks_per_node": max(self.max_tasks_per_node / 4, 1),
                "threads": 1,
            },
            20: {
                "nodes": 6,
                "tasks_per_node": max(self.max_tasks_per_node / 4, 1),
                "threads": 1,
            },
            50: {
                "nodes": 12,
                "tasks_per_node": self.max_tasks_per_node,
                "threads": 1,
            },
            100: {
                "nodes": 256,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
            180: {
                "nodes": 512,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
            400: {
                "nodes": 1024,
                "tasks_per_node": int(max(self.max_tasks_per_node / 3, 1)),
                "threads": 1,
            },
        }
        return resources
