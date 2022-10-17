import ipywidgets as ipw
import pandas as pd
import traitlets as trt
from aiida import orm

STYLE = {"description_width": "120px"}


class ProcessResourcesWidget(ipw.VBox):
    """Setup metadata for an AiiDA process."""

    def __init__(self):
        """Metadata widget to generate metadata"""

        self.walltime = ipw.Text(
            description="Walltime:",
            placeholder="10:30:00",
            style=STYLE,
        )

        self.wrong_syntax = ipw.HTML(
            value="""<i class="fa fa-times" style="color:red;font-size:2em;" ></i> wrong syntax""",
            layout={"visibility": "hidden"},
        )
        self.time_info = ipw.HTML()
        self.walltime.observe(self.parse_time_string, "value")

        self.walltime.value = "10:00:00"

        self.nodes = ipw.IntText(value=48, description="# Nodes", style=STYLE)
        self.tasks_per_node = ipw.IntText(
            value=12, description="# Tasks per node", style=STYLE
        )
        self.threads_per_task = ipw.IntText(
            value=1, description="# Threads per task", style=STYLE
        )

        children = [
            self.nodes,
            self.tasks_per_node,
            self.threads_per_task,
            ipw.HBox([self.walltime, self.time_info, self.wrong_syntax]),
        ]

        super().__init__(children=children)
        # ---------------------------------------------------------

    def return_dict(self):

        return {
            "nodes": self.nodes.value,
            "tasks_per_node": self.tasks_per_node.value,
            "threads_per_task": self.threads_per_task.value,
            "walltime": self.walltime.value,
        }

    def parse_time_string(self, _=None):
        """Parse the time string and set the time in seconds"""
        self.wrong_syntax.layout.visibility = "hidden"
        self.time_info.value = ""
        try:
            dtime = pd.Timedelta(self.walltime.value)
            self.time_info.value = str(dtime)
        except Exception:
            self.wrong_syntax.layout.visibility = "visible"
            self.time_info.value = ""


class ResourcesEstimatorWidget(ipw.VBox):

    details = trt.Dict()
    uks = trt.Bool()
    selected_code = trt.Union([trt.Unicode(), trt.Instance(orm.Code)], allow_none=True)

    def __init__(self):
        """Resources estimator widget to generate metadata"""

        self.estimate_resources_button = ipw.Button(
            description="Estimate resources", button_style="warning"
        )
        self.estimate_resources_button.on_click(self._estimate_resources)

        super().__init__([self.estimate_resources_button])

    def link_to_resources_widget(self, resources_widget):
        self.resources = resources_widget

    @trt.observe("details")
    def _observe_details(self, _=None):
        try:
            self.systype = (
                "Slab"
                if "Slab" in self.details["system_type"]
                else self.details["system_type"]
            )
            self.element_list = self.details["all_elements"]
        except KeyError:
            self.systype = "Other"
            self.element_list = []
        self.calctype = self.systype + "-DFT"

    @trt.observe("selected_code")
    def _observe_code(self, _=None):
        try:
            self.max_tasks_per_node = (
                self.selected_code.computer.get_default_mpiprocs_per_machine()
            )
        except AttributeError:
            self.max_tasks_per_node = 1

    def _compute_cost(self, element_list=None, systype="Slab", uks=False):
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
                if isinstance(s[-1], type(1)):
                    s = s[:-1]
                if s in cost.keys():
                    the_cost += cost[s]
                else:
                    the_cost += 4
            if systype == "Slab" or systype == "Bulk":
                the_cost = int(the_cost / 11)
            else:
                the_cost = int(the_cost / 4)
            if uks:
                the_cost = the_cost * 1.26
        return the_cost

    def _estimate_resources(self, _=None):
        """Determine the resources needed for the calculation."""

        resources = {
            "Slab-DFT": {
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
            "Bulk-DFT": {
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
            "Molecule-DFT": {
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
            "Other-DFT": {
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
            "gw": {
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
            },
            "gw_ic": {
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
            },
        }

        cost = self._compute_cost(
            element_list=self.element_list, systype=self.systype, uks=self.uks
        )

        theone = min(resources[self.calctype], key=lambda x: abs(x - cost))

        self.resources.nodes.value = resources[self.calctype][theone]["nodes"]
        self.resources.tasks_per_node.value = resources[self.calctype][theone][
            "tasks_per_node"
        ]
        self.resources.threads_per_task.value = resources[self.calctype][theone][
            "threads"
        ]
