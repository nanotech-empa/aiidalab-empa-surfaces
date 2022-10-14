import traitlets as trt
import ipywidgets as ipw
from aiida import orm


STYLE = {"description_width": "120px"}
LAYOUT = {"width": "35%"}


class ProcessResourcesWidget(ipw.VBox):
    """Setup metadata for an AiiDA process."""

    def __init__(self):
        """Metadata widget to generate metadata"""

        self.walltime = ipw.Text(
            value="86400",
            description="Walltime:",
            style={"description_width": "initial"},
            layout={"width": "initial"},
        )

        self.nodes = ipw.IntText(
            value=48, description="# Nodes", style=STYLE, layout=LAYOUT
        )
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
            self.walltime,
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



class ResourcesEstimatorWidget(ipw.VBox):

    details = trt.Dict()
    uks = trt.Bool()
    selected_code = trt.Union([trt.Unicode(), trt.Instance(orm.Code)], allow_none=True)

    def __init__(self):
        """Resources estimator widget to generate metadata"""

        self.estimate_nodes_button = ipw.Button(
            description="Estimate resources", button_style="warning"
        )
        self.estimate_nodes_button.on_click(self._suggest_resources)

        super().__init__([self.estimate_resources_button])

    def _suggest_resources(self, _=None):
        try:
            max_tasks_per_node = (
                self.selected_code.computer.get_default_mpiprocs_per_machine()
            )
        except AttributeError:
            max_tasks_per_node = None
        if max_tasks_per_node is None:
            max_tasks_per_node = 1

        try:
            systype = self.details["system_type"]
            element_list = self.details["all_elements"]
        except KeyError:
            systype = "Other"
            element_list = []

        if "Slab" in systype:
            systype = "Slab"
        calctype = systype + "-DFT"

        (
            self.nodes.value,
            self.tasks_per_node.value,
            self.threads_per_task.value,
        ) = get_nodes(
            element_list=element_list,
            calctype=calctype,
            systype=systype,
            max_tasks_per_node=max_tasks_per_node,
            uks=False,
        )

    def compute_cost(element_list=None, systype="Slab", uks=False):
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


    def get_nodes(
        element_list=None,
        calctype="default",
        systype="Slab",
        max_tasks_per_node=1,
        uks=False,
    ):
        """ "Determine the resources needed for the calculation."""

        resources = {
            "Slab-DFT": {
                50: {"nodes": 4, "tasks_per_node": max_tasks_per_node, "threads": 1},
                200: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                1400: {"nodes": 27, "tasks_per_node": max_tasks_per_node, "threads": 1},
                3000: {"nodes": 48, "tasks_per_node": max_tasks_per_node, "threads": 1},
                4000: {"nodes": 75, "tasks_per_node": max_tasks_per_node, "threads": 1},
                10000: {"nodes": 108, "tasks_per_node": max_tasks_per_node, "threads": 1},
            },
            "Bulk-DFT": {
                50: {"nodes": 4, "tasks_per_node": max_tasks_per_node, "threads": 1},
                200: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                1400: {"nodes": 27, "tasks_per_node": max_tasks_per_node, "threads": 1},
                3000: {"nodes": 48, "tasks_per_node": max_tasks_per_node, "threads": 1},
                4000: {"nodes": 75, "tasks_per_node": max_tasks_per_node, "threads": 1},
                10000: {"nodes": 108, "tasks_per_node": max_tasks_per_node, "threads": 1},
            },
            "Molecule-DFT": {
                50: {"nodes": 4, "tasks_per_node": max_tasks_per_node, "threads": 1},
                100: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                180: {"nodes": 27, "tasks_per_node": max_tasks_per_node, "threads": 1},
                400: {"nodes": 48, "tasks_per_node": max_tasks_per_node, "threads": 1},
            },
            "Other-DFT": {
                50: {"nodes": 4, "tasks_per_node": max_tasks_per_node, "threads": 1},
                100: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                180: {"nodes": 27, "tasks_per_node": max_tasks_per_node, "threads": 1},
                400: {"nodes": 48, "tasks_per_node": max_tasks_per_node, "threads": 1},
            },
            "gw": {
                10: {
                    "nodes": 2,
                    "tasks_per_node": max(max_tasks_per_node / 4, 1),
                    "threads": 1,
                },
                20: {
                    "nodes": 6,
                    "tasks_per_node": max(max_tasks_per_node / 4, 1),
                    "threads": 1,
                },
                50: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                100: {
                    "nodes": 256,
                    "tasks_per_node": int(max(max_tasks_per_node / 3, 1)),
                    "threads": 1,
                },
                180: {
                    "nodes": 512,
                    "tasks_per_node": int(max(max_tasks_per_node / 3, 1)),
                    "threads": 1,
                },
                400: {
                    "nodes": 1024,
                    "tasks_per_node": int(max(max_tasks_per_node / 3, 1)),
                    "threads": 1,
                },
            },
            "gw_ic": {
                10: {
                    "nodes": 2,
                    "tasks_per_node": max(max_tasks_per_node / 4, 1),
                    "threads": 1,
                },
                20: {
                    "nodes": 6,
                    "tasks_per_node": max(max_tasks_per_node / 4, 1),
                    "threads": 1,
                },
                50: {"nodes": 12, "tasks_per_node": max_tasks_per_node, "threads": 1},
                100: {
                    "nodes": 256,
                    "tasks_per_node": int(max(max_tasks_per_node / 3, 1)),
                    "threads": 1,
                },
                180: {
                    "nodes": 512,
                    "tasks_per_node": int(max(max_tasks_per_node / 3, 1)),
                    "threads": 1,
                },
                400: {
                    "nodes": 1024,
                    "tasks_per_node": int(max(max_tasks_per_node / 3, 1)),
                    "threads": 1,
                },
            },
        }

        cost = compute_cost(element_list=element_list, systype=systype, uks=uks)

        theone = min(resources[calctype], key=lambda x: abs(x - cost))
        nodes = resources[calctype][theone]["nodes"]
        tasks_per_node = resources[calctype][theone]["tasks_per_node"]
        threads = resources[calctype][theone]["threads"]
        return nodes, tasks_per_node, threads
