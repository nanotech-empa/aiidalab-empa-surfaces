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
