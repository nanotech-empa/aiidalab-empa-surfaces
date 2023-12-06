import datetime
import base64
import io
from PIL import Image

import ipywidgets as ipw
from aiida import common, orm
from aiida_nanotech_empa.utils import common_utils
from IPython.display import clear_output

VIEWERS = {
    "CP2K_AdsorptionE": "view_adsorption_energy.ipynb",
    "CP2K_GeoOpt": "view_geometry_optimization.ipynb",
    "CP2K_CellOpt": "view_geometry_optimization.ipynb",
    "CP2K_ORBITALS": "view_orbitals.ipynb",
    "CP2K_PDOS": "view_pdos.ipynb",
    "CP2K_STM": "view_stm.ipynb",
    "CP2K_AFM": "view_afm.ipynb",
    "CP2K_HRSTM": "view_hrstm.ipynb",
    "CP2K_Phonons": "view_phonons.ipynb",
    "CP2K_NEB": "view_neb.ipynb",
    "CP2K_Replica": "view_replica.ipynb",
    "ReplicaWorkChain": "view_replica.ipynb",
    "TBD": "view_workflow.ipynb",
}


def find_first_workchain(node):
    """Find the first workchain in the provenance that created the structure node."""
    lastcalling = None
    if isinstance(node, orm.StructureData):
        previous_node = node.creator
    else:
        previous_node = node
    while previous_node is not None:
        lastcalling = previous_node
        previous_node = lastcalling.caller
    if lastcalling is not None:
        return lastcalling.label, lastcalling.pk, lastcalling.description
    return None, None, ""


def thunmnail_raw(
    nrows=1, thumbnail=None, pk=None, uuid=None, description="", tclass="tg-dark"
):
    """Returns an image with a link to structure export."""
    html = f'<td class="{tclass}" rowspan={nrows}><a target="_blank" href="./export_structure.ipynb?uuid={uuid}">'
    html += f'<img width="100px" src="data:image/png;base64,{thumbnail}" title="input structure PK:{pk} {description}">'
    html += "</a></td>"
    return html


def link_to_viewer(description="", pk="", label=""):
    the_viewer = VIEWERS[label]
    return (
        f'<li><a target="_blank" href="{the_viewer}?pk={pk}"> {description} </a></li>'
    )


def header(pk="", label="", tclass="tg-dark", last_modified=""):
    if pk is None:
        return f"""<tr><td class="{tclass}" colspan=3> Structure created by input and last modified {last_modified}</td></tr>"""
    else:
        try:
            the_viewer = VIEWERS[label]
        except KeyError:
            return f"""<tr><td class="{tclass}" colspan=3> Structure created by {label} and last modified {last_modified}</td></tr>"""
        else:
            return f"""<tr><td class="{tclass}" colspan=3>  <a target="_blank" href="{the_viewer}?pk={pk}">Structure created by {label} PK-{pk}</a> and last modified {last_modified}</td></tr>"""


def uuids_to_nodesdict(uuids=[]):
    workflows = {}
    nworkflows = 0

    for uuid in uuids:
        try:
            node = orm.load_node(uuid)
            workchain_label = "TBD"
            if node.label in VIEWERS:
                workchain_label = node.label
            nworkflows += 1
            if workchain_label in workflows:
                workflows[workchain_label].append(node)
            else:
                workflows[workchain_label] = [node]
        except common.NotExistent:
            pass

    return nworkflows, workflows


class SearchStructuresWidget(ipw.VBox):
    def __init__(self):
        # Date selection.
        dt_now = datetime.datetime.now()
        dt_from = dt_now - datetime.timedelta(days=20)
        self.date_start = ipw.Text(
            value=dt_from.strftime("%Y-%m-%d"),
            description="From: ",
            style={"description_width": "60px"},
            layout={"width": "225px"},
        )

        self.date_end = ipw.Text(
            value=dt_now.strftime("%Y-%m-%d"),
            description="To: ",
            style={"description_width": "60px"},
            layout={"width": "225px"},
        )
        self.date_type = ipw.RadioButtons(
            options=[("Modification t", "mtime"), ("Creation t", "ctime")],
            value="mtime",
            style={"description_width": "60px"},
            layout={"width": "225px"},
        )
        self.date_text = ipw.HTML(value="<p>Select the date range:</p>", width="150px")
        # keywords selection
        self.keywords = ipw.Text(description="Keywords: ", layout={"width": "225px"})
        self.and_or = ipw.RadioButtons(options=["and", "or"], value="and")
        search_crit = ipw.VBox(
            [
                ipw.HBox(
                    [self.date_text, self.date_start, self.date_end, self.date_type]
                ),
                ipw.HBox([self.keywords, self.and_or]),
            ]
        )
        button = ipw.Button(description="Search")

        self.results = ipw.HTML()
        self.info_out = ipw.Output()

        def on_click(b):
            with self.info_out:
                clear_output()
                self.search()

        button.on_click(on_click)

        super().__init__([search_crit, button, self.results, self.info_out])

    def search(self):
        self.results.value = "searching..."

        try:  # If the date range is valid, use it for the search
            start_date = datetime.datetime.strptime(self.date_start.value, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(
                self.date_end.value, "%Y-%m-%d"
            ) + datetime.timedelta(hours=24)
        except ValueError:  # Otherwise revert to the standard (i.e. last 10 days)
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=20)
            self.date_start.value = start_date.strftime("%Y-%m-%d")
            self.date_end.value = end_date.strftime("%Y-%m-%d")
        search_keywords = self.keywords.value.split()
        # list of workchains with matching description, not obsolete and not failed
        if len(search_keywords) != 0:
            qb = orm.QueryBuilder()
            qb.append(
                orm.WorkChainNode,
                tag="wc",
                filters={
                    "attributes.exit_status": {"==": 0},
                    "or": [
                        {"extras": {"!has_key": "obsolete"}},
                        {"extras.obsolete": {"==": False}},
                    ],
                    # self.date_type.value: {"and": [{"<=": end_date}, {">": start_date}]},
                    "description": {
                        self.and_or.value: [
                            {"ilike": f"%{keyword}%"} for keyword in search_keywords
                        ]
                    },
                },
            )
            qb.append(orm.StructureData, with_outgoing="wc")
            structures = qb.all(flat=True)
        else:
            # Search with QB structures with extra "surfaces" not empty.
            qb = orm.QueryBuilder()
            qb.append(
                orm.StructureData,
                filters={
                    "and": [
                        {"extras": {"has_key": "surfaces"}},
                        {"extras.surfaces": {"longer": 0}},
                    ],
                    self.date_type.value: {
                        "and": [{"<=": end_date}, {">": start_date}]
                    },
                },
                # project=["extras.surfaces", "mtime", "ctime"],
            )
            qb.order_by({orm.StructureData: {self.date_type.value: "desc"}})
            structures = qb.all(flat=True)

        # For each structure obtained with the queries and teh connected workchains, create dash nodes.
        structure_nodes = []
        workchain_nodes = []
        edges = []
        roots = []
        for structure in structures:
            workflows_uuids = structure.extras["surfaces"]
            if len(workflows_uuids) > 0:
                if "thumbnail" not in structure.extras:
                    structure.base.extras.set(
                        "thumbnail",
                        common_utils.thumbnail(ase_struc=structure.get_ase()),
                    )
                thumbnail_w, thumbnail_h = Image.open(
                    io.BytesIO(base64.b64decode(structure.extras["thumbnail"]))
                ).size

                structure_nodes.append(
                    (str(structure.pk), 200, int(200 * thumbnail_h / thumbnail_w))
                )
                creator_label, creator_pk, creator_description = find_first_workchain(
                    structure
                )
                if creator_pk is not None:
                    workchain_nodes.append(
                        (str(creator_pk), creator_label, creator_description)
                    )
                    edges.append((workchain_nodes[-1][0], structure_nodes[-1][0]))
                for uuid in workflows_uuids:
                    try:
                        workchain = orm.load_node(uuid)
                        workchain_nodes.append(
                            (str(workchain.pk), workchain.label, workchain.description)
                        )
                        edges.append((structure_nodes[-1][0], workchain_nodes[-1][0]))
                    except common.NotExistent:
                        pass

        sources = [edge[0] for edge in edges]
        targets = [edge[1] for edge in edges]
        roots = [source for source in sources if source not in targets]
        # print(structure_nodes)
        # print(workchain_nodes)
        # print(edges)
        workchain_nodes = list(set(workchain_nodes))
        structure_nodes = list(set(structure_nodes))
        edges = list(set(edges))
        roots = list(set(roots))
        dash_structure_nodes = [
            {
                "data": {
                    "id": spk,
                    "label": spk,
                    "width": sw,
                    "height": sh,
                },
                "classes": "structure",
            }
            for spk, sw, sh in structure_nodes
        ]
        dash_workchain_nodes = [
            {
                "data": {"id": wpk, "label": wdescription},
                "classes": wtype,
            }
            for wpk, wtype, wdescription in workchain_nodes
        ]

        dash_edges = [
            {
                "data": {
                    "source": source,
                    "target": target,
                    "label": source + "-->" + target,
                }
            }
            for source, target in edges
        ]
        # print("WORKCHAINS")
        print("workchains=", dash_workchain_nodes)
        # print("STRUCTURES")
        print("structures=", dash_structure_nodes)
        # print("EDGES")
        print("edges=", dash_edges)
        # print("ROOTS")
        print("roots=", roots)
