import datetime

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
        return lastcalling.label, lastcalling.pk
    return None, None


def thunmnail_raw(
    nrows=1, thumbnail=None, pk=None, uuid=None, description="", tclass="tg-dark"
):
    """Returns an image with a link to structure export."""
    html = f'<td class="{tclass}" rowspan={nrows}><a target="_blank" href="./export_structure.ipynb?uuid={uuid}">'
    html += f'<img width="100px" src="data:image/png;base64,{thumbnail}" title="PK{pk}: {description}">'
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
        self.date_type = ipw.Dropdown(
            options=["mtime", "ctime"],
            value="mtime",
            style={"description_width": "60px"},
            layout={"width": "225px"},
        )
        self.date_text = ipw.HTML(value="<p>Select the date range:</p>", width="150px")
        self.keywords = ipw.Text(description="Keywords: ", layout={"width": "225px"})
        search_crit = ipw.HBox(
            [
                ipw.HBox(
                    [self.date_text, self.date_start, self.date_end, self.date_type]
                ),
                self.keywords,
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
                        "and": [
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
        # For each structure obtained with the queries, create a dictionary with info on the workflows computed on it.
        data = []
        for structure in structures:
            # print("node ", node.pk, " extras ", node.extras["surfaces"])
            workflows_uuids = structure.extras["surfaces"]
            nworkflows = 0
            nworkflows, workflows = uuids_to_nodesdict(uuids=workflows_uuids)
            if nworkflows > 0:
                nrows = nworkflows
                if "thumbnail" not in structure.extras:
                    structure.base.extras.set(
                        "thumbnail",
                        common_utils.thumbnail(ase_struc=structure.get_ase()),
                    )

                entry = {
                    "creator": find_first_workchain(structure),
                    "pk": structure.pk,
                    "uuid": structure.uuid,
                    "nrows": nrows,
                    "mtime": structure.mtime.strftime("%d/%m/%y"),
                    "ctime": structure.ctime.strftime("%d/%m/%y"),
                    "workflows": workflows,
                    "thumbnail": structure.extras["thumbnail"],
                }
                data.append(entry)
        # populate the table with the data
        # aiida_results td,th {padding: 2px}
        html = """
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:2px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-dark{background-color:#f1f7f7;border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-llyw{background-color:#fefee2;border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:middle}
</style>
<table class="tg">
<thead>
<tr>
    <th class="tg-dark">Calc. Type</th>
    <th class="tg-dark" >Description</th>
    <th class="tg-dark" >Thumbnail</th>
</tr>
</thead>
<tbody>
"""
        odd = -1
        tclass = ["", "tg-dark", "tg-llyw"]
        for entry in data:
            entry["nrows"]
            nrows_done = 0
            html += header(
                pk=entry["creator"][1],
                label=entry["creator"][0],
                tclass=tclass[odd],
                last_modified=entry["mtime"],
            )
            html += "<tr>"
            # html += f"""<td class="{tclass[odd]}" rowspan={str(nrows1)}> {entry["mtime"]}  </td>"""

            for workflow in entry["workflows"]:
                if nrows_done != 0:
                    html += "<tr>"
                nrowsw = len(entry["workflows"][workflow])
                html += f"<td class={tclass[odd]} rowspan={nrowsw}>  {workflow} </td>"
                html += f"<td class={tclass[odd]} rowspan={nrowsw}>"
                html += "<ul>"
                for node in entry["workflows"][workflow]:
                    html += link_to_viewer(
                        description=f"PK-{node.pk} {node.description}",
                        pk=node.pk,
                        label=node.label,
                    )
                html += "</ul></td>"
                if nrows_done == 0:
                    html += thunmnail_raw(
                        nrows=entry["nrows"],
                        thumbnail=entry["thumbnail"],
                        pk=entry["pk"],
                        uuid=entry["uuid"],
                        tclass=tclass[odd],
                        description="",
                    )
                    html += "</tr>"
                    nrows_done += 1
                for _ in range(1, nrowsw):
                    html += "<tr></tr>"
                    nrows_done += 1
            odd *= -1
        html += "</tbody></table>"

        self.results.value = html
