import datetime

import ipywidgets as ipw
from aiida import common, orm
from IPython.display import clear_output

from surfaces_tools import utils

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


def link_to_viewer(description="", pk="", label="", energy=None, pk_eq_geo=None):
    the_viewer = VIEWERS[label]
    html = (
        f'<li><a target="_blank" href="{the_viewer}?pk={pk}"> {description} </a></li>'
    )
    if energy is not None:
        html += f"&nbsp;&nbsp;&nbsp;Energy: {energy:.3f} (Hartree)<br>"
    if pk_eq_geo is not None:
        html += f"&nbsp;&nbsp;&nbsp;Eq. geometry: <a target='_blank' href='{the_viewer}?pk={pk_eq_geo}'> PK: {pk_eq_geo} </a><br> "
    return html


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


def uuids_to_nodesdict(uuids):
    workflows = {}
    nworkflows = 0
    for uuid in uuids:
        try:
            node = orm.load_node(uuid)
            nodeisobsolete = "obsolete" in node.extras and node.extras["obsolete"]
            if node.label in VIEWERS and not nodeisobsolete:
                nworkflows += 1
                if node.label in workflows:
                    workflows[node.label].append(node)
                else:
                    workflows[node.label] = [node]
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

        self.date_text = ipw.HTML(value="<p>Select the date range:</p>", width="150px")
        search_crit = ipw.HBox([self.date_text, self.date_start, self.date_end])
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

        # Search with QB structures with extra "surfaces".
        qb = orm.QueryBuilder()
        qb.append(
            orm.StructureData,
            filters={
                "extras": {"has_key": "surfaces"},
                "mtime": {"and": [{"<=": end_date}, {">": start_date}]},
            },
        )
        qb.order_by({orm.StructureData: {"mtime": "desc"}})

        # For each structure in QB create a dictionary with info on the workflows computed on it.
        data = []
        for node in qb.all(flat=True):
            # print("node ", node.pk, " extras ", node.extras["surfaces"])
            extras = node.extras["surfaces"]
            nworkflows = 0
            if isinstance(extras, list):
                nworkflows, workflows = uuids_to_nodesdict(node.extras["surfaces"])
            if nworkflows > 0:
                nrows = nworkflows
                if "thumbnail" not in node.extras:
                    node.base.extras.set(
                        "thumbnail", utils.ase_to_thumbnail(structure=node.get_ase())
                    )

                entry = {
                    "creator": find_first_workchain(node),
                    "pk": node.pk,
                    "uuid": node.uuid,
                    "nrows": nrows,
                    "mtime": node.mtime.strftime("%d/%m/%y"),
                    "workflows": workflows,
                    "thumbnail": node.extras["thumbnail"],
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
                    energy = None
                    pk_eq_geo = None
                    if node.label == "CP2K_GeoOpt":
                        try:
                            energy = node.outputs.output_parameters.get_dict()["energy"]
                        except AttributeError:
                            energy = None
                        try:
                            pk_eq_geo = node.outputs.output_structure.pk
                        except AttributeError:
                            pk_eq_geo = None

                        html += link_to_viewer(
                            description=f"PK-{node.pk} {node.description}",
                            pk=node.pk,
                            label=node.label,
                            energy=energy,
                            pk_eq_geo=pk_eq_geo,
                        )

                html += "</ul></td>"
                if nrows_done == 0:
                    html += utils.thumbnail_raw(
                        nrows=entry["nrows"],
                        thumbnail=entry["thumbnail"],
                        pk=entry["pk"],
                        uuid=entry["uuid"],
                        tclass=tclass[odd],
                        description="Inp. structure",
                    )
                    html += "</tr>"
                    nrows_done += 1
                for _ in range(1, nrowsw):
                    html += "<tr></tr>"
                    nrows_done += 1
            odd *= -1
        html += "</tbody></table>"

        self.results.value = html
