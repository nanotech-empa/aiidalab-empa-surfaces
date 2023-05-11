import datetime

import ipywidgets as ipw
from aiida import common, orm
from aiida_nanotech_empa.utils import common_utils
from IPython.display import clear_output

VIEWERS = {
    "CP2K_AdsorptionE": "view_ade.ipynb",
    "CP2K_GeoOpt": "view_geoopt.ipynb",
    "CP2K_CellOpt": "view_geoopt.ipynb",
    "CP2K_Orbitals": "view_orb.ipynb",
    "CP2K_PDOS": "view_pdos.ipynb",
    "CP2K_STM": "view_stm.ipynb",
    "CP2K_AFM": "view_afm.ipynb",
    "CP2K_HRSTM": "view_hrstm.ipynb",
    "CP2K_Phonons": "view_phonons.ipynb",
    "CP2K_NEB": "view_neb.ipynb",
}


def thunmnail_raw(nrows=1, thumbnail=None, pk=None, description=""):
    """Returns an image with a link to structure export."""
    html = (
        f'<td rowspan={nrows}><a target="_blank" href="./export_structure.ipynb?{pk=}">'
    )
    html += f'<img width="100px" src="data:image/png;base64,{thumbnail}" title="PK{pk}: {description}">'
    html += "</a></td>"
    return html


def link_to_viewer(description="", pk="", label=""):
    pk = str(pk)
    the_viewer = VIEWERS[label]
    return f'<li><a target="_blank" href="{the_viewer}?{pk=}"> {description} </a></li>'


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
                        "thumbnail", common_utils.thumbnail(ase_struc=node.get_ase())
                    )
                entry = {
                    "pk": node.pk,
                    "nrows": nrows,
                    "mtime": node.mtime.strftime("%d/%m/%y"),
                    "workflows": workflows,
                    "thumbnail": thunmnail_raw(
                        nrows=nrows,
                        thumbnail=node.extras["thumbnail"],
                        pk=node.pk,
                        description="",
                    ),
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
.tg .tg-dark{background-color:#c0c0c0;border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-llyw{background-color:#efefef;border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:middle}
</style>
<table class="tg">
<thead>
<tr>
    <th class="tg-dark" >Date last</th>
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
            nrows1 = entry["nrows"]
            nrows_done = 0
            html += "<tr>"
            html += f"""<td class="{tclass[odd]}" rowspan={str(nrows1)}> {entry["mtime"]}  </td>"""
            odd *= -1
            for workflow in entry["workflows"]:
                if nrows_done != 0:
                    html += "<tr>"
                nrowsw = len(entry["workflows"][workflow])
                html += f'<td class="tg-0pky" rowspan={nrowsw}>  {workflow} </td>'
                html += f'<td class="tg-0pky" rowspan={nrowsw}>'
                html += "<ul>"
                for node in entry["workflows"][workflow]:
                    html += link_to_viewer(
                        description=f"PK-{node.pk} {node.description}",
                        pk=node.pk,
                        label=node.label,
                    )
                html += "</ul></td>"
                if nrows_done == 0:
                    html += entry["thumbnail"]
                    html += "</tr>"
                    nrows_done += 1
                for _ in range(1, nrowsw):
                    html += "<tr></tr>"
                    nrows_done += 1
        html += "</tbody></table>"

        self.results.value = html
