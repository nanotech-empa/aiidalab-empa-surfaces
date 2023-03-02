import datetime

import ipywidgets as ipw
from aiida.orm import QueryBuilder, StructureData, load_node
from IPython.display import clear_output
from aiida_nanotech_empa.utils import common_utils
from aiida.common.exceptions import NotExistent

VIEWERS = {
    "Cp2kAdsorbedGwIcWorkChain_pks": "./gw/view_gw-ic.ipynb",
    "Cp2kMoleculeOptGwWorkChain_pks": "./gw/view_gw.ipynb",
    "CP2K_AdsorptionE": "view_ade.ipynb",
    "CP2K_GeoOpt": "view_geoopt.ipynb",
    "CP2K_Orbitals": "view_orb.ipynb",
    "CP2K_Pdos": "view_pdos.ipynb",
    "CP2K_STM": "view_stm.ipynb",
    "CP2K_AFM": "view_afm.ipynb",
    "CP2K_HRSTM": "view_hrstm.ipynb",
}


def thunmnail_raw(nrows=1, thumbnail=None, uuid=None, pk=None, description=""):
    # image with a link to structure export

    html = (
        '<td rowspan=%s><a target="_blank" href="./export_structure.ipynb?uuid=%s">'
        % (str(nrows), uuid)
    )
    html += '<img width="100px" src="data:image/png;base64,%s" title="PK%d: %s">' % (
        thumbnail,
        pk,
        description,
    )
    html += "</a></td>"
    return html


def link_to_viewer(description="", uuid="", label=""):
    the_viewer = VIEWERS[label]
    return '<li><a target="_blank" href="%s?uuid=%s"> %s </a></li>' % (
        the_viewer,
        uuid,
        description,
    )


def uuids_to_nodesdict(uuids):
    workflows = {}
    for uuid in uuids:
        try:
            node = load_node(uuid)
            if node.label in VIEWERS:
                if node.label in workflows:
                    workflows[node.label].append(node)
                else:
                    workflows[node.label] = [node]
        except NotExistent:
            pass

    return workflows


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

        # search with QB structures with extra "surfaces"
        qb = QueryBuilder()
        # "mtime": {"and": [{"<=": end_date}, {">": start_date}]},
        qb.append(
            StructureData,
            filters={
                "extras": {"has_key": "surfaces"},
            },
        )
        qb.order_by({StructureData: {"mtime": "desc"}})

        # for each structure in QB create a dictionary with info on the workflows computed on it
        print(qb.count())
        data = []
        for node_tuple in qb.iterall():
            node = node_tuple[0]
            workflows = uuids_to_nodesdict(node.extras["surfaces"])
            print(node.pk)
            if len(workflows) > 0:
                nrows = len(node.extras["surfaces"])
                if "thumbnail" not in node.extras:
                    print("setting thumbnail to ", node.pk)
                    node.set_extra(
                        "thumbnail", common_utils.thumbnail(ase_struc=node.get_ase())
                    )
                    print("done updating node", node.pk)
                    print("EXTRAS: ", node.extras)
                entry = {
                    "nrows": nrows,
                    "mtime": node.mtime.strftime("%d/%m/%y"),
                    "workflows": workflows,
                    "thumbnail": thunmnail_raw(
                        nrows=nrows,
                        thumbnail=node.extras["thumbnail"],
                        uuid=node.uuid,
                        pk=node.pk,
                        description="",
                    ),
                }
                print("ENTRY_DICT ", entry)
                data.append(entry)
        print("end loop")
        # populate the table with the data
        html = """<style>#aiida_results td,th {padding: 2px}</style>
        <table border=1 id="aiida_results" style="margin:0px">
        <thead>
        <tr>
            <th >Date last</th>
            <th >Calc. Type</th>
            <th >Description</th>
            <th >Thumbnail</th>
        </tr>
        </thead>
        <tbody>"""
        for entry in data:
            nrows1 = entry["nrows"]
            nrows_done = 0
            html += "<tr>"
            html += "<td rowspan=%s> %s  </td>" % (str(nrows1), entry["mtime"])
            for workflow in entry["workflows"]:
                nrowsw = len(entry["workflows"][workflow])
                html += "<td rowspan=%s>  %s </td>" % (str(nrowsw), workflow)
                html += "<td><ul>"
                for node in entry["workflows"][workflow]:
                    html += link_to_viewer(
                        description="PK-" + str(node.pk) + " " + node.description,
                        uuid=node.uuid,
                        label=node.label,
                    )
                html += "</td></ul>"
                if nrows_done == 0:
                    html += entry["thumbnail"]
                    nrows_done = 1
                for tr_empty in range(1, nrowsw):
                    html += "<tr></tr>"
            html += "</tr>"
        html += "</tbody></table>"

        self.results.value = html
