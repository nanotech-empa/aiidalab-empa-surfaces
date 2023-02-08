import datetime
import importlib
import pathlib

import ipywidgets as ipw
from aiida.orm import QueryBuilder, WorkChainNode, load_node
from IPython.display import clear_output

FIELDS_DISABLE_DEFAULT = {
    "cell": True,
    "cell_opt": True,
    "volume": True,
    "extras": True,
}

AU_TO_EV = 27.211386245988
VIEWERS = {
    "Cp2kAdsorbedGwIcWorkChain_pks": {
        "viewer_path": "./gw/view_gw-ic.ipynb",
        "label": "GW-IC",
    },
    "Cp2kMoleculeOptGwWorkChain_pks": {
        "viewer_path": "./gw/view_gw.ipynb",
        "label": "GW",
    },
    "Cp2kAdsorptionEnergyWorkChain_pks": {
        "viewer_path": "./view_ade.ipynb",
        "label": "Ad.E",
    },
    "Cp2kOrbitalsWorkChain_pks": {
        "viewer_path": "../scanning_probe/orb/view_orb.ipynb",
        "label": "KS",
    },
    "Cp2kPdosWorkChain_pks": {
        "viewer_path": "../scanning_probe/pdos/view_pdos.ipynb",
        "label": "PDOS",
    },
    "Cp2kStmWorkChain_pks": {
        "viewer_path": "../scanning_probe/stm/view_stm.ipynb",
        "label": "STM",
    },
    "Cp2kOrbitalsWorkChain_uuids": {
        "viewer_path": "../scanning_probe/orb/view_orb.ipynb",
        "label": "KS",
    },
    "Cp2kPdosWorkChain_uuids": {
        "viewer_path": "../scanning_probe/pdos/view_pdos.ipynb",
        "label": "PDOS",
    },
    "Cp2kStmWorkChain_uuids": {
        "viewer_path": "../scanning_probe/stm/view_stm.ipynb",
        "label": "STM",
    },
}


class SearchCompletedWidget(ipw.VBox):
    def __init__(self, wlabel="", fields_disable={}):

        self.fields_disable = FIELDS_DISABLE_DEFAULT
        for fd in fields_disable:
            self.fields_disable[fd] = fields_disable[fd]
        # Search UI.
        self.wlabel = wlabel
        style = {"description_width": "150px"}
        layout = ipw.Layout(width="600px")
        self.inp_pks = ipw.Text(
            description="PKs",
            placeholder="e.g. 4062 4753 (space separated)",
            layout=layout,
            style=style,
        )
        self.inp_formula = ipw.Text(
            description="Formulas:",
            placeholder="e.g. C44H16 C36H4",
            layout=layout,
            style=style,
        )
        self.text_description = ipw.Text(
            description="Calculation Name: ",
            placeholder="e.g. keywords",
            layout=layout,
            style=style,
        )

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

        search_crit = [
            self.inp_pks,
            self.inp_formula,
            self.text_description,
            ipw.HBox([self.date_text, self.date_start, self.date_end]),
        ]

        button = ipw.Button(description="Search")

        self.results = ipw.HTML()
        self.info_out = ipw.Output()

        def on_click(b):
            with self.info_out:
                clear_output()
                self.search()

        button.on_click(on_click)

        self.show_comments_check = ipw.Checkbox(
            value=False, description="show comments", indent=False
        )

        buttons_hbox = ipw.HBox([button, self.show_comments_check])

        app = ipw.VBox(
            children=search_crit + [buttons_hbox, self.results, self.info_out]
        )

        super().__init__([app])

    def search(self):

        self.results.value = "searching..."
        scanning_probe_common = (
            pathlib.Path.home() / "apps" / "scanning_probe" / "common.py"
        )
        if scanning_probe_common.exists():
            loader = importlib.machinery.SourceFileLoader(
                "common", str(scanning_probe_common)
            )
            spec = importlib.util.spec_from_loader("common", loader)
            common = importlib.util.module_from_spec(spec)
            loader.exec_module(common)
            common.preprocess_spm_calcs(
                workchain_list=[
                    "STMWorkChain",
                    "PdosWorkChain",
                    "AfmWorkChain",
                    "HRSTMWorkChain",
                    "OrbitalWorkChain",
                ]
            )
            self.fields_disable["extras"] = False
        else:
            print("Warning: scanning_probe app not found, skipping spm preprocess.")
            self.fields_disable["extras"] = True

        self.value = "searching..."

        # html table header
        html = "<style>#aiida_results td,th {padding: 2px}</style>"
        html += '<table border=1 id="aiida_results" style="margin:0px"><tr>'
        html += "<th>PK</th>"
        html += "<th>Creation Time</th>"
        html += "<th >Formula</th>"
        html += "<th>Calculation name</th>"
        html += "<th>Energy(eV)</th>"
        html += "<th>Abs. mag.</th>"
        if not self.fields_disable["cell"]:
            html += "<th>Cell</th>"
        if not self.fields_disable["cell_opt"]:
            html += "<th>Cell optimized</th>"
        if not self.fields_disable["volume"]:
            html += "<th>Volume</th>"
        html += '<th style="width: 100px">Structure</th>'
        if self.show_comments_check.value:
            html += "<th>Comments</th>"
        if not self.fields_disable["extras"]:
            html += '<th style="width: 10%">Extras</th>'
        html += "</tr>"

        # query AiiDA database
        filters = {}
        filters["label"] = self.wlabel
        filters["attributes.exit_status"] = 0

        pk_list = self.inp_pks.value.strip().split()
        if pk_list:
            filters["id"] = {"in": pk_list}

        formula_list = self.inp_formula.value.strip().split()
        if self.inp_formula.value:
            filters["extras.formula"] = {"in": formula_list}

        if len(self.text_description.value) > 1:
            filters["description"] = {"like": f"%{self.text_description.value}%"}

        try:  # If the date range is valid, use it for the search
            start_date = datetime.datetime.strptime(self.date_start.value, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(
                self.date_end.value, "%Y-%m-%d"
            ) + datetime.timedelta(hours=24)
        except ValueError:  # Otherwise revert to the standard (i.e. last 10 days)
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=20)

            date_start.value = start_date.strftime("%Y-%m-%d")
            date_end.value = end_date.strftime("%Y-%m-%d")

        filters["ctime"] = {"and": [{"<=": end_date}, {">": start_date}]}

        qb = QueryBuilder()
        qb.append(WorkChainNode, filters=filters)
        qb.order_by({WorkChainNode: {"ctime": "desc"}})

        for node_tuple in qb.iterall():
            node = node_tuple[0]
            thumbnail = ""
            try:
                thumbnail = node.extras["thumbnail"]
            except KeyError:
                pass
            description = node.description
            opt_structure = node.outputs.output_structure

            # Find all extra calculations done on the optimized geometry.
            extra_calc_links = ""
            st_extras = opt_structure.extras

            # --------------------------------------------------
            # Add links to SPM calcs.
            try:
                import apps.scanning_probe.common

                extra_calc_links += apps.scanning_probe.common.create_viewer_link_html(
                    st_extras, "../"
                )
            except Exception:
                pass
            ### --------------------------------------------------

            ### --------------------------------------------------

            ## add links to computed properties
            for property in VIEWERS:
                if property in st_extras:
                    calc_links_str = ""
                    nr = 0
                    for pk_or_uuid in st_extras[property]:
                        print(opt_structure.pk,pk_or_uuid)
                        pk = load_node(pk_or_uuid).pk
                        nr += 1
                        calc_links_str += (
                            "<a target='_blank' href='%s?pk=%s'>%s %s</a><br />"
                            % (
                                VIEWERS[property]["viewer_path"],
                                pk,
                                VIEWERS[property]["label"],
                                nr,
                            )
                        )
                    extra_calc_links += calc_links_str

            ### --------------------------------------------------

            extra_calc_area = (
                "<div id='wrapper' style='overflow-y:auto; height:100px; line-height:1.5;'> %s </div>"
                % extra_calc_links
            )

            out_params = node.outputs.output_parameters
            abs_mag = "-"
            if "integrated_abs_spin_dens" in dict(out_params):
                abs_mag = f"{out_params['integrated_abs_spin_dens'][-1]:.2f}"

            # append table row
            html += "<tr>"
            html += f"""<td><a target="_blank" href="../aiidalab-widgets-base/notebooks/process.ipynb?id={node.pk}">{node.pk}</a></td>"""
            html += "<td>%s</td>" % node.ctime.strftime("%Y-%m-%d %H:%M")
            try:
                html += (
                    "<td>%s</td>" % node.extras["formula"]
                )  # opt_structure.get_formula()
            except KeyError:
                html += "<td>%s</td>" % opt_structure.get_formula()
            html += "<td>%s</td>" % node.description
            html += "<td>%.4f</td>" % (float(out_params["energy"]) * AU_TO_EV)
            html += f"<td>{abs_mag}</td>"
            if not self.fields_disable["cell"]:
                cell = ""
                for cellpar in [
                    "cell_a_angs",
                    "cell_b_angs",
                    "cell_c_angs",
                    "cell_alp_deg",
                    "cell_bet_deg",
                    "cell_gam_deg",
                ]:
                    cell += " " + str(
                        node.outputs.output_parameters["motion_step_info"][cellpar][-1]
                    )
                html += "<td>%s</td>" % cell
            if not self.fields_disable["cell_opt"]:
                html += "<td>%s</td>" % node.outputs.output_parameters["run_type"]
            if not self.fields_disable["volume"]:
                html += (
                    "<td>%f</td>"
                    % node.outputs.output_parameters["motion_step_info"][
                        "cell_vol_angs3"
                    ][-1]
                )
            # image with a link to structure export
            html += (
                '<td><a target="_blank" href="./export_structure.ipynb?uuid=%s">'
                % opt_structure.uuid
            )
            html += (
                '<img width="100px" src="data:image/png;base64,%s" title="PK%d: %s">'
                % (thumbnail, opt_structure.pk, description)
            )
            html += "</a></td>"

            if self.show_comments_check.value:
                comment_area = "<div id='wrapper' style='overflow-y:auto; height:100px; line-height:1.5;'>"
                comment_area += (
                    '<a target="_blank" href="./comments.ipynb?pk=%s">add/view</a><br>'
                    % node.pk
                )
                for comment in node.get_comments():
                    comment_area += (
                        "<hr style='padding:0px; margin:0px;' />"
                        + comment.content.replace("\n", "<br>")
                    )
                comment_area += "</div>"
                html += "<td>%s</td>" % (comment_area)

            if not self.fields_disable["extras"]:
                html += "<td>%s</td>" % extra_calc_area
            html += "</td>"
            html += "</tr>"

        html += "</table>"
        html += "Found %d matching entries.<br>" % qb.count()

        self.results.value = html
