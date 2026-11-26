"""Prepare a standard STM export with optional selected microscopy files."""

import html

import ipywidgets as ipw


class StmMicroscopyExport(ipw.VBox):
    def __init__(self, plotter):
        self._closed = False
        self.plotter = plotter
        self.review = None
        self.export_widget = None
        self.source_uuid = None
        self.source_geometry = None
        self._setting_area = False
        self._kept = {}
        self.keep_button = ipw.Button(description="Keep to export", disabled=True)
        self.kept_items = ipw.Dropdown(description="Kept for export", disabled=True)
        self.remove_kept = ipw.Button(description="Remove", disabled=True)
        self.clear_kept = ipw.Button(description="Clear all", disabled=True)
        self.kept_status = ipw.HTML()
        self.area_width = ipw.FloatText(description="Width [Å]", disabled=True)
        self.area_height = ipw.FloatText(description="Height [Å]", disabled=True)
        self.area_center_x = ipw.FloatText(description="Centre X [Å]", disabled=True)
        self.area_center_y = ipw.FloatText(description="Centre Y [Å]", disabled=True)
        self.reset_area_button = ipw.Button(
            description="Use simulation cell",
            disabled=True,
            layout=ipw.Layout(width="auto"),
        )
        self.area_info = ipw.HTML(
            "<p>Load a stored STM result to set the SXM area.</p>"
        )
        from .stm_sampling import StmSamplingPanel

        self.sampling = StmSamplingPanel(plotter, self.image_area)
        self.sampling_section = ipw.Accordion(
            children=[ipw.VBox([self.sampling, self.keep_button])], selected_index=None
        )
        self.sampling_section.set_title(
            0, "Extract STS point spectra or STM line profiles (optional)"
        )
        self.prepare_button = ipw.Button(
            description="Prepare export",
            icon="upload",
            layout=ipw.Layout(width="auto"),
        )
        self.status = ipw.HTML()
        self.review_box = ipw.VBox()
        self.prepare_button.on_click(self._prepare_clicked)
        self.reset_area_button.on_click(self._reset_area)
        self.keep_button.on_click(self._keep_clicked)
        self.remove_kept.on_click(self._remove_kept_clicked)
        self.clear_kept.on_click(self._clear_kept_clicked)
        for control in self._area_controls():
            control.observe(self._selection_changed, names="value")
        self.sampling.observe(self._sampling_changed, names="changed")
        super().__init__(
            [
                ipw.HTML(
                    "<h3>Export STM simulation to openBIS</h3>"
                    "<p>The standard export includes the AiiDA archive. Use "
                    "<b>Keep to export</b> to add the current map and any points or lines.</p>"
                ),
                ipw.HTML(
                    "<p><b>SXM image area</b> · Cartesian XY coordinates in Å. "
                    "Periodic directions are tiled from the native simulation cell. "
                    "Original maps are retained. Optional DAT files contain explicitly selected point spectra or line profiles.</p>"
                ),
                ipw.HBox(
                    [self.area_width, self.area_height],
                    layout=ipw.Layout(flex_flow="row wrap"),
                ),
                ipw.HBox(
                    [self.area_center_x, self.area_center_y],
                    layout=ipw.Layout(flex_flow="row wrap"),
                ),
                self.reset_area_button,
                self.area_info,
                self.sampling_section,
                self.kept_items,
                ipw.HBox([self.remove_kept, self.clear_kept]),
                self.kept_status,
                self.prepare_button,
                self.status,
                self.review_box,
            ]
        )

        self.add_class("stm-microscopy-export")

    def _area_controls(self):
        return (
            self.area_width,
            self.area_height,
            self.area_center_x,
            self.area_center_y,
        )

    def image_area(self):
        return {
            "width_angstrom": self.area_width.value,
            "height_angstrom": self.area_height.value,
            "center_angstrom": [self.area_center_x.value, self.area_center_y.value],
        }

    def _clear_review(self):
        previous = self.export_widget
        self.export_widget = None
        self.review = None
        self.review_box.children = ()
        if previous is not None:
            from ._lifecycle import close_owned_widgets

            close_owned_widgets(previous)

    def _selection_changed(self, change):
        if (
            not self._setting_area
            and isinstance(change, dict)
            and change.get("owner") in self._area_controls()
        ):
            self.sampling.refresh_area()
        if not self._setting_area and (
            self.export_widget is not None or self.review is not None
        ):
            self._clear_review()
            self.status.value = (
                "<p role='status'>Export options changed. Prepare a new review.</p>"
            )

    def _sampling_changed(self, _):
        geometry = self.sampling.geometry
        self.keep_button.disabled = not (
            self.source_geometry
            and geometry
            and geometry["source_uuid"] == self.source_uuid
            and self.sampling.series.value
        )

    def _refresh_kept(self):
        previous = self.kept_items.value
        options = [(entry["label"], key) for key, entry in self._kept.items()]
        self.kept_items.options = options
        if options:
            self.kept_items.value = (
                previous if previous in self._kept else options[0][1]
            )
        for control in (self.kept_items, self.remove_kept, self.clear_kept):
            control.disabled = not options
        counts = [
            len(entry["request"].get(field, []))
            for entry in self._kept.values()
            if entry["request"] is not None
            for field in ("points", "lines")
        ]
        self.kept_status.value = (
            f"<p>{len(options)} SXM map(s), {sum(counts)} DAT curve(s) kept for export.</p>"
            if options
            else "<p>Only the simulation and AiiDA archive will be exported.</p>"
        )

    def _keep_clicked(self, _):
        if self.keep_button.disabled:
            return
        try:
            series = self.sampling.series.value
            source = self.plotter.series_sources[series]
            index, bias = source["series_index"], int(self.sampling.bias.value)
            key = (index, bias)
            if source["source_uuid"] != self.source_uuid:
                raise ValueError("Choose a map from the current simulation.")
            requests = self.sampling.requests()
            request = requests[0] if requests else None
            if request and request["kind"] == "sts_points":
                request["reference_energy_index"] = bias
            previous = self._kept.get(key, {}).get("request")
            if previous and request:
                from aiida_openbis.stm_sampling import marker_color

                field = "points" if request["kind"] == "sts_points" else "lines"
                coordinates = (
                    ("position_angstrom",)
                    if field == "points"
                    else ("start_angstrom", "end_angstrom")
                )
                markers = list(previous[field])
                prefix = "P" if field == "points" else "L"
                serial = max(int(marker["id"][1:]) for marker in markers)
                for item in request[field]:
                    if not any(
                        all(saved[c] == item[c] for c in coordinates)
                        for saved in markers
                    ):
                        serial += 1
                        markers.append(
                            {
                                **item,
                                "id": f"{prefix}{serial}",
                                "color": marker_color(serial - 1),
                            }
                        )
                request = {**previous, field: markers}
            elif previous:
                request = previous
            voltage = self.plotter.series[series][2]["energies"][bias]
        except (ValueError, KeyError, IndexError) as error:
            self.status.value = "<p role='alert'>" + html.escape(str(error)) + "</p>"
            return
        self._kept[key] = {"label": f"{series} · {voltage:.8g} V", "request": request}
        self.sampling.series.value = ""
        self._clear_review()
        self._refresh_kept()
        self.status.value = "<p role='status'>Map kept for export. Choose the next map or prepare the export.</p>"

    def _remove_kept_clicked(self, _):
        self._kept.pop(self.kept_items.value, None)
        self._clear_review()
        self._refresh_kept()
        self.status.value = ""

    def _clear_kept_clicked(self, _):
        self._kept.clear()
        self._clear_review()
        self._refresh_kept()
        self.status.value = ""

    def _reset_area(self, _=None):
        if self.source_geometry is None:
            return
        self._clear_review()
        area = self.source_geometry["default_image_area"]
        self._setting_area = True
        try:
            self.area_width.value = area["width_angstrom"]
            self.area_height.value = area["height_angstrom"]
            self.area_center_x.value, self.area_center_y.value = area["center_angstrom"]
        finally:
            self._setting_area = False

    def set_source_uuid(self, source_uuid):
        """Read only plain cell/PBC metadata; never retain an ORM node."""
        from aiida_openbis.export import describe_stm_imaging

        self._clear_review()
        self.source_uuid = str(source_uuid)
        self.source_geometry = None
        self._kept.clear()
        self._refresh_kept()
        self.keep_button.disabled = True
        for control in (*self._area_controls(), self.reset_area_button):
            control.disabled = True
        geometry = describe_stm_imaging(self.source_uuid)
        self.source_geometry = geometry
        self._reset_area()
        for control in (*self._area_controls(), self.reset_area_button):
            control.disabled = False
        self.area_info.value = (
            "<p>Native CP2K cell periodicity: <b>"
            + html.escape(geometry["periodic_setting"])
            + "</b>. Default: smallest XY rectangle enclosing the simulation cell. "
            "Periodic directions repeat the calculated map. Beyond nonperiodic boundaries, areas without calculated values are left empty.</p>"
        )
        self.sampling.configure(geometry)

    def selection(self):
        return {
            "source_uuid": self.source_uuid,
            "selections": [
                {"series_index": series, "energy_index": energy}
                for series, energy in self._kept
            ],
            "kept_labels": [entry["label"] for entry in self._kept.values()],
        }

    def prepare(self):
        from aiida_openbis.export import make_export_widget, review_stm_imaging

        self._clear_review()
        if self.source_uuid is None:
            raise ValueError("Load a stored STM simulation before preparing an export.")
        selections = self.selection()["selections"]
        sampling = [
            entry["request"]
            for entry in self._kept.values()
            if entry["request"] is not None
        ]
        review = None
        if selections or sampling:
            review = review_stm_imaging(
                source_uuid=self.source_uuid,
                selections=selections,
                sampling=sampling,
                image_area=self.image_area(),
            )
        widget = make_export_widget(self.source_uuid, microscopy_review=review)
        self.review = review
        self.export_widget = widget
        self.review_box.children = (widget,)
        self.status.value = "<p role='status'>Review prepared. Select the destination and confirm Export to openBIS below.</p>"
        return self.context()

    def _prepare_clicked(self, _):
        try:
            self.prepare()
        except Exception as error:
            self.status.value = "<p role='alert'>" + html.escape(str(error)) + "</p>"

    def context(self):
        reviewed = self.review
        publication_plan = getattr(
            self.export_widget, "microscopy_publication_plan", lambda: None
        )()
        counts = (
            {
                "maps": len(reviewed["entries"]),
                "curve_datasets": len(reviewed["curve_groups"]),
                "extracted_curves": sum(
                    group["curve_count"] for group in reviewed["curve_groups"]
                ),
            }
            if reviewed is not None
            else {"maps": 0, "curve_datasets": 0, "extracted_curves": 0}
        )
        return {
            "microscopy_image_area": self.image_area()
            if self.source_geometry
            else None,
            "microscopy_source_geometry": self.source_geometry,
            "microscopy_dataset_counts": (
                publication_plan["dataset_counts"] if publication_plan else counts
            ),
            "microscopy_publication_plan": publication_plan,
            "microscopy_review": self.review,
            "stm_selection": self.selection(),
            "microscopy_sampling": self.sampling.context(),
            "message": self.status.value,
            "publication_status": getattr(
                getattr(self.export_widget, "export_status_html", None), "value", ""
            ),
            "microscopy_results": getattr(
                self.export_widget, "_microscopy_results", []
            ),
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.prepare_button.on_click(self._prepare_clicked, remove=True)
        self.reset_area_button.on_click(self._reset_area, remove=True)
        self.keep_button.on_click(self._keep_clicked, remove=True)
        self.remove_kept.on_click(self._remove_kept_clicked, remove=True)
        self.clear_kept.on_click(self._clear_kept_clicked, remove=True)
        for control in self._area_controls():
            control.unobserve(self._selection_changed, names="value")
        self.sampling.unobserve(self._sampling_changed, names="changed")
        from ._lifecycle import close_owned_widgets

        close_owned_widgets(
            *self._area_controls(),
            self.kept_items,
            self.remove_kept,
            self.clear_kept,
            self.kept_status,
            self.reset_area_button,
            self.area_info,
            self.sampling_section,
            self.prepare_button,
            self.status,
            self.review_box,
        )
        self.plotter = None
        super().close()
