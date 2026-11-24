"""Viewer and chat share one read-only preparation and publication review."""

import html

import ipywidgets as ipw


class StmMicroscopyExport(ipw.VBox):
    def __init__(self, plotter):
        self._closed = False
        self.plotter = plotter
        self.review = None
        self.export_widget = None
        self.source_geometry = None
        self._setting_area = False
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
        self.include_curve = ipw.Checkbox(
            description="Include original-map mean curves (at least two biases per series)",
            value=False,
            indent=False,
            layout=ipw.Layout(width="auto"),
        )
        self.prepare_button = ipw.Button(
            description="Prepare microscopy export",
            icon="upload",
            layout=ipw.Layout(width="auto"),
        )
        self.status = ipw.HTML()
        self.review_box = ipw.VBox()
        self.prepare_button.on_click(self._prepare_clicked)
        self.reset_area_button.on_click(self._reset_area)
        for control in self._area_controls():
            control.observe(self._selection_changed, names="value")
        self.include_curve.observe(self._selection_changed, names="value")
        super().__init__(
            [
                ipw.HTML(
                    "<h3>Export simulation maps to openBIS microscopy</h3>"
                    "<p>Select series and energies above. Export uses the stored values, "
                    "independently of plot colors or normalization.</p>"
                ),
                ipw.HTML(
                    "<p><b>SXM image area</b> · Cartesian XY coordinates in Å. "
                    "Periodic directions are tiled from the native simulation cell. "
                    "Original maps are retained; optional curves average the original grid.</p>"
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
                self.include_curve,
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

    def _selection_changed(self, _):
        if not self._setting_area and self.review is not None:
            self._clear_review()
            self.status.value = (
                "<p role='status'>Export options changed. Prepare a new review.</p>"
            )

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
        self.source_geometry = None
        for control in (*self._area_controls(), self.reset_area_button):
            control.disabled = True
        geometry = describe_stm_imaging(str(source_uuid))
        self.source_geometry = geometry
        self._reset_area()
        for control in (*self._area_controls(), self.reset_area_button):
            control.disabled = False
        self.area_info.value = (
            "<p>Native CP2K cell periodicity: <b>"
            + html.escape(geometry["periodic_setting"])
            + "</b>. Default: smallest Cartesian rectangle containing cell vectors A and B. "
            "Outside nonperiodic map directions: no data (NaN).</p>"
        )

    def prepare(self):
        from aiida_openbis.export import make_export_widget, review_stm_imaging

        self._clear_review()
        selection = self.plotter.microscopy_selection()
        if (
            self.source_geometry is None
            or self.source_geometry["source_uuid"] != selection["source_uuid"]
        ):
            self.set_source_uuid(selection["source_uuid"])
        review = review_stm_imaging(
            **selection,
            include_curve=self.include_curve.value,
            image_area=self.image_area(),
        )
        widget = make_export_widget(selection["source_uuid"], microscopy_review=review)
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
        counts = (
            {
                "maps": len(reviewed["entries"]),
                "curves": len({item["series_index"] for item in reviewed["entries"]})
                if reviewed["include_curve"]
                else 0,
            }
            if reviewed is not None
            else None
        )
        return {
            "microscopy_image_area": self.image_area()
            if self.source_geometry
            else None,
            "microscopy_source_geometry": self.source_geometry,
            "microscopy_dataset_counts": counts,
            "microscopy_review": self.review,
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
        for control in self._area_controls():
            control.unobserve(self._selection_changed, names="value")
        self.include_curve.unobserve(self._selection_changed, names="value")
        from ._lifecycle import close_owned_widgets

        close_owned_widgets(
            *self._area_controls(),
            self.reset_area_button,
            self.area_info,
            self.include_curve,
            self.prepare_button,
            self.status,
            self.review_box,
        )
        self.plotter = None
        super().close()
