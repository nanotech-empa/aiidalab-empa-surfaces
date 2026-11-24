"""Viewer and chat share one read-only preparation and publication review."""

import html

import ipywidgets as ipw


class StmMicroscopyExport(ipw.VBox):
    def __init__(self, plotter):
        self.plotter = plotter
        self.review = None
        self.export_widget = None
        self.include_curve = ipw.Checkbox(
            description="Include image-area mean curves (at least two energies per series)",
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
        super().__init__(
            [
                ipw.HTML(
                    "<h3>Export simulation maps to openBIS microscopy</h3>"
                    "<p>Select series and energies above. Export uses the stored values, "
                    "independently of plot colors or normalization.</p>"
                ),
                self.include_curve,
                self.prepare_button,
                self.status,
                self.review_box,
            ]
        )

        self.add_class("stm-microscopy-export")

    def prepare(self):
        from aiida_openbis.export import make_export_widget, review_stm_imaging

        previous = self.export_widget
        self.export_widget = None
        self.review = None
        self.review_box.children = ()
        if previous is not None:
            from ._lifecycle import close_owned_widgets

            close_owned_widgets(previous)
        selection = self.plotter.microscopy_selection()
        review = review_stm_imaging(**selection, include_curve=self.include_curve.value)
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
        return {
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
        self.prepare_button.on_click(self._prepare_clicked, remove=True)
        from ._lifecycle import close_owned_widgets

        close_owned_widgets(
            self.include_curve, self.prepare_button, self.status, self.review_box
        )
        self.plotter = None
        super().close()
