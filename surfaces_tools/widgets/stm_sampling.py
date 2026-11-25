"""Visual STS point spectra and STM line profiles; no global-image averages."""

import base64
import copy
import io

import anywidget
import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
from traitlets import Int, List, Unicode

from ._lifecycle import close_owned_widgets


class SamplingMap(anywidget.AnyWidget):
    image = Unicode("").tag(sync=True)
    extent = List([0.0, 1.0, 0.0, 1.0]).tag(sync=True)
    marks = List([]).tag(sync=True)
    mode = Unicode("points").tag(sync=True)
    _esm = r"""
    function render({model, el}) {
      const canvas = document.createElement('canvas');
      canvas.className = 'stm-sampling-map';
      canvas.setAttribute('role', 'img');
      canvas.setAttribute('aria-label', 'Map for selecting STS points or STM lines');
      canvas.style.cssText = 'display:block;max-width:100%;cursor:crosshair;background:#fafafa;';
      const coordinate = document.createElement('div');
      coordinate.setAttribute('aria-live', 'polite');
      const img = new Image();
      el.append(canvas, coordinate);
      function pixel(xy) {
        const [x0,x1,y0,y1] = model.get('extent');
        return [(xy[0]-x0)/(x1-x0)*canvas.width, (y1-xy[1])/(y1-y0)*canvas.height];
      }
      function draw() {
        const [x0,x1,y0,y1] = model.get('extent');
        canvas.width = 800; canvas.height = Math.max(1, Math.round(800*(y1-y0)/(x1-x0)));
        const ctx = canvas.getContext('2d');
        if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        for (const mark of model.get('marks')) {
          ctx.strokeStyle = mark.color; ctx.fillStyle = mark.color; ctx.lineWidth = 3;
          const p = pixel(mark.position_angstrom || mark.start_angstrom);
          if (mark.end_angstrom) {
            const q = pixel(mark.end_angstrom);
            ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.stroke();
            ctx.fillRect(q[0]-4,q[1]-4,8,8);
          }
          ctx.beginPath();ctx.arc(p[0],p[1],6,0,2*Math.PI);ctx.fill();
          ctx.font = 'bold 17px sans-serif';ctx.strokeStyle = 'white';ctx.lineWidth = 4;
          ctx.strokeText(mark.id,p[0]+9,p[1]-9);ctx.fillText(mark.id,p[0]+9,p[1]-9);
        }
      }
      function coords(event) {
        const rect = canvas.getBoundingClientRect();
        const [x0,x1,y0,y1] = model.get('extent');
        return [x0+(event.clientX-rect.left)/rect.width*(x1-x0),y1-(event.clientY-rect.top)/rect.height*(y1-y0)];
      }
      function click(event) {if (img.naturalWidth) model.send({action:'pick',position_angstrom:coords(event)});}
      function move(event) {const [x,y]=coords(event);coordinate.textContent=`X ${x.toFixed(4)} Å · Y ${y.toFixed(4)} Å`;}
      function imageChanged() {img.src=model.get('image');}
      img.onload=draw;canvas.addEventListener('click',click);canvas.addEventListener('pointermove',move);
      model.on('change:image',imageChanged);model.on('change:extent',draw);model.on('change:marks',draw);
      imageChanged();draw();
      return () => {model.off('change:image',imageChanged);model.off('change:extent',draw);model.off('change:marks',draw);canvas.removeEventListener('click',click);canvas.removeEventListener('pointermove',move);img.onload=null;};
    }
    export default {render};
    """


class StmSamplingPanel(ipw.VBox):
    changed = Int(0)

    def __init__(self, plotter, image_area):
        self._closed = False
        self.plotter, self.image_area = plotter, image_area
        self.geometry = None
        self._ready = False
        self._loading = False
        self._markers, self._pending = [], None
        self._serial = 0
        self._groups = []
        self.series = ipw.Dropdown(
            description="Analyse",
            options=[("Choose a stored STS or STM series", "")],
            layout=ipw.Layout(width="95%"),
        )
        self.bias = ipw.SelectionSlider(
            description="Map bias [V]",
            options=[("0", 0)],
            continuous_update=False,
            disabled=True,
            layout=ipw.Layout(width="95%"),
        )
        self.load_button = ipw.Button(
            description="Open sampling map",
            disabled=True,
            layout=ipw.Layout(width="auto"),
        )
        self.instructions = ipw.HTML(
            "<p>STS: mark points to obtain dI/dV versus bias. STM: mark two endpoints to obtain a distance profile at the selected bias.</p>"
        )
        self.map = SamplingMap(layout=ipw.Layout(width="100%"))
        self.map.layout.display = "none"
        self.selected = ipw.Select(
            description="Markers", options=[], rows=3, layout=ipw.Layout(width="95%")
        )
        self.remove_button = ipw.Button(
            description="Remove selected marker",
            disabled=True,
            layout=ipw.Layout(width="auto"),
        )
        self.clear_button = ipw.Button(description="Clear points/lines", disabled=True)
        self.status = ipw.HTML()
        self.curve_plot = ipw.Image(format="png", layout=ipw.Layout(max_width="100%"))
        self.download_button = ipw.Button(
            description="Download DAT files",
            disabled=True,
            layout=ipw.Layout(width="auto"),
        )
        self.download = ipw.HTML()
        self.x = ipw.FloatText(description="X [Å]")
        self.y = ipw.FloatText(description="Y [Å]")
        self.add_coordinate = ipw.Button(description="Add coordinate", disabled=True)
        coordinates = ipw.Accordion(
            children=[
                ipw.VBox(
                    [
                        ipw.HBox(
                            [self.x, self.y], layout=ipw.Layout(flex_flow="row wrap")
                        ),
                        self.add_coordinate,
                    ]
                )
            ],
            selected_index=None,
        )
        coordinates.set_title(0, "Enter exact coordinates instead of clicking")
        self.series.observe(self._series_changed, names="value")
        self.bias.observe(self._bias_changed, names="value")
        self.selected.observe(self._selected_changed, names="value")
        self.load_button.on_click(self._load_clicked)
        self.clear_button.on_click(self._clear_clicked)
        self.remove_button.on_click(self._remove_clicked)
        self.download_button.on_click(self._download_clicked)
        self.add_coordinate.on_click(self._coordinate_clicked)
        self.map.on_msg(self._map_message)
        super().__init__(
            [
                ipw.HTML("<h3>Point spectra and line profiles</h3>"),
                self.instructions,
                self.series,
                self.bias,
                self.load_button,
                self.map,
                coordinates,
                self.selected,
                ipw.HBox(
                    [self.remove_button, self.clear_button],
                    layout=ipw.Layout(flex_flow="row wrap"),
                ),
                self.status,
                self.curve_plot,
                self.download_button,
                self.download,
            ]
        )
        self.add_class("stm-local-sampling")

    def configure(self, geometry):
        self.geometry = copy.deepcopy(geometry)
        options = [
            (label, label)
            for label, source in self.plotter.series_sources.items()
            if source["source_uuid"] == geometry["source_uuid"]
            and self.plotter.series[label][1]["type"]
            in {
                "const-height sts",
                "const-isovalue sts",
                "const-height stm",
                "const-isovalue stm",
            }
        ]
        self.series.options = [("Choose a stored STS or STM series", ""), *options]
        self.series.value = ""
        self.clear()

    def _archive(self):
        if not self.series.value or self.geometry is None:
            raise ValueError("Choose a stored STS or STM series.")
        values, info, general = self.plotter.series[self.series.value]
        index = self.plotter.series_sources[self.series.value]["series_index"]
        # Keep the original index; no live ORM nodes or normalized display values.
        series = [{} for _ in range(index + 1)]
        series[index] = info
        data = {index: values}
        return {
            "general": general,
            "series": series,
            "data": data,
            "simulation": self.geometry,
        }, index

    def _series_changed(self, _):
        self._ready = False
        self.map.layout.display = "none"
        self.clear()
        self.load_button.disabled = not bool(self.series.value)
        if self.series.value:
            data, info, general = self.plotter.series[self.series.value]
            self._loading = True
            self.bias.options = [
                (f"{energy:.6g}", i) for i, energy in enumerate(general["energies"])
            ]
            selected = list(self.plotter.select_indexes_function())
            self.bias.value = int(selected[0]) if selected else 0
            self._loading = False
            self.bias.disabled = False
            self.instructions.value = (
                "<p><b>STS point spectra:</b> click points on the map. Every point has a matching coloured dI/dV(V) curve using all stored energies. Map bias changes the reference image, not the spectrum. Drawn spectra are included in the export review.</p>"
                if "sts" in info["type"]
                else "<p><b>STM line profiles:</b> click a start and an end. Each line has a matching coloured height or current profile versus distance. Changing the map bias updates the profile; export records that bias. Drawn profiles are included in the export review.</p>"
            )

    def _render_map(self):
        from aiida_openbis.nanonis import archive_grid, rectangular_raster

        archive, index = self._archive()
        values = archive["data"][index][self.bias.value]
        origin, steps = archive_grid(archive["general"], values.shape)
        area = self.image_area()
        raster = rectangular_raster(
            values,
            origin,
            steps,
            area,
            cell_vectors_angstrom=self.geometry["in_plane_vectors_angstrom"],
            periodic_axes=self.geometry["periodic_axes"],
        )
        stream = io.BytesIO()
        plt.imsave(
            stream, raster.values.T, origin="lower", cmap="gist_heat", format="png"
        )
        self.map.image = (
            "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
        )
        cx, cy = area["center_angstrom"]
        w, h = area["width_angstrom"], area["height_angstrom"]
        self.map.extent = [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]
        self.map.mode = (
            "points" if "sts" in archive["series"][index]["type"] else "lines"
        )
        self.map.layout.display = "block"

    def _load_clicked(self, _):
        try:
            self._render_map()
            self._ready = True
            self.clear_button.disabled = False
            self.add_coordinate.disabled = False
            self.status.value = (
                "<p>Click a point.</p>"
                if self.map.mode == "points"
                else "<p>Click the start of a line.</p>"
            )
        except Exception as error:
            self._error(error)

    def _bias_changed(self, _):
        if self._ready and not self._loading:
            try:
                self._render_map()
                self._derive()
            except Exception as error:
                self._error(error)

    def refresh_area(self):
        if self._ready:
            try:
                self._render_map()
            except Exception as error:
                self._error(error)

    def _map_message(self, _, message, buffers):
        if message.get("action") == "pick":
            self.add_position(message["position_angstrom"])

    def _coordinate_clicked(self, _):
        self.add_position([self.x.value, self.y.value])

    def add_position(self, position):
        if not self._ready:
            return
        from aiida_openbis.stm_sampling import marker_color

        xy = np.asarray(position, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            return
        if self.map.mode == "points":
            self._serial += 1
            self._markers.append(
                {
                    "id": f"P{self._serial}",
                    "color": marker_color(self._serial - 1),
                    "position_angstrom": xy.tolist(),
                }
            )
        elif self._pending is None:
            self._pending = xy.tolist()
            self.map.marks = [
                *self._markers,
                {
                    "id": f"L{self._serial + 1}",
                    "color": marker_color(self._serial),
                    "start_angstrom": xy.tolist(),
                },
            ]
            self.status.value = "<p>Click the end of the line.</p>"
            self.changed += 1
            return
        else:
            self._serial += 1
            self._markers.append(
                {
                    "id": f"L{self._serial}",
                    "color": marker_color(self._serial - 1),
                    "start_angstrom": self._pending,
                    "end_angstrom": xy.tolist(),
                }
            )
            self._pending = None
        self.map.marks = copy.deepcopy(self._markers)
        self._update_options()
        self._derive()

    def _update_options(self):
        self.selected.options = [
            (
                m["id"]
                + ": "
                + str(
                    m.get(
                        "position_angstrom",
                        [m.get("start_angstrom"), m.get("end_angstrom")],
                    )
                )
                + " Å",
                m["id"],
            )
            for m in self._markers
        ]
        self.remove_button.disabled = not bool(self._markers)

    def _selected_changed(self, _):
        self.remove_button.disabled = self.selected.value is None

    def _remove_clicked(self, _):
        self._markers = [m for m in self._markers if m["id"] != self.selected.value]
        self.map.marks = copy.deepcopy(self._markers)
        self._update_options()
        self._derive()

    def _request(self):
        if not self._markers:
            return []
        _, index = self._archive()
        if self.map.mode == "points":
            return [
                {
                    "kind": "sts_points",
                    "series_index": index,
                    "points": copy.deepcopy(self._markers),
                }
            ]
        return [
            {
                "kind": "stm_lines",
                "series_index": index,
                "energy_index": int(self.bias.value),
                "lines": copy.deepcopy(self._markers),
            }
        ]

    def requests(self, strict=True):
        if strict and self._pending is not None:
            raise ValueError("Finish or clear the pending line before exporting.")
        if strict and self._markers and not self._groups:
            raise ValueError("Fix or remove invalid sampling markers before exporting.")
        return [copy.deepcopy(g["request"]) for g in self._groups]

    def _derive(self):
        from aiida_openbis.stm_sampling import derive_sampling

        self.download.value = ""
        self.download_button.disabled = True
        self._groups = []
        self.curve_plot.value = b""
        self.changed += 1
        if not self._markers:
            return
        try:
            archive, _ = self._archive()
            groups = derive_sampling(archive, self._request())
            self._groups = groups
            group = groups[0]
            fig, ax = plt.subplots(figsize=(8, 4))
            for curve in group["curves"]:
                ax.plot(curve["x"], curve["y"], color=curve["color"], label=curve["id"])
            ax.set_xlabel("Bias [V]" if group["x_unit"] == "V" else "Distance [Å]")
            unit = group["quantity"]["native_unit"]
            label = (
                "Height above highest atom [Å]"
                if unit == "angstrom"
                else (
                    "Simulated dI/dV [native units]"
                    if self.map.mode == "points"
                    else "Simulated current [native units]"
                )
            )
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            stream = io.BytesIO()
            fig.savefig(stream, format="png", dpi=120)
            plt.close(fig)
            self.curve_plot.value = stream.getvalue()
            self.status.value = (
                f"<p>{len(group['curves'])} coloured curve(s), sampled from the original data. "
                + (
                    "All stored energies; no spatial averaging."
                    if self.map.mode == "points"
                    else f"Bias {group['request']['bias_voltage_v']:.6g} V; distance includes both endpoints."
                )
                + "</p>"
            )
            self.download_button.disabled = False
            self.changed += 1
        except Exception as error:
            self._error(error)

    def _download_clicked(self, _):
        try:
            from aiida_openbis.export import download_stm_sampling

            archive = download_stm_sampling(
                self.geometry["source_uuid"], self.requests()
            )
            self.download.value = (
                '<a download="stm-local-curves.zip" href="data:application/zip;base64,'
                + base64.b64encode(archive).decode()
                + '">Download DAT files and sampling coordinates</a>'
            )
        except Exception as error:
            self._error(error)

    def _error(self, error):
        import html

        self.status.value = "<p role='alert'>" + html.escape(str(error)) + "</p>"

    def clear(self):
        self._markers = []
        self._pending = None
        self._serial = 0
        self._groups = []
        self.map.marks = []
        self.selected.options = []
        self.curve_plot.value = b""
        self.download.value = ""
        self.download_button.disabled = True
        self.changed += 1

    def _clear_clicked(self, _):
        self.clear()

    def context(self):
        return {
            "source_uuid": self.geometry["source_uuid"] if self.geometry else None,
            "series": self.series.value,
            "map_bias_index": int(self.bias.value),
            "mode": self.map.mode,
            "markers": copy.deepcopy(self._markers),
            "requests": self.requests(strict=False),
            "pending_line_start": copy.deepcopy(self._pending),
            "valid": self._pending is None
            and (not self._markers or bool(self._groups)),
            "message": self.status.value,
        }

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.series.unobserve(self._series_changed, names="value")
        self.bias.unobserve(self._bias_changed, names="value")
        self.selected.unobserve(self._selected_changed, names="value")
        self.map.on_msg(self._map_message, remove=True)
        for control, callback in [
            (self.load_button, self._load_clicked),
            (self.clear_button, self._clear_clicked),
            (self.remove_button, self._remove_clicked),
            (self.download_button, self._download_clicked),
            (self.add_coordinate, self._coordinate_clicked),
        ]:
            control.on_click(callback, remove=True)
        close_owned_widgets(*self.children)
        self.plotter = None
        self.geometry = None
        super().close()
