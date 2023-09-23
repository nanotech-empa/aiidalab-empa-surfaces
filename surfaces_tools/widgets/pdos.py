import base64
import io
import math

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import traitlets as tl
from aiida import common, orm
from IPython.display import clear_output, display

from ..utils import pdos_postprocess, spm
from . import stack


def get_colors(colors, n_points):
    colors = np.linspace(colors[0], colors[1], n_points).astype(int)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]


def make_image_link(figure, text="PNG", data_format="png"):
    image_data = io.BytesIO()
    figure.savefig(image_data, format=data_format, dpi=300, bbox_inches="tight")
    image_data.seek(0)  # rewind the data
    image_file = base64.b64encode(image_data.getvalue()).decode()

    filename = f"pdos.{data_format}"

    html = f'<a download="{filename}" href="'
    html += f'data:image/{data_format};name={filename};base64,{image_file}"'
    html += ' id=f"pdos_{data_format}_link"'
    html += f' target="_blank">{text}</a>'

    return html


def make_txt_link(collected_data):
    headers, data = collected_data
    header = ", ".join(headers)
    tempio = io.BytesIO()
    np.savetxt(tempio, data, header=header, fmt="%.4e", delimiter=", ")
    enc_file = base64.b64encode(tempio.getvalue()).decode()

    filename = "pdos.txt"

    html = f'<a download="{filename}" href="'
    html += f'data:chemical/txt;name={filename};base64,{enc_file}"'
    html += ' id="export_link"'
    html += ' target="_blank">TXT</a>'

    return html


class _BaseSelectionWidget(stack.HorizontalItemWidget):
    data = tl.Dict(allow_none=True)
    options = tl.Union([tl.Dict(), tl.List()], allow_none=True)

    def __init__(self, spin=0, color="black", factor=1.0, **kwargs):
        self._spin_selector = ipw.ToggleButtons(
            options=[("up", 0), ("down", 1)],
            value=spin,
            description="Spin:",
            disabled=False,
            style={"description_width": "auto", "button_width": "60px"},
            layout=ipw.Layout(width="180px"),
        )

        self._data_selection = ipw.Dropdown(
            options=[],
            description="Series:",
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="250px"),
        )

        self._color_picker = ipw.ColorPicker(
            concise=False,
            description="Color:",
            value=color,
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="200px"),
        )

        self._norm_factor = ipw.FloatText(
            value=factor,
            step=0.01,
            description="Norm factor:",
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="150px"),
        )

        super().__init__(
            children=[
                self._spin_selector,
                self._data_selection,
                self._color_picker,
                self._norm_factor,
            ]
        )


class _BaseStackWidget(stack.VerticalStackWidget):
    data = tl.Dict(allow_none=True)
    options = tl.Union([tl.Dict(), tl.List()], allow_none=True)

    def add_item(self, _):
        self.items += (
            self.item_class(
                color="black",
                factor=1.0,
            ),
        )
        tl.dlink((self, "options"), (self.items[-1], "options"))
        tl.dlink((self, "data"), (self.items[-1], "data"))


class PdosSelectionWidget(_BaseSelectionWidget):
    def __init__(self, spin=0, color="black", factor=1.0, **kwargs):
        super().__init__(spin, color, factor, **kwargs)
        tl.dlink((self, "options"), (self._data_selection, "options"))

    def return_all_values(self):
        return (
            self._data_selection.label,
            self._color_picker.value,
            self._norm_factor.value,
            self._spin_selector.value,
            self.data[self._data_selection.value][self._spin_selector.value],
        )

    @tl.observe("data")
    def _on_data_change(self, change):
        if change["new"]:
            # If there is only one spin, disable the spin selector.
            if len(self.data["tdos"]) == 1:
                self._spin_selector.disabled = True
                self._spin_selector.value = 0


class PdosStackWidget(_BaseStackWidget):
    workchain = tl.Instance(orm.WorkChainNode, allow_none=True)

    @tl.observe("workchain")
    def _on_workchain_change(self, change):
        workchain = change["new"]
        data = pdos_postprocess.process_pdos_files(workchain)

        self.options = {
            "Total DOS": "tdos",
            "Molecule PDOS": "mol",
            **{
                f"kind {name.split('_')[-1]}": name
                for name in data
                if name.startswith("kind_")
            },
        }

        labels_are_present = (
            "molecule" in workchain.inputs.pdos_lists[0]
            and len(workchain.inputs.pdos_lists) > 1
        )

        if labels_are_present:
            labels = [sel[1] for sel in workchain.inputs.pdos_lists[1:]]
            self.options.update(
                {
                    labels[int(name[4:]) - 2]: name
                    for name in data
                    if name.startswith("sel")
                }
            )
        else:
            self.options.update(
                {
                    f"selection {name.split('_')[-1]}": name
                    for name in data
                    if name.startswith("sel")
                }
            )

        # Trigger the data change.
        self.data = data

    @tl.observe("data")
    def _on_data_change(self, change):
        n_spin = len(self.data["tdos"])
        if self.data:
            self.items = ()
            for spin in range(n_spin):
                total_pdos = PdosSelectionWidget(color="lightgray", factor=0.02)
                tl.dlink((self, "data"), (total_pdos, "data"))
                tl.dlink((self, "options"), (total_pdos, "options"))
                total_pdos._data_selection.label = "Total DOS"
                total_pdos._spin_selector.value = spin

                molecule_pdos = PdosSelectionWidget(
                    color="black",
                )
                tl.dlink((self, "data"), (molecule_pdos, "data"))
                tl.dlink((self, "options"), (molecule_pdos, "options"))
                molecule_pdos._data_selection.label = "Molecule PDOS"
                molecule_pdos._spin_selector.value = spin

                self.items += (total_pdos, molecule_pdos)


class OverlapSelectionWidget(_BaseSelectionWidget):
    def __init__(self, spin=0, color="black", factor=1.0, **kwargs):
        super().__init__(spin, color, factor, **kwargs)
        self._spin_selector.observe(self._on_spin_change, names="value")

    @tl.observe("options")
    def _on_spin_change(self, _=None):
        self._data_selection.options = self.options[self._spin_selector.value]

    def return_all_values(self):
        return (
            self._data_selection.label,
            self._color_picker.value,
            self._norm_factor.value,
            self._spin_selector.value,
            self.data["overlap_matrix"][self._spin_selector.value][
                :, self._data_selection.value
            ],
        )

    @tl.observe("data")
    def _on_data_change(self, change):
        if change["new"]:
            # If there is only one spin, disable the spin selector.
            if self.data["nspin_g2"] == 1:
                self._spin_selector.disabled = True
                self._spin_selector.value = 0


class OverlapStackWidget(_BaseStackWidget):
    @tl.observe("data")
    def _on_data_change(self, change):
        n_spin = self.data["nspin_g2"]
        options = []
        for spin in range(n_spin):
            n_orbitals = len(self.data["orb_indexes_g2"][spin])
            options.append(
                {
                    pdos_postprocess.get_full_orbital_label(spin, i, self.data): i
                    for i in range(n_orbitals)
                }
            )

        self.options = options

        color_palette_homo = (
            (255, 0, 0),  # Red (#ff0000)
            (255, 255, 0),  # Yellow (#ffff00)
        )
        color_palette_lumo = (
            (200, 255, 255),  # No idea (#00ffff)
            (30, 100, 200),  # something blueish (#0000ff)
        )

        items = ()
        for spin in range(n_spin):
            n_homo = sum("HOMO" in s for s in self.options[spin])
            colors_homo = get_colors(color_palette_homo, n_homo)
            n_lumo = sum("LUMO" in s for s in self.options[spin])
            colors_lumo = get_colors(color_palette_lumo, n_lumo)
            for i, key in enumerate(self.options[spin].keys()):
                overlap_item = OverlapSelectionWidget(
                    color=colors_homo[i] if "HOMO" in key else colors_lumo[i - n_homo],
                    spin=spin,
                    factor=1.0,
                )

                tl.dlink((self, "data"), (overlap_item, "data"))
                tl.dlink((self, "options"), (overlap_item, "options"))
                overlap_item._data_selection.label = key
                items += (overlap_item,)

        self.items = items


class PdosOverlapViewerWidget(ipw.VBox):
    def __init__(self):
        style = {"description_width": "140px"}
        layout = {"width": "50%"}
        self._fwhm_slider = ipw.FloatSlider(
            value=0.10,
            min=0.01,
            max=0.2,
            step=0.01,
            description="Broadening fwhm (eV):",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".2f",
            style=style,
            layout=layout,
        )
        self._energy_range_slider = ipw.FloatRangeSlider(
            value=[0.0, 0.0],
            min=0.0,
            max=0.0,
            step=0.1,
            description="Energy range (eV):",
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            style=style,
            layout=layout,
        )
        self._plot_output = ipw.Output()
        self._geometry_info = ipw.HTML()
        self._projections = PdosStackWidget(
            item_class=PdosSelectionWidget, add_button_text="Add PDOS"
        )

        self._overlap = OverlapStackWidget(
            item_class=OverlapSelectionWidget, add_button_text="Add Overlap"
        )
        plot_button = ipw.Button(description="Plot")
        plot_button.on_click(self.make_plot)

        self.cumulative_plot = ipw.Checkbox(description="Cumulative plot", value=True)

        clear_button = ipw.Button(description="Clear")
        clear_button.on_click(self.clear_plot)

        super().__init__(
            [
                self._fwhm_slider,
                self._energy_range_slider,
                self._geometry_info,
                self._plot_output,
                self._projections,
                self._overlap,
                ipw.HBox([plot_button, self.cumulative_plot, clear_button]),
            ]
        )

    def load_data(self, reference=None):
        workchain = orm.load_node(pk=reference)

        try:
            self._geometry_info.value = spm.get_slab_calc_info(
                workchain.inputs.structure
            )
        except common.NotExistentAttributeError:
            self._geometry_info.value = spm.get_slab_calc_info(
                workchain.inputs.slabsys_structure
            )

        # Dealing with the projections data.
        self._projections.workchain = workchain

        # Dealing with the overlap data.
        overlap_calculation = spm.get_calc_by_label(workchain, "overlap")
        with overlap_calculation.outputs.retrieved.open(
            "overlap.npz", mode="rb"
        ) as fhandle:
            self._overlap.data = pdos_postprocess.load_overlap_npz(fhandle.name)

        # Initialize selections.
        energy_lim = [
            float(workchain.inputs.overlap_params["--emin1"]),
            float(workchain.inputs.overlap_params["--emax1"]),
        ]
        self._energy_range_slider.min = energy_lim[0]
        self._energy_range_slider.max = energy_lim[1]
        self._energy_range_slider.value = energy_lim

    def make_plot(self, _=None):
        with self._plot_output:
            fig, collected_data = self._create_the_plot()
            links = f"""Export in: {make_image_link(fig)}, {make_image_link(fig, text="PDF", data_format="pdf")}, {make_txt_link(collected_data)}."""
            display(ipw.HTML(links))

    def clear_plot(self, _=None):
        with self._plot_output:
            clear_output()

    def _create_the_plot(self):
        delta_e = np.min([self._fwhm_slider.value / 10, 0.005])
        elim = self._energy_range_slider.value
        energy_arr = np.arange(elim[0], elim[1], delta_e)

        # Collect data into an array (w headers) as well.
        collect_data = np.reshape(energy_arr, (1, energy_arr.size))
        collect_data_headers = ["energy [eV]"]

        # Make the figure.
        fig = plt.figure(figsize=(12, 6))

        # Pdos part.
        ax1 = plt.gca()

        ylim = [None, None]

        self._plot_projections(
            ax1, ylim, energy_arr, collect_data, collect_data_headers
        )
        self._plot_overlaps(ax1, ylim, energy_arr, collect_data, collect_data_headers)

        plt.legend(
            ncol=self._overlap.data["nspin_g2"],
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )
        plt.xlim([np.min(energy_arr), np.max(energy_arr)])
        if self._overlap.data["nspin_g2"] == 1:
            ylim[0] = 0.0
        plt.ylim(ylim)
        plt.axhline(0.0, color="k", lw=2.0, zorder=200)
        plt.ylabel("Density of States [a.u.]")
        plt.xlabel("$E-E_F$ [eV]")
        plt.show()

        return fig, (collect_data_headers, collect_data.T)

    def _plot_projections(
        self, ax1, ylim, energy_arr, collect_data, collect_data_headers
    ):
        for line_serie in self._projections.items:
            (
                label,
                picked_color,
                norm_factor,
                spin,
                data,
            ) = line_serie.return_all_values()
            if not math.isclose(norm_factor, 1.0):
                label = rf"${norm_factor:.2f}\cdot$ {label}"

            series = pdos_postprocess.create_series_w_broadening(
                data[:, 0],
                data[:, 1],
                energy_arr,
                self._fwhm_slider.value,
            )
            series *= norm_factor

            kwargs = {}
            if spin == 0:
                kwargs["label"] = label
            if "molecule" in label.lower():
                kwargs["zorder"] = 300
                if spin == 0:
                    ylim[1] = 1.2 * np.max(series)
                else:
                    ylim[0] = 1.2 * np.min(-series)

            ax1.plot(energy_arr, series * (-2 * spin + 1), picked_color, **kwargs)
            ax1.fill_between(
                energy_arr,
                0.0,
                series * (-2 * spin + 1),
                facecolor=picked_color,
                alpha=0.2,
            )

            collect_data_headers.append(f"{label} s{spin}")
            collect_data = np.vstack([collect_data, series])

    def _plot_overlaps(self, ax1, ylim, energy_arr, collect_data, collect_data_headers):
        cumulative_plot = [None, None]
        for i_serie, line_serie in enumerate(self._overlap.items):
            (
                label,
                picked_color,
                norm_factor,
                spin,
                data,
            ) = line_serie.return_all_values()

            if not math.isclose(norm_factor, 1.0):
                label = rf"${norm_factor:.1f}\cdot$ {label}"

            series = pdos_postprocess.create_series_w_broadening(
                self._overlap.data["energies_g1"][spin],
                data,
                energy_arr,
                self._fwhm_slider.value,
            )

            series *= norm_factor

            if self.cumulative_plot.value and cumulative_plot[spin] is not None:
                cumulative_plot[spin] += series
            else:
                cumulative_plot[spin] = series

            ax1.fill_between(
                energy_arr,
                0.0,
                cumulative_plot[spin] * (-2 * spin + 1),
                facecolor=picked_color,
                alpha=1.0,
                zorder=-i_serie + 100,
                label=label,
            )

            collect_data_headers.append(f"{label} s{spin}")
            collect_data = np.vstack([collect_data, series])
