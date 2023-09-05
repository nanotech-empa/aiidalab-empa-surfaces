import base64
import io
import math
from collections import OrderedDict

import ipywidgets as ipw
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from aiida import orm
from IPython.display import clear_output, display

from ..utils import pdos_postprocess, spm
from . import stack


def make_png_link(figure, text="PNG", data_format="png"):
    imgdata = io.BytesIO()
    figure.savefig(imgdata, format=data_format, dpi=300, bbox_inches="tight")
    imgdata.seek(0)  # rewind the data
    pngfile = base64.b64encode(imgdata.getvalue()).decode()

    filename = f"pdos.{data_format}"

    html = f'<a download="{filename}" href="'
    html += f'data:image/{data_format};name={filename};base64,{pngfile}"'
    html += ' id="pdos_png_link"'
    html += f' target="_blank">{text}</a>'

    return html


def make_pdf_link(figure, text="PDF", data_format="pdf"):
    imgdata = io.BytesIO()
    figure.savefig(imgdata, format=data_format, dpi=300, bbox_inches="tight")
    imgdata.seek(0)  # rewind the data
    pdffile = base64.b64encode(imgdata.getvalue()).decode()

    filename = f"pdos.{data_format}"

    html = f'<a download="{filename}" href="'
    html += f'data:image/{data_format};name={filename};base64,{pdffile}"'
    html += ' id="pdos_pdf_link"'
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
    html += ' target="_blank">Export txt</a>'

    return html


class PdosSelectionWidget(stack.HorizontalItemWidget):
    def __init__(self, series, i_sel=0, spin=0, color="black", factor=1.0):
        self._series_selection = ipw.Dropdown(
            options=series,
            value=series[i_sel],
            description="series:",
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="200px"),
        )
        self._color_picker = ipw.ColorPicker(
            concise=False,
            description="color",
            value=color,
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="200px"),
        )
        self._norm_factor = ipw.FloatText(
            value=factor,
            step=0.01,
            description="factor",
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="150px"),
        )

        self._spin_selector = ipw.ToggleButtons(
            options=[("up", 0), ("down", 1)],
            value=spin,
            description="spin:",
            disabled=False,
            style={"description_width": "auto", "button_width": "60px"},
            layout=ipw.Layout(width="180px"),
        )

        super().__init__(
            children=[
                self._series_selection,
                self._color_picker,
                self._spin_selector,
                self._norm_factor,
            ]
        )

    def return_all_values(self):
        return (
            self._series_selection.value,
            self._color_picker.value,
            self._norm_factor.value,
            self._spin_selector.value,
        )


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
            disabled=False,
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
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            style=style,
            layout=layout,
        )
        self._plot_output = ipw.Output()
        self._geometry_info = ipw.HTML()
        self._projections = stack.VerticalStackWidget(
            item_class=PdosSelectionWidget, add_button_text="Add PDOS"
        )

        self._overlap = stack.VerticalStackWidget(
            item_class=PdosSelectionWidget, add_button_text="Add Overlap"
        )
        plot_button = ipw.Button(description="plot")
        plot_button.on_click(self.make_plot)

        super().__init__(
            [
                self._fwhm_slider,
                self._energy_range_slider,
                self._geometry_info,
                self._plot_output,
                self._projections,
                self._overlap,
                plot_button,
            ]
        )

    def load_data(self, reference=None):
        workchain = orm.load_node(pk=reference)
        overlap_calculation = spm.get_calc_by_label(workchain, "overlap")
        labels_are_present = (
            "molecule" in workchain.inputs.pdos_lists[0]
            and len(workchain.inputs.pdos_lists) > 1
        )
        self._energy_lim = [
            float(workchain.inputs.overlap_params["--emin1"]),
            float(workchain.inputs.overlap_params["--emax1"]),
        ]
        self._geometry_info.value = spm.get_slab_calc_info(workchain.inputs.structure)
        dos_data = pdos_postprocess.process_pdos_files(workchain)
        if not labels_are_present:
            self._dos_options = OrderedDict(
                [
                    ("total DOS", dos_data["tdos"]),
                    ("molecule PDOS", dos_data["mol"]),
                    *[
                        (f"selection {k.split('_')[-1]}", dos_data[k])
                        for k in dos_data
                        if k.startswith("sel")
                    ],
                    *[
                        (f"kind {k.split('_')[-1]}", dos_data[k])
                        for k in dos_data
                        if k.startswith("kind_")
                    ],
                ]
            )
        else:
            labels = [sel[1] for sel in workchain.inputs.pdos_lists[1:]]
            self._dos_options = OrderedDict(
                [
                    ("total DOS", dos_data["tdos"]),
                    ("molecule PDOS", dos_data["mol"]),
                    *[
                        (labels[int(k[4:]) - 2], dos_data[k])
                        for k in dos_data
                        if k.startswith("sel")
                    ],
                    *[
                        (f"kind {k.split('_')[-1]}", dos_data[k])
                        for k in dos_data
                        if k.startswith("kind_")
                    ],
                ]
            )

        with overlap_calculation.outputs.retrieved.open(
            "overlap.npz", mode="rb"
        ) as fhandle:
            self._overlap_data = pdos_postprocess.load_overlap_npz(fhandle.name)

        self._orbital_labels = pdos_postprocess.get_full_orbital_labels(
            self._overlap_data
        )
        self.initialize_selections()
        self.initialize_pdos_lines()
        self.initialize_overlap_lines()

    def make_plot(self, _=None):
        with self._plot_output:
            fig, collected_data = self._create_the_plot()
            links = f"""Export in: {make_png_link(fig)}, {make_pdf_link(fig)}, {make_txt_link(collected_data)}"""
            display(ipw.HTML(links))

    def clear_plot(self, _=None):
        with self._plot_output:
            clear_output()

    def initialize_selections(self):
        self._energy_range_slider.min = self._energy_lim[0]
        self._energy_range_slider.max = self._energy_lim[1]
        self._energy_range_slider.value = self._energy_lim

    def initialize_pdos_lines(self):
        self._projections.items = [
            PdosSelectionWidget(
                series=list(self._dos_options.keys()),
                i_sel=0,
                color="lightgray",
                factor=0.02,
            ),
            PdosSelectionWidget(
                series=list(self._dos_options.keys()),
                i_sel=1,
                color="black",
            ),
        ]

    def initialize_overlap_lines(self):
        mpl_def_colors = [color["color"] for color in mpl.rcParams["axes.prop_cycle"]]
        mpl_def_blu = ["#219ebc", "#e0fbfc", "#98c1d9", "#3d5a80"]
        mpl_def_red = ["#d90429", "#ee6c4d", "#fb8500", "#ffb703"]

        # Rotate the default colors.
        for _ in range(8 - self._overlap_data["homo_i_g2"][0]):
            mpl_def_colors.append(mpl_def_colors.pop(0))

        ihomo = 0
        ilumo = 0
        overlaps = []
        for i_spin in range(self._overlap_data["nspin_g2"]):
            for i_gas, label in enumerate(self._orbital_labels[i_spin]):
                if "HOMO" in label:
                    color = mpl_def_red[ihomo % len(mpl_def_red)]
                    ihomo += 1
                else:
                    color = mpl_def_blu[ilumo % len(mpl_def_red)]
                    ilumo += 1

                overlaps.append(
                    PdosSelectionWidget(
                        series=self._orbital_labels[i_spin],
                        spin=i_spin,
                        i_sel=i_gas,
                        color=color,
                    )
                )

        self._overlap.items = overlaps

    def _create_the_plot(self):
        de = np.min([self._fwhm_slider.value / 10, 0.005])
        elim = self._energy_range_slider.value
        energy_arr = np.arange(elim[0], elim[1], de)

        # Collect data into an array (w headers) as well.
        collect_data = np.reshape(energy_arr, (1, energy_arr.size))
        collect_data_headers = ["energy [eV]"]

        # Make the figure.
        fig = plt.figure(figsize=(12, 6))

        # Pdos part.
        ax1 = plt.gca()

        ylim = [None, None]

        for line_serie in self._projections.items:
            label, picked_color, norm_factor, spin = line_serie.return_all_values()
            data = self._dos_options[label]
            if not math.isclose(norm_factor, 1.0):
                label = rf"${norm_factor:.2f}\cdot$ {label}"

            for i_spin in range(len(data)):
                series = pdos_postprocess.create_series_w_broadening(
                    data[i_spin][:, 0],
                    data[i_spin][:, 1],
                    energy_arr,
                    self._fwhm_slider.value,
                )
                series *= norm_factor

                kwargs = {}
                if i_spin == 0:
                    kwargs["label"] = label
                if "molecule" in label.lower():
                    kwargs["zorder"] = 300
                    if i_spin == 0:
                        ylim[1] = 1.2 * np.max(series)
                    else:
                        ylim[0] = 1.2 * np.min(-series)

                ax1.plot(energy_arr, series * (-2 * i_spin + 1), picked_color, **kwargs)
                ax1.fill_between(
                    energy_arr,
                    0.0,
                    series * (-2 * i_spin + 1),
                    facecolor=picked_color,
                    alpha=0.2,
                )

                collect_data_headers.append(f"{label} s{i_spin}")
                collect_data = np.vstack([collect_data, series])

        # Overlap part.
        for i_serie, line_serie in enumerate(self._overlap.items):
            label, picked_color, norm_factor, spin = line_serie.return_all_values()
            i_orb = self._orbital_labels[spin].index(label)
            data = self._overlap_data["overlap_matrix"][spin][:, i_orb]

            if not math.isclose(norm_factor, 1.0):
                label = rf"${norm_factor:.1f}\cdot$ {label}"

            series = pdos_postprocess.create_series_w_broadening(
                self._overlap_data["energies_g1"][spin],
                data,
                energy_arr,
                self._fwhm_slider.value,
            )
            series *= norm_factor

            ax1.fill_between(
                energy_arr,
                0.0,
                series * (-2 * spin + 1),
                facecolor=picked_color,
                alpha=1.0,
                zorder=-i_serie + 100,
                label=label,
            )

            collect_data_headers.append(f"{label} s{spin}")
            collect_data = np.vstack([collect_data, series])

            if i_spin == 0 and self._overlap_data["nspin_g2"] == 2:
                # Add empty legend entries to align the spin channels.
                for _i in range(self._overlap.length()):
                    ax1.fill_between([0.0], 0.0, [0.0], color="w", alpha=0, label=" ")

        plt.legend(
            ncol=self._overlap_data["nspin_g2"],
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
        )

        plt.xlim([np.min(energy_arr), np.max(energy_arr)])

        if self._overlap_data["nspin_g2"] == 1:
            ylim[0] = 0.0
        plt.ylim(ylim)

        plt.axhline(0.0, color="k", lw=2.0, zorder=200)

        plt.ylabel("Density of States [a.u.]")
        plt.xlabel("$E-E_F$ [eV]")

        plt.show()

        return fig, (collect_data_headers, collect_data.T)
