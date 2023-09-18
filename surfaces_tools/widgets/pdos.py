import base64
import io
import math

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
from aiida import orm
from IPython.display import clear_output, display

from ..utils import pdos_postprocess, spm
from . import stack


def generate_custom_color_palette(n_colors):
    # Define the color points in RGB format
    colors = [
        (0, 0, 255),  # Blue (#0000ff)
        (0, 255, 255),  # Cyan (#00ffff)
        (0, 255, 0),  # Green (#00ff00)
        (255, 255, 0),  # Yellow (#ffff00)
        (255, 0, 0),  # Red (#ff0000)
    ]

    # Initialize an empty array to store RGB colors
    color_palette = np.zeros((n_colors, 3), dtype=np.uint8)

    # Determine how many points are needed between each color transition
    points_per_segment = n_colors // (len(colors) - 1)

    for i in range(len(colors) - 1):
        start_color = colors[i]
        end_color = colors[i + 1]

        for j in range(points_per_segment):
            # Interpolate the color components (R, G, B)
            r = int(
                np.interp(j, [0, points_per_segment], [start_color[0], end_color[0]])
            )
            g = int(
                np.interp(j, [0, points_per_segment], [start_color[1], end_color[1]])
            )
            b = int(
                np.interp(j, [0, points_per_segment], [start_color[2], end_color[2]])
            )

            # Store the RGB color in the array
            index = i * points_per_segment + j
            color_palette[index] = (r, g, b)

    return color_palette[::-1]


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
    html += ' target="_blank">Export txt</a>'

    return html


class PdosSelectionWidget(stack.HorizontalItemWidget):
    def __init__(self, data, selected_index=0, spin=0, color="black", factor=1.0):
        n_spins = len(list(data.values())[0])
        self._data = data
        labels = list(data.keys())
        self._series_selection = ipw.Dropdown(
            options=labels,
            label=labels[selected_index],
            description="Series:",
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="200px"),
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

        self._spin_selector = ipw.ToggleButtons(
            options=[("up", 0), ("down", 1)],
            value=spin,
            description="Spin:",
            disabled=n_spins == 1,
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
            self._data[self._series_selection.value][self._spin_selector.value],
        )


class PdosStackWidget(stack.VerticalStackWidget):
    def add_item(self, _):
        self.items += (
            PdosSelectionWidget(
                data=self.series,
                selected_index=0,
                color="black",
                factor=1.0,
            ),
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
        self._projections = PdosStackWidget(
            item_class=PdosSelectionWidget, add_button_text="Add PDOS"
        )

        self._overlap = PdosStackWidget(
            item_class=PdosSelectionWidget, add_button_text="Add Overlap"
        )
        plot_button = ipw.Button(description="Plot")
        plot_button.on_click(self.make_plot)
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
                ipw.HBox([plot_button, clear_button]),
            ]
        )

    def load_data(self, reference=None):
        workchain = orm.load_node(pk=reference)
        overlap_calculation = spm.get_calc_by_label(workchain, "overlap")
        labels_are_present = (
            "molecule" in workchain.inputs.pdos_lists[0]
            and len(workchain.inputs.pdos_lists) > 1
        )

        self._geometry_info.value = spm.get_slab_calc_info(workchain.inputs.structure)
        dos_data = pdos_postprocess.process_pdos_files(workchain)
        dos_options = {
            "Total DOS": dos_data[
                "tdos"
            ],  # Contains both spin channels [spin1, spin2] if spin-polarized, otherwise [spin1]
            "molecule PDOS": dos_data["mol"],
            **{
                f"kind {k.split('_')[-1]}": dos_data[k]
                for k in dos_data
                if k.startswith("kind_")
            },
        }

        if not labels_are_present:
            dos_options.update(
                {
                    f"selection {k.split('_')[-1]}": dos_data[k]
                    for k in dos_data
                    if k.startswith("sel")
                }
            )
        else:
            labels = [sel[1] for sel in workchain.inputs.pdos_lists[1:]]
            dos_options.update(
                {
                    labels[int(k[4:]) - 2]: dos_data[k]
                    for k in dos_data
                    if k.startswith("sel")
                }
            )

        self._projections.series = dos_options

        self._projections.items = [
            PdosSelectionWidget(
                data=dos_options,
                selected_index=0,
                color="lightgray",
                factor=0.02,
            ),
            PdosSelectionWidget(
                data=dos_options,
                selected_index=1,
                color="black",
            ),
        ]

        with overlap_calculation.outputs.retrieved.open(
            "overlap.npz", mode="rb"
        ) as fhandle:
            self._overlap_data = pdos_postprocess.load_overlap_npz(fhandle.name)

        orbital_labels = pdos_postprocess.get_full_orbital_labels(self._overlap_data)

        overlaps = []

        # Transform (spin, point, orbital) to (orbital, spin, point) and create a dictionary.
        data_array = np.transpose(
            np.array(self._overlap_data["overlap_matrix"]), (2, 0, 1)
        )
        n_orbitals = data_array.shape[0]
        data_dict = {
            label: data_array[i_orb] for i_orb, label in enumerate(orbital_labels[0])
        }
        self._overlap.series = data_dict

        # Generate a color palette and assign it to the initial set of orbitals.
        color_palette = generate_custom_color_palette(n_orbitals)

        for orbital in range(n_orbitals):
            r, g, b = color_palette[orbital]
            overlaps.append(
                PdosSelectionWidget(
                    data=data_dict,
                    selected_index=orbital,
                    color=f"#{r:02x}{g:02x}{b:02x}",
                )
            )

        self._overlap.items = overlaps

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
            links = f"""Export in: {make_image_link(fig)}, {make_image_link(fig, text="PDF", data_format="pdf")}, {make_txt_link(collected_data)}"""
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

            if spin == 0 and self._overlap_data["nspin_g2"] == 2:
                # Add empty legend entries to align the spin channels.
                for _i in range(self._overlap.length()):
                    ax1.fill_between([0.0], 0.0, [0.0], color="w", alpha=0, label=" ")
