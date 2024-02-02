import base64
import copy
import io
import math
import re

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
import traitlets as tl
from aiida import common, orm
from IPython.display import clear_output, display

from ..helpers import HART_2_EV
from ..utils import spm
from . import stack


def read_and_process_pdos_file(pdos_path):
    try:
        header = open(pdos_path).readline()
    except TypeError:
        header = pdos_path.readline()
    try:  # noqa: TC101
        kind = re.search(r"atomic kind.(\S+)", header).group(1)
    except Exception:
        kind = None
    data = np.loadtxt(pdos_path)

    # Determine fermi by counting the number of electrons and
    # taking the middle of HOMO and LUMO.
    n_el = int(np.round(np.sum(data[:, 2])))
    if data[0, 2] > 1.5:
        n_el = int(np.round(n_el / 2))
    fermi = 0.5 * (data[n_el - 1, 1] + data[n_el, 1])

    out_data = np.zeros((data.shape[0], 2))
    out_data[:, 0] = (data[:, 1] - fermi) * HART_2_EV  # energy
    out_data[:, 1] = np.sum(data[:, 3:], axis=1)  # "contracted pdos"
    return out_data, kind


def process_pdos_files(pdos_workchain, newversion=True):
    dos = {}

    if newversion:
        retr_files = pdos_workchain.outputs.slab_retrieved.list_object_names()
        retr_folder = pdos_workchain.outputs.slab_retrieved
    else:
        for process in pdos_workchain.called_descendants:
            if process.label == "slab_scf":
                slab_scf = process
        retr_files = slab_scf.outputs.retrieved.list_object_names()
        retr_folder = slab_scf.outputs.retrieved

    # Make sets that contain filenames with PDOS
    all_pdos = {f for f in retr_files if f.endswith(".pdos")}
    all_user_defined_pdos = {f for f in all_pdos if "list" in f}
    element_pdos = all_pdos - all_user_defined_pdos

    def _extract_pdos_num(file):
        return int(re.search("list(.*)-", file).group(1))

    # Sort the sets
    all_user_defined_pdos = sorted(all_user_defined_pdos, key=_extract_pdos_num)

    # Identify the number of spin channels.
    nspin = 2 if any("BETA" in f for f in all_pdos) else 1

    def _read_pdos(file):
        with retr_folder.open(file) as fhandle:
            try:
                path = fhandle.name
            except AttributeError:
                path = fhandle
            pdos, kind = read_and_process_pdos_file(path)
        return pdos, kind

    # Element-wise PDOS.
    for file in sorted(element_pdos):
        i_spin = 1 if "BETA" in file else 0
        pdos, kind = _read_pdos(file)

        # Remove any digits from kind.
        kind = "".join([c for c in kind if not c.isdigit()])
        label = f"kind_{kind}"
        if label not in dos:
            dos[label] = [None] * nspin
        if dos[label][i_spin] is not None:
            dos[label][i_spin][:, 1] += pdos[:, 1]
        else:
            dos[label][i_spin] = pdos

    # User-defined PDOS.
    for file in all_user_defined_pdos:
        i_spin = 1 if "BETA" in file else 0
        pdos, kind = _read_pdos(file)
        num = _extract_pdos_num(file)
        label = f"sel_{num}"
        if label not in dos:
            dos[label] = [None] * nspin
        dos[label][i_spin] = pdos

    tdos = None
    for k in dos:
        if k.startswith("kind"):
            if tdos is None:
                tdos = copy.deepcopy(dos[k])
            else:
                for i_spin in range(nspin):
                    tdos[i_spin][:, 1] += dos[k][i_spin][:, 1]
    dos["tdos"] = tdos
    return dos


def create_series_w_broadening(x_values, y_values, x_arr, fwhm, shape="g"):
    spectrum = np.zeros(len(x_arr))

    def lorentzian(x_):
        # factor = np.pi*fwhm/2 # to make maximum 1.0
        return 0.5 * fwhm / (np.pi * (x_**2 + (0.5 * fwhm) ** 2))

    def gaussian(x_):
        sigma = fwhm / 2.3548
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x_**2) / (2 * sigma**2))

    for xv, yv in zip(x_values, y_values):
        if shape == "g":
            spectrum += yv * gaussian(x_arr - xv)
        else:
            spectrum += yv * lorentzian(x_arr - xv)
    return spectrum


def match_and_reduce_spin_channels(om):
    # In principle, for high-spin states such as triplet, we could assume
    # that alpha is always the higher-populated spin.
    # However, for uks-1, we have to check both configurations and pick higher overlap,
    # so might as well do it in all cases

    # In rks case, just remove the 2nd spin index
    if len(om[0]) == 1:
        return [om[0][0]]

    # Same spin channels.
    same_contrib = 0
    for i_spin in range(2):
        same_contrib = np.sum(om[i_spin][i_spin])

    # Opposite spin channels
    oppo_contrib = 0
    for i_spin in range(2):
        oppo_contrib = np.sum(om[i_spin][(i_spin + 1) % 2])

    if same_contrib >= oppo_contrib:
        return [om[i][i] for i in range(2)]
    else:
        return [om[i][(i + 1) % 2] for i in range(2)]


def load_overlap_npz_legacy(loaded_data):
    overlap_data = {
        "nspin_g1": 1,
        "nspin_g2": 1,
        "homo_i_g2": [int(loaded_data["homo_grp2"])],
        "overlap_matrix": [loaded_data["overlap_matrix"]],
        "energies_g1": [loaded_data["en_grp1"]],
        "energies_g2": [loaded_data["en_grp2"]],
    }
    return overlap_data


def load_overlap_npz(npz_path):
    loaded_data = np.load(npz_path, allow_pickle=True)

    if "metadata" not in loaded_data:
        return load_overlap_npz_legacy(loaded_data)

    metadata = loaded_data["metadata"][0]

    overlap_matrix = []

    for i_spin_g1 in range(metadata["nspin_g1"]):
        overlap_matrix.append([])
        for i_spin_g2 in range(metadata["nspin_g2"]):
            overlap_matrix[-1].append(
                loaded_data[f"overlap_matrix_s{i_spin_g1}s{i_spin_g2}"]
            )

    energies_g1 = []
    for i_spin_g1 in range(metadata["nspin_g1"]):
        energies_g1.append(loaded_data[f"energies_g1_s{i_spin_g1}"])

    energies_g2 = []
    orb_indexes_g2 = []
    for i_spin_g2 in range(metadata["nspin_g2"]):
        energies_g2.append(loaded_data[f"energies_g2_s{i_spin_g2}"])
        orb_indexes_g2.append(loaded_data[f"orb_indexes_g2_s{i_spin_g2}"])

    overlap_data = {
        "nspin_g1": metadata["nspin_g1"],
        "nspin_g2": metadata["nspin_g2"],
        "homo_i_g2": metadata["homo_i_g2"],
        "overlap_matrix": overlap_matrix,
        "energies_g1": energies_g1,
        "energies_g2": energies_g2,
        "orb_indexes_g2": orb_indexes_g2,
    }

    overlap_data["overlap_matrix"] = match_and_reduce_spin_channels(
        overlap_data["overlap_matrix"]
    )

    return overlap_data


def get_orbital_label(i_orb_wrt_homo):
    if i_orb_wrt_homo < 0:
        label = "HOMO%+d" % i_orb_wrt_homo
    elif i_orb_wrt_homo == 0:
        label = "HOMO"
    elif i_orb_wrt_homo == 1:
        label = "LUMO"
    elif i_orb_wrt_homo > 1:
        label = "LUMO%+d" % (i_orb_wrt_homo - 1)
    return label


def get_full_orbital_label(i_spin, i_orb, overlap_data):
    energy = overlap_data["energies_g2"][i_spin][i_orb]

    spin_letter = ""
    if overlap_data["nspin_g2"] == 2:
        spin_letter = "a-" if i_spin == 0 else "b-"

    i_wrt_homo = i_orb - overlap_data["homo_i_g2"][i_spin]
    label = get_orbital_label(i_wrt_homo)

    full_label = f"{spin_letter}{label:6} (E={energy:5.2f})"

    if "orb_indexes_g2" in overlap_data:
        index = overlap_data["orb_indexes_g2"][i_spin][i_orb]
        full_label = f"MO{index:2} {full_label}"
    return full_label


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
        try:
            data = process_pdos_files(workchain)
        except (KeyError, common.NotExistentAttributeError):
            data = process_pdos_files(workchain, newversion=False)

        self.options = {
            "Total DOS": "tdos",
            **{
                f"kind {name.split('_')[-1]}": name
                for name in data
                if name.startswith("kind_")
            },
        }
        labels = [sel[1] if isinstance(sel, tuple) else f"Selection {sel}" for sel in workchain.inputs.pdos_lists]
        self.options.update(
            {
                labels[int(name[4:]) - 1]: name
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
                self.items += (total_pdos,)


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
            n_orbitals = len(self.data["energies_g2"][spin])
            options.append(
                {
                    get_full_orbital_label(spin, i, self.data): i
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
                self._geometry_info,
                self._plot_output,
                self._fwhm_slider,
                self._energy_range_slider,
                ipw.HBox([plot_button, self.cumulative_plot, clear_button]),
                self._projections,
                self._overlap,
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
            self._overlap.data = load_overlap_npz(fhandle.name)

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

            series = create_series_w_broadening(
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

            series = create_series_w_broadening(
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
