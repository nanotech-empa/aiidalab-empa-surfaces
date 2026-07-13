import copy
import io
import os
import zipfile

import ipywidgets as ipw
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.transforms import Affine2D

from ..utils import igor

BOHR_TO_ANG = 0.529177


colormaps = ["seismic", "gist_heat"]


def remove_from_tuple(tup, index):
    tmp_list = list(tup)
    del tmp_list[index]
    return tuple(tmp_list)


def make_plot(
    fig,
    ax,
    data,
    extent,
    grid_geometry=None,
    title=None,
    title_size=None,
    center0=False,
    vmin=None,
    vmax=None,
    cmap="gist_heat",
    noadd=False,
):
    imshow_args = {
        "origin": "lower",
        "cmap": cmap,
        "interpolation": "bicubic",
        "vmin": vmin,
        "vmax": vmax,
    }
    if center0:
        data_amax = np.max(np.abs(data))
        imshow_args["vmin"] = -data_amax
        imshow_args["vmax"] = data_amax

    if grid_geometry is not None:
        transform = Affine2D.from_values(*grid_geometry["transform"])
        im = ax.imshow(
            data.T,
            extent=(0, data.shape[0], 0, data.shape[1]),
            transform=transform + ax.transData,
            **imshow_args,
        )
        corners = grid_geometry["corners"]
        ax.set_xlim(grid_geometry["xlim"])
        ax.set_ylim(grid_geometry["ylim"])
        ax.plot(
            corners[[0, 1, 2, 3, 0], 0],
            corners[[0, 1, 2, 3, 0], 1],
            color="black",
            linewidth=0.8,
        )
    else:
        im = ax.imshow(
            data.T,
            extent=extent,
            **imshow_args,
        )

    if noadd:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(r"x ($\AA$)")
        ax.set_ylabel(r"y ($\AA$)")
        cb = fig.colorbar(im, ax=ax)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()
    ax.set_title(title, loc="left")
    if title_size:
        ax.title.set_fontsize(title_size)
    if grid_geometry is not None:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(grid_geometry["xlim"])
        ax.set_ylim(grid_geometry["ylim"])
    else:
        ax.axis("scaled")


def _grid_geometry(general_info, data_shape):
    x_arr = general_info["x_arr"] * BOHR_TO_ANG
    y_arr = general_info["y_arr"] * BOHR_TO_ANG

    extent = [np.min(x_arr), np.max(x_arr), np.min(y_arr), np.max(y_arr)]
    figure_xy_ratio = (extent[1] - extent[0]) / (extent[3] - extent[2])

    dv_vectors = general_info.get("dv_vectors")
    if dv_vectors is None:
        return extent, figure_xy_ratio, None

    dv_vectors = np.asarray(dv_vectors, dtype=float) * BOHR_TO_ANG
    origin = np.asarray(general_info.get("origin", [0.0, 0.0, 0.0]), dtype=float)
    origin = origin * BOHR_TO_ANG

    a_vec = dv_vectors[0, :2]
    b_vec = dv_vectors[1, :2]
    nx, ny = data_shape
    corners = np.array(
        [
            origin[:2],
            origin[:2] + nx * a_vec,
            origin[:2] + nx * a_vec + ny * b_vec,
            origin[:2] + ny * b_vec,
        ]
    )

    is_skewed = abs(a_vec[1]) > 1.0e-10 or abs(b_vec[0]) > 1.0e-10
    if not is_skewed:
        return extent, figure_xy_ratio, None

    x_span = corners[:, 0].max() - corners[:, 0].min()
    y_span = corners[:, 1].max() - corners[:, 1].min()
    pad_x = max(0.02 * x_span, 1.0e-8)
    pad_y = max(0.02 * y_span, 1.0e-8)

    geometry = {
        "transform": (
            a_vec[0],
            a_vec[1],
            b_vec[0],
            b_vec[1],
            origin[0],
            origin[1],
        ),
        "corners": corners,
        "xlim": (corners[:, 0].min() - pad_x, corners[:, 0].max() + pad_x),
        "ylim": (corners[:, 1].min() - pad_y, corners[:, 1].max() + pad_y),
    }

    figure_xy_ratio = (geometry["xlim"][1] - geometry["xlim"][0]) / (
        geometry["ylim"][1] - geometry["ylim"][0]
    )

    return extent, figure_xy_ratio, geometry


def make_series_label(info, i_spin=None):
    if info["type"] == "const-height sts":
        label = f"ch-sts fwhm={info['fwhm']:.2f} h={info['height']:.1f}"
    elif info["type"] == "const-height stm":
        label = f"ch-stm fwhm={info['fwhm']:.2f} h={info['height']:.1f}"
    elif info["type"] == "const-isovalue sts":
        label = f"cc-sts fwhm={info['fwhm']:.2f} isov={info['isovalue']:.0e}"
    elif info["type"] == "const-isovalue stm":
        label = f"cc-stm fwhm={info['fwhm']:.2f} isov={info['isovalue']:.0e}"

    elif info["type"] == "const-height orbital":
        label = f"ch-orb h={info['height']:.1f}"
    elif info["type"] == "const-height orbital^2":
        label = f"ch-orb^2 h={info['height']:.1f}"
    elif info["type"] == "const-isovalue orbital^2":
        label = f"cc-orb^2 isov={info['isovalue']:.0e}"

    elif info["type"] == "const-height orbital sts":
        label = f"ch-orb^2 h={info['height']:.1f}"
    elif info["type"] == "const-isovalue orbital sts":
        label = f"cc-orb^2 isov={info['isovalue']:.0e}"
    else:
        print("No support for: " + str(info))

    if i_spin is not None:
        label += f", s{i_spin}"

    return label


def make_orb_label(index, homo_index):
    i_rel_homo = index - homo_index

    if i_rel_homo < 0:
        hl_label = f"HOMO{i_rel_homo:+d}"
    elif i_rel_homo == 0:
        hl_label = "HOMO"
    elif i_rel_homo == 1:
        hl_label = "LUMO"
    else:
        hl_label = f"LUMO{i_rel_homo - 1:+d}"

    return f"MO {str(index) + hl_label}, "


class SeriesPlotter:
    def __init__(self, select_indexes_function, zip_prepend):
        self.series = {}

        self.extent = None
        self.figure_xy_ratio = None
        self.grid_geometry = None
        self.wc_pk = None

        self.zip_prepend = zip_prepend

        self.select_indexes_function = select_indexes_function

        # Selector
        self.elem_list = []
        self.selections_vbox = ipw.VBox([])

        self.add_row_btn = ipw.Button(description="Add series row", disabled=True)
        self.add_row_btn.on_click(lambda b: self.add_selection_row())

        self.selector_widget = ipw.VBox([self.add_row_btn, self.selections_vbox])

        # Plotter.
        self.plot_btn = ipw.Button(description="Plot", disabled=True)
        self.plot_btn.on_click(self.plot_series)

        self.clear_btn = ipw.Button(description="Clear", disabled=True)
        self.clear_btn.on_click(self.full_clear)

        self.plot_output = ipw.VBox()

        self.fig_y = 4

        # Creating a zip file.
        self.zip_btn = ipw.Button(description="Image zip", disabled=True)
        self.zip_btn.on_click(self.create_zip_link)

        self.zip_progress = ipw.FloatProgress(
            value=0,
            min=0,
            max=1.0,
            description="progress:",
            bar_style="info",
            orientation="horizontal",
        )

        self.link_out = ipw.Output()

    def add_series_collection(self, general_info, series_info, series_data):
        for info, data in zip(series_info, series_data):
            spin = general_info.get("spin")
            series_label = make_series_label(info, i_spin=spin)

            self.series[series_label] = (data, info, general_info)

            if info["type"] == "const-height orbital":
                sq_info = copy.deepcopy(info)
                sq_info["type"] = "const-height orbital^2"
                sq_data = copy.deepcopy(data) ** 2

                series_label = make_series_label(sq_info, i_spin=spin)
                self.series[series_label] = (sq_data, sq_info, general_info)

        self.extent, self.figure_xy_ratio, self.grid_geometry = _grid_geometry(
            general_info, series_data.shape[-2:]
        )

    def setup_added_collections(self, wc_pk):
        self.add_selection_row()
        self.add_row_btn.disabled = False
        self.plot_btn.disabled = False
        self.clear_btn.disabled = False
        self.zip_btn.disabled = False

        self.wc_pk = wc_pk

    def add_selection_row(self):
        drop_full_series = ipw.Dropdown(
            description="series",
            options=sorted(self.series.keys(), reverse=True),
            style={"description_width": "auto"},
        )
        drop_cmap = ipw.Dropdown(
            description="colormap",
            options=colormaps,
            style={"description_width": "auto"},
        )
        sym_check = ipw.Checkbox(
            value=False,
            description="sym. zero",
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="auto"),
        )
        norm_check = ipw.Checkbox(
            value=False,
            description="normalize",
            disabled=False,
            style={"description_width": "auto"},
            layout=ipw.Layout(width="auto"),
        )
        rm_btn = ipw.Button(description="x", layout=ipw.Layout(width="30px"))
        rm_btn.on_click(lambda b: self.remove_line_row(b))

        elements = [drop_full_series, drop_cmap, sym_check, norm_check, rm_btn]
        element_widths = ["280px", "210px", "120px", "120px", "35px"]

        boxed_row = ipw.HBox(
            [
                ipw.HBox([row_el], layout=ipw.Layout(border="0.1px solid", width=row_w))
                for row_el, row_w in zip(elements, element_widths)
            ]
        )

        self.elem_list.append(elements)
        self.selections_vbox.children += (boxed_row,)

    def remove_line_row(self, b):
        rm_btn_list = [elem[4] for elem in self.elem_list]
        rm_index = rm_btn_list.index(b)
        del self.elem_list[rm_index]
        self.selections_vbox.children = remove_from_tuple(
            self.selections_vbox.children, rm_index
        )

    def plot_series(self, b):
        fig_y_in_px = 0.8 * self.fig_y * matplotlib.rcParams["figure.dpi"]

        num_series = len(self.elem_list)

        box_layout = ipw.Layout(
            overflow_x="scroll",
            border="3px solid black",
            width="100%",
            height=f"{fig_y_in_px * num_series + 70}px",
            display="inline-flex",
            flex_flow="column wrap",
            align_items="flex-start",
        )

        plot_hbox = ipw.Box(layout=box_layout)
        self.plot_output.children += (plot_hbox,)

        plot_hbox.children = ()

        index_list = self.select_indexes_function()

        for i in index_list:
            plot_out = ipw.Output()
            plot_hbox.children += (plot_out,)
            with plot_out:
                fig = plt.figure(
                    figsize=(self.fig_y * self.figure_xy_ratio, self.fig_y * num_series)
                )

                for i_ser in range(num_series):
                    series_label = self.elem_list[i_ser][0].value
                    cmap = self.elem_list[i_ser][1].value
                    sym_check = self.elem_list[i_ser][2].value
                    norm_check = self.elem_list[i_ser][3].value

                    # Retrieve the series data.
                    data, info, general_info = self.series[series_label]

                    energy = general_info["energies"][i]

                    # Build labels, title and file name.
                    orb_indexes = general_info.get("orb_indexes")
                    homo = general_info.get("homo")

                    mo_label = None
                    if orb_indexes is not None:
                        mo_label = make_orb_label(orb_indexes[i], homo)

                    title = f"{series_label}\n"
                    if mo_label is not None:
                        title += mo_label + " "
                    title += f"E={energy:.2f} eV"

                    # Is normalization enabled?
                    vmin = None
                    vmax = None
                    if norm_check:
                        vmin = np.min(data[index_list, :, :])
                        vmax = np.max(data[index_list, :, :])

                    # Make the plot.
                    ax = plt.subplot(num_series, 1, i_ser + 1)

                    make_plot(
                        fig,
                        ax,
                        data[i, :, :],
                        center0=sym_check,
                        vmin=vmin,
                        vmax=vmax,
                        extent=self.extent,
                        grid_geometry=self.grid_geometry,
                        title=title,
                        cmap=cmap,
                        noadd=True,
                    )

                plt.show()

    def create_zip_link(self, b):
        self.zip_btn.disabled = True

        filename = f"{self.zip_prepend}_pk{self.wc_pk}.zip"

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
            self.data_to_zip(zip_file)

        os.makedirs("tmp", exist_ok=True)

        with open("tmp/" + filename, "wb") as f:
            f.write(zip_buffer.getvalue())

        with self.link_out:
            display(HTML(f'<a href="tmp/{filename}" target="_blank">download zip</a>'))

    def data_to_zip(self, zip_file):
        index_list = self.select_indexes_function()

        num_series = len(self.elem_list)

        total_pics = len(index_list) * num_series

        for i in index_list:
            for i_ser in range(num_series):
                series_label = self.elem_list[i_ser][0].value
                cmap = self.elem_list[i_ser][1].value
                sym_check = self.elem_list[i_ser][2].value
                norm_check = self.elem_list[i_ser][3].value

                # Retrieve the series data.
                data, info, general_info = self.series[series_label]
                energy = general_info["energies"][i]

                # Build labels, title and file name.
                orb_indexes = general_info.get("orb_indexes")
                homo = general_info.get("homo")

                mo_label = None
                if orb_indexes is not None:
                    mo_label = make_orb_label(orb_indexes[i], homo)

                title = f"{series_label}\n"
                if mo_label is not None:
                    title += mo_label + " "
                title += f"E={energy:.2f} eV"

                plot_name = (
                    series_label.lower()
                    .replace(" ", "_")
                    .replace("=", "")
                    .replace("^", "")
                    .replace(",", "")
                )
                if mo_label is not None:
                    plot_name += "_mo%03d_e%.2f" % (orb_indexes[i], energy)
                else:
                    plot_name += "_%03d_e%.2f" % (i, energy)

                # Is normalization enabled?
                vmin = None
                vmax = None
                if norm_check:
                    vmin = np.min(data[index_list, :, :])
                    vmax = np.max(data[index_list, :, :])

                # Add the png to zip.
                fig = plt.figure(
                    figsize=(self.fig_y * self.figure_xy_ratio, self.fig_y)
                )
                ax = plt.gca()

                make_plot(
                    fig,
                    ax,
                    data[i, :, :],
                    center0=sym_check,
                    vmin=vmin,
                    vmax=vmax,
                    extent=self.extent,
                    grid_geometry=self.grid_geometry,
                    title=title,
                    cmap=cmap,
                    noadd=False,
                )

                imgdata = io.BytesIO()
                fig.savefig(imgdata, format="png", dpi=200, bbox_inches="tight")
                zip_file.writestr(plot_name + ".png", imgdata.getvalue())
                plt.close()

                # Add txt data to the zip.
                header = "xlim=({:.2f}, {:.2f}), ylim=({:.2f}, {:.2f})".format(
                    *self.extent[:4]
                )
                txtdata = io.BytesIO()
                np.savetxt(txtdata, data[i, :, :], header=header, fmt="%.3e")
                zip_file.writestr("txt/" + plot_name + ".txt", txtdata.getvalue())

                # Add IGOR format to zip.
                igorwave = igor.Wave2d(
                    data=data[i, :, :],
                    xmin=self.extent[0],
                    xmax=self.extent[1],
                    xlabel="x [Angstroms]",
                    ymin=self.extent[2],
                    ymax=self.extent[3],
                    ylabel="y [Angstroms]",
                    name=f"'{plot_name}'",
                )
                zip_file.writestr("itx/" + plot_name + ".itx", str(igorwave))
                self.zip_progress.value += 1.0 / float(total_pics - 1)

    def full_clear(self, b):
        self.plot_output.children = ()
