import io
import zipfile
from pathlib import Path

import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np
from aiida import orm
from IPython.display import clear_output

from surfaces_tools.utils import spm


def load_afm_pp_data(afm_pp_calc):
    try:
        with afm_pp_calc.outputs.retrieved.open("df.npy", mode="rb") as fhandle:
            df_data = np.load(fhandle)
        with afm_pp_calc.outputs.retrieved.open("df_vec.npy", mode="rb") as fhandle:
            df_vec_data = np.load(fhandle)
    except FileNotFoundError:
        with afm_pp_calc.outputs.retrieved.open("df.npz", mode="rb") as fhandle:
            df_data = np.load(fhandle)["data"]
            df_vec_data = np.load(fhandle)["lvec"]

    x_arr = df_vec_data[0, 0] + np.linspace(0.0, df_vec_data[1, 0], df_data.shape[2])
    y_arr = df_vec_data[0, 1] + np.linspace(0.0, df_vec_data[2, 1], df_data.shape[1])

    x_grid, y_grid = np.meshgrid(x_arr, y_arr)

    return x_grid, y_grid, df_data


def make_afm_pic(fig, ax, data, i_z, title, extent):
    im = ax.imshow(
        data[2][i_z, :, :],
        origin="lower",
        cmap="gray",
        interpolation="bicubic",
        extent=extent,
    )
    # pcm = ax.pcolormesh(data[0], data[1], data[2][i_z, :, :], cmap='gray', antialiased=True)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ax.axis("scaled")
    ax.set_xlabel(r"Tip$_x$ [$\AA$]")
    ax.set_ylabel(r"Tip$_y$ [$\AA$]")


class ViewAfmLegacy(ipw.VBox):
    """Widget to view the legacy AFM calculations."""

    def __init__(self, pk):
        self.workcalc = orm.load_node(pk=pk)
        self.afm_pp_calc = spm.get_calc_by_label(self.workcalc, "afm_pp")
        self.afm_2pp_calc = spm.get_calc_by_label(self.workcalc, "afm_2pp")
        self.afm_out = ipw.Output()
        self.download_zip_link = ipw.HTML()
        self.mk_zip_btn = ipw.Button(description="Make ZIP", disabled=True)
        self.mk_zip_btn.on_click(self.create_zip_dl_link)
        self.zip_progress = ipw.FloatProgress(
            value=0,
            min=0,
            max=1.0,
            description="Progress:",
            bar_style="info",
            orientation="horizontal",
        )

        scan_start_z = self.workcalc.inputs.afm_pp_params.dict.scanMin[2]
        ampl = self.workcalc.inputs.afm_pp_params.dict.Amplitude
        self.dz = self.workcalc.inputs.afm_pp_params.dict.scanStep[2]
        self.h0 = scan_start_z + ampl / 2.0

        self.data_pp = load_afm_pp_data(self.afm_pp_calc)
        self.data_2pp = load_afm_pp_data(self.afm_2pp_calc)

        self.extent = [
            self.data_pp[0][0, 0],
            self.data_pp[0][0, -1],
            self.data_pp[1][0, 0],
            self.data_pp[1][-1, 0],
        ]
        fig_y_size = 6.0
        self.figsize = (
            (self.extent[1] - self.extent[0])
            / (self.extent[3] - self.extent[2])
            * fig_y_size
            * 1.2,
            fig_y_size,
        )

        with self.afm_out:
            clear_output()
            for i_z in range(self.data_pp[2].shape[0]):
                fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(self.figsize[0] * 2, self.figsize[1])
                )
                make_afm_pic(
                    fig,
                    ax1,
                    self.data_pp,
                    i_z,
                    r"PP Tip$_z = %.2f\ \AA$" % (self.h0 + i_z * self.dz),
                    self.extent,
                )
                make_afm_pic(
                    fig,
                    ax2,
                    self.data_2pp,
                    i_z,
                    r"2PP Tip$_z = %.2f\ \AA$" % (self.h0 + i_z * self.dz),
                    self.extent,
                )
                plt.show()

        self.mk_zip_btn.disabled = False

        super().__init__(
            [
                ipw.HTML(spm.get_slab_calc_info(self.workcalc.inputs.structure)),
                ipw.HBox([self.mk_zip_btn, self.zip_progress]),
                self.download_zip_link,
                self.afm_out,
            ]
        )

    def create_zip_dl_link(self, _=None):
        self.mk_zip_btn.disabled = True

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i_z in range(self.data_pp[2].shape[0]):
                self.zip_progress.value = i_z / float(self.data_pp[2].shape[0] - 1)

                tipz = self.h0 + i_z * self.dz

                # Add images to the zip
                imgdata = io.BytesIO()
                fig = plt.figure(figsize=self.figsize)
                make_afm_pic(
                    fig,
                    plt.gca(),
                    self.data_pp,
                    i_z,
                    rf"PP Tip$_z = {tipz:.2f}\ \AA$",
                    self.extent,
                )
                fig.savefig(imgdata, format="png", dpi=200, bbox_inches="tight")
                zip_file.writestr(f"pp/pp_{i_z:02d}.png", imgdata.getvalue())
                plt.close()
                imgdata = io.BytesIO()
                fig = plt.figure(figsize=self.figsize)
                make_afm_pic(
                    fig,
                    plt.gca(),
                    self.data_2pp,
                    i_z,
                    rf"2PP Tip$_z = {tipz:.2f}\ \AA$",
                    self.extent,
                )
                fig.savefig(imgdata, format="png", dpi=200, bbox_inches="tight")
                zip_file.writestr(f"pp2/pp2_{i_z:02d}.png", imgdata.getvalue())
                plt.close()

                # Add raw data to the zip
                header = (
                    "tipz={:.2f}, xlim=({:.2f}, {:.2f}), ylim=({:.2f}, {:.2f})".format(
                        tipz,
                        self.extent[0],
                        self.extent[1],
                        self.extent[2],
                        self.extent[3],
                    )
                )
                txtdata = io.BytesIO()
                np.savetxt(
                    txtdata, self.data_pp[2][i_z, :, :], header=header, fmt="%.2e"
                )
                zip_file.writestr(f"pp/pp_{i_z:02d}.txt", txtdata.getvalue())
                txtdata = io.BytesIO()
                np.savetxt(
                    txtdata, self.data_2pp[2][i_z, :, :], header=header, fmt="%.2e"
                )
                zip_file.writestr(f"pp2/pp2_{i_z:02d}.txt", txtdata.getvalue())

        filename = f"afm_{self.workcalc.pk}.zip"

        with open("tmp/" + filename, "wb") as f:
            f.write(zip_buffer.getvalue())
        self.download_zip_link.value = (
            f'<a href="tmp/{filename}" target="_blank">download zip</a>'
        )


class ViewAfmWidget(ipw.VBox):
    """Widget to view the legacy AFM calculations."""

    def __init__(self, pk):
        self.workcalc = orm.load_node(pk=pk)
        self.ppafm_calc = spm.get_calc_by_label(self.workcalc, "ppafm")
        self.afm_out = ipw.Output()
        self.download_zip_link = ipw.HTML()
        self.mk_zip_btn = ipw.Button(description="Make ZIP", disabled=True)
        self.mk_zip_btn.on_click(self.create_zip_dl_link)
        self.zip_progress = ipw.FloatProgress(
            value=0,
            min=0,
            max=1.0,
            description="Progress:",
            bar_style="info",
            orientation="horizontal",
        )

        scan_start_z = self.workcalc.inputs.ppafm_params.dict.scanMin[2]
        ampl = self.workcalc.inputs.ppafm_params.dict.Amplitude
        self.dz = self.workcalc.inputs.ppafm_params.dict.scanStep[2]
        self.h0 = scan_start_z + ampl / 2.0

        self.data_ppafm = load_afm_pp_data(self.ppafm_calc)

        self.extent = [
            self.data_ppafm[0][0, 0],
            self.data_ppafm[0][0, -1],
            self.data_ppafm[1][0, 0],
            self.data_ppafm[1][-1, 0],
        ]
        fig_y_size = 6.0
        self.figsize = (
            (self.extent[1] - self.extent[0])
            / (self.extent[3] - self.extent[2])
            * fig_y_size
            * 1.2,
            fig_y_size,
        )

        super().__init__(
            [
                ipw.HTML(spm.get_slab_calc_info(self.workcalc.inputs.structure)),
                ipw.HBox([self.mk_zip_btn, self.zip_progress]),
                self.download_zip_link,
                self.afm_out,
            ]
        )

        with self.afm_out:
            clear_output()
            for i_z in range(self.data_ppafm[2].shape[0]):
                fig, ax1 = plt.subplots(1, figsize=(self.figsize[0], self.figsize[1]))
                make_afm_pic(
                    fig,
                    ax1,
                    self.data_ppafm,
                    i_z,
                    r"PP Tip$_z = %.2f\ \AA$" % (self.h0 + i_z * self.dz),
                    self.extent,
                )
                plt.show()

        self.mk_zip_btn.disabled = False

    def create_zip_dl_link(self, _=None):
        self.mk_zip_btn.disabled = True

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i_z in range(self.data_ppafm[2].shape[0]):
                self.zip_progress.value = i_z / float(self.data_ppafm[2].shape[0] - 1)

                tipz = self.h0 + i_z * self.dz

                # Add images to the zip
                imgdata = io.BytesIO()
                fig = plt.figure(figsize=self.figsize)
                make_afm_pic(
                    fig,
                    plt.gca(),
                    self.data_ppafm,
                    i_z,
                    rf"PP Tip$_z = {tipz:.2f}\ \AA$",
                    self.extent,
                )
                fig.savefig(imgdata, format="png", dpi=200, bbox_inches="tight")
                zip_file.writestr(f"ppafm_{i_z:02d}.png", imgdata.getvalue())
                plt.close()

                # Add raw data to the zip
                header = (
                    "tipz={:.2f}, xlim=({:.2f}, {:.2f}), ylim=({:.2f}, {:.2f})".format(
                        tipz,
                        self.extent[0],
                        self.extent[1],
                        self.extent[2],
                        self.extent[3],
                    )
                )
                txtdata = io.BytesIO()
                np.savetxt(
                    txtdata, self.data_ppafm[2][i_z, :, :], header=header, fmt="%.2e"
                )
                zip_file.writestr(f"ppafm_{i_z:02d}.txt", txtdata.getvalue())

        filename = f"afm_{self.workcalc.pk}.zip"

        # Create the tmp directory if it doesn't exist
        Path("tmp").mkdir(parents=True, exist_ok=True)

        with open("tmp/" + filename, "wb") as f:
            f.write(zip_buffer.getvalue())
        self.download_zip_link.value = (
            f'<a href="tmp/{filename}" target="_blank">download zip</a>'
        )
