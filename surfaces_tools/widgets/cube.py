# SPDX-License-Identifier: MIT
# flake8: noqa
# NOTE: this file intentionally contains widgets only in English.

import copy
import logging
import re
import tempfile

import ase.io.cube
import ipywidgets as ipw
import matplotlib.pyplot as plt
import nglview
import numpy as np
import toml
import traitlets as tl
from aiida import engine, orm, plugins
from aiida_shell import ShellJob
from cubehandler import Cube

logger = logging.getLogger(__name__)

Cp2kOrbitalsWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.orbitals")
Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")
Cp2kFragmentSeparationWorkChain = plugins.WorkflowFactory(
    "nanotech_empa.cp2k.fragment_separation"
)


# ------------------------- Isosurface controls -----------------------------


class OneIsovalue(ipw.HBox):
    def __init__(self, structure=None):
        self.isovalue_widget = ipw.BoundedFloatText(
            value=1e-3, min=1e-5, max=1e-1, step=1e-5, description="Isovalue"
        )
        self.color_widget = ipw.ColorPicker(
            concise=False, description="Pick a color", value="cyan", disabled=False
        )
        super().__init__([self.isovalue_widget, self.color_widget])


class IsovaluesWidget(ipw.VBox):
    iso_min = tl.Float(1.0e-5, allow_none=True)
    iso_max = tl.Float(1.0e-1, allow_none=True)

    def __init__(self):
        self.isovalues = ipw.VBox()
        self.info = ipw.HTML(f"min: {self.iso_min:.1e}, max: {self.iso_max:.1e}")

        self.add_isovalue_button = ipw.Button(
            description="Add isovalue",
            layout={"width": "initial"},
            button_style="success",
        )
        self.add_isovalue_button.on_click(self.add_isovalue)

        self.remove_isovalue_button = ipw.Button(
            description="Remove isovalue",
            layout={"width": "initial"},
            button_style="danger",
        )
        self.remove_isovalue_button.on_click(self.remove_isovalue)

        super().__init__(
            [
                self.isovalues,
                ipw.HBox(
                    [self.info, self.add_isovalue_button, self.remove_isovalue_button]
                ),
            ]
        )

    def return_values(self):
        return zip(
            *[
                (child.children[0].value, child.children[1].value)
                for child in self.isovalues.children
            ]
        )

    def set_range(self, vmin=-0.001, vmax=0.001):
        default = min(0.001, vmax)
        self.info.value = f"min: {vmin:.1e}, max: {vmax:.1e}"
        self.iso_min = vmin
        self.iso_max = vmax
        if abs(vmin) > abs(vmax):
            default = max(-0.001, vmin)
        for isovalue in self.isovalues.children:
            isovalue.children[0].min = vmin
            isovalue.children[0].max = vmax
            isovalue.children[0].value = default

    @tl.observe("iso_min", "iso_max")
    def _observe_range(self, _=None):
        self.set_range(vmin=self.iso_min, vmax=self.iso_max)

    def add_isovalue(self, _=None):
        new = OneIsovalue()
        new.min = self.iso_min
        new.max = self.iso_max
        new.value = (
            min(0.001, self.iso_max)
            if self.iso_max >= 1.0e-8
            else max(-0.001, self.iso_min)
        )
        self.isovalues.children += (new,)

    def remove_isovalue(self, _=None):
        self.isovalues.children = self.isovalues.children[:-1]

    def traits_to_link(self):
        return ["cube"]


# ----------------------------- 3D viewer ----------------------------------


class CubeArrayData3dViewerWidget(ipw.VBox):
    """View structure + cube; add isosurfaces from the cube."""

    cube = tl.Instance(Cube, allow_none=True)

    def __init__(self, **kwargs):
        self.structure = None
        self.viewer = nglview.NGLWidget()
        self.isovalues = IsovaluesWidget()
        self.show_isosurfaes_button = ipw.Button(
            description="Show isosurfaces",
            layout={"width": "initial"},
            button_style="success",
        )
        self.show_isosurfaes_button.on_click(lambda b: self.update_plot())

        # components we manage
        self._structure_component = None
        self._cube_component = None

        super().__init__(
            [self.viewer, self.isovalues, self.show_isosurfaes_button], **kwargs
        )

    @tl.observe("cube")
    def on_observe_cube(self, _=None):
        if self.cube is None:
            return
        self.isovalues.set_range(
            vmin=float(np.min(self.cube.data)), vmax=float(np.max(self.cube.data))
        )
        self.structure = self.cube.ase_atoms
        self.update_plot()

    def update_plot(self):
        """Refresh structure + volumetric cube components."""
        if self._structure_component is not None:
            try:
                self.viewer.remove_component(self._structure_component.id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("remove_component(structure) failed: %s", exc)
            self._structure_component = None

        if self._cube_component is not None:
            try:
                self.viewer.remove_component(self._cube_component.id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("remove_component(cube) failed: %s", exc)
            self._cube_component = None

        self.setup_cube_plot()

        try:
            isovalues, colors = self.isovalues.return_values()
            self.set_cube_isosurf(isovalues, colors)
        except ValueError:
            # no isovalues added yet
            pass

    def setup_cube_plot(self):
        if self.structure is None or self.cube is None:
            return
        self._structure_component = self.viewer.add_component(
            nglview.ASEStructure(self.structure)
        )
        with tempfile.NamedTemporaryFile(mode="w") as tempf:
            ase.io.cube.write_cube(tempf, self.structure, self.cube.data)
            self._cube_component = self.viewer.add_component(tempf.name, ext="cube")
            self._cube_component.clear()

    def set_cube_isosurf(self, isovals, colors):
        if self._cube_component is None:
            return
        self._cube_component.clear()
        for isov, col in zip(isovals, colors):
            self._cube_component.add_surface(
                color=col, isolevelType="value", isolevel=isov
            )


# ----------------------- render helper (AiiDA) ----------------------------


@engine.calcfunction
def render_cube_file(settings, remote_folder):
    fd = orm.FolderData()
    for name, value in settings.items():
        full_name = f"{name}.{value['format']}"
        with open(full_name, "rb") as file_h:
            fd.put_object_from_bytes(file_h.read(), path=full_name)
    return {"rendered_images": fd}


# ------------------------ Main handle widget ------------------------------


class HandleCubeFiles(ipw.VBox):

    node_pk = tl.Int(None, allow_none=True)
    nel_up = tl.Int(None, allow_none=True)
    nel_dw = tl.Int(None, allow_none=True)
    uks = tl.Bool(False)
    render_instructions = tl.Dict({}, allow_none=True)
    remote_data_uuid = tl.Unicode(allow_none=True)

    def __init__(self):
        self.node = None

        self._viewer = CubeArrayData3dViewerWidget(layout=ipw.Layout(width="550px"))
        self.slice2d = CubePlaneCut2D()

        # Keep the cubes in sync (3D viewer ↔ 2D slicer)
        self._cube_sync = tl.dlink((self._viewer, "cube"), (self.slice2d, "cube"))

        self.cube_selector = ipw.Select(
            options=[],
            description="Cube files:",
            layout=ipw.Layout(width="400px"),
            style={"description_width": "initial"},
        )
        self.cube_selector.observe(self.show_selected_cube, names="value")

        self.camera_orientation = ipw.Textarea(
            description="Camera orientation",
            placeholder="0, 0, 0, 0\n0, 0, 0, 0\n0, 0, 0, 0\n0, 0, 0, 0",
            layout=ipw.Layout(width="400px", height="100px"),
            style={"description_width": "initial"},
        )
        tl.dlink(
            (self._viewer.viewer, "_camera_orientation"),
            (self.camera_orientation, "value"),
            transform=lambda x: " ".join(
                [
                    f"{float(t):4.4}" if (i + 1) % 4 else f"{float(t):4.4}\n"
                    for i, t in enumerate(x)
                ]
            ),
        )

        self.render_this_view = ipw.Button(
            description="Render this view", layout=ipw.Layout(width="120px")
        )
        self.render_this_view.on_click(self.append_render_instructions)

        self.render_name = ipw.Text(
            description="Render name:",
            layout=ipw.Layout(width="276px"),
            style={"description_width": "initial"},
        )
        self.error_message = ipw.HTML()

        self.render_instructions_widget = ipw.Textarea(
            description="Render instructions:",
            style={"description_width": "initial"},
            layout=ipw.Layout(width="380px", height="300px"),
        )
        accordion = ipw.Accordion(
            children=[self.render_instructions_widget], layout=ipw.Layout(width="400px")
        )
        accordion.set_title(0, "Render instructions")
        accordion.selected_index = None
        self.render_instructions_widget.disabled = True

        self.render_all_button = ipw.Button(description="Render all")
        self.render_all_button.on_click(self.render_all)

        super().__init__(
            [
                ipw.HBox(
                    [
                        ipw.VBox([self._viewer, self.slice2d]),
                        ipw.VBox(
                            [
                                self.cube_selector,
                                self.camera_orientation,
                                ipw.HBox(
                                    [
                                        self.render_this_view,
                                        self.render_name,
                                        self.error_message,
                                    ]
                                ),
                                accordion,
                                self.render_all_button,
                            ]
                        ),
                    ]
                ),
            ]
        )

    def show_selected_cube(self, _=None):
        if not self.cube_selector.value:
            return
        cube_obj = Cube.from_content(
            self.node.get_object_content(self.cube_selector.value)
        )
        self._viewer.cube = cube_obj  # slice2d is linked via dlink

    def get_calcs(self):
        query = orm.QueryBuilder()
        query.append(
            (
                Cp2kOrbitalsWorkChain,
                Cp2kGeoOptWorkChain,
                Cp2kFragmentSeparationWorkChain,
            ),
            tag="orbitals_wc",
            project="description",
        )
        query.append(
            ShellJob,
            filters={
                "label": {"in": ["cube-shrink", "charge-lowres", "ChargeDiff-lowres"]}
            },
            project="uuid",
            with_incoming="orbitals_wc",
        )
        return query.all()

    def select_calculation(self, _=None):
        if not self.node_pk:
            return
        calc = orm.load_node(self.node_pk)
        self.node = calc.outputs.out_cubes
        try:
            self.remote_data_uuid = calc.inputs.nodes.remote_previous_job.uuid
        except Exception as exc:  # noqa: BLE001
            logger.debug("fallback to outputs.remote_folder: %s", exc)
            self.remote_data_uuid = calc.outputs.remote_folder.uuid

        orb_options = []
        pattern = re.compile(r"WFN_(\d+)_([12])\-")  # orbital index + spin

        for name in self.node.list_object_names():
            label = None
            if "WFN" in name:
                m = pattern.search(name)
                if not m:
                    continue
                n_orb = int(m.group(1))
                spin_index = int(m.group(2))  # 1->up, 2->down
                if self.uks:
                    n_morb = self.nel_up if spin_index == 1 else self.nel_dw
                    spin_prefix = "UP-" if spin_index == 1 else "DW-"
                else:
                    n_morb = self.nel_up
                    spin_prefix = ""
                if n_orb < n_morb:
                    label = f"{spin_prefix}HOMO - {n_morb - n_orb}"
                elif n_orb == n_morb:
                    label = f"{spin_prefix}HOMO"
                elif n_orb == n_morb + 1:
                    label = f"{spin_prefix}LUMO"
                else:
                    label = f"{spin_prefix}LUMO +{n_orb - n_morb - 1}"
            elif "ELECTRON" in name:
                label = "CHARGE-DENSITY"
            elif "SPIN" in name:
                label = "SPIN-DENSITY"
            elif "ChargeDiff" in name:
                label = "ChargeDifference"

            if label is not None:
                orb_options.append((label, name))

        self.cube_selector.value = None
        self.cube_selector.options = orb_options

    def append_render_instructions(self, _=None):
        if not self.render_name.value:
            self.error_message.value = "Please provide a render name"
            return
        self.error_message.value = ""

        iso_vals, iso_cols = [], []
        try:
            iv, ic = self._viewer.isovalues.return_values()
            iso_vals = list(iv)
            iso_cols = list(ic)
        except Exception as exc:  # noqa: BLE001
            logger.debug("no isovalues to capture: %s", exc)

        new_dict = copy.deepcopy(self.render_instructions)
        new_dict[self.render_name.value] = {
            "cube": self.cube_selector.value,
            "camera_orientation": self._viewer.viewer._camera_orientation,
            "isovalue": float(iso_vals[0]) if iso_vals else None,
            "isovalues": iso_vals,
            "colors": iso_cols,
            "format": "png",
        }
        self.render_instructions = new_dict

    @tl.observe("node_pk")
    def _observe_node_pk(self, _=None):
        if self.node_pk:
            self.select_calculation()
        else:
            self.node = None
            self.cube_selector.options = []
            self._viewer.cube = None
            self.render_instructions = {}

    @tl.observe("render_instructions")
    def _observe_render_instructions(self, _=None):
        self.render_instructions_widget.value = toml.dumps(self.render_instructions)

    def render_all(self, _=None):
        render_cube_file(self.render_instructions, orm.load_node(self.remote_data_uuid))


# ---------------------- rendered images viewer ----------------------------


class DisplayRenderedCubes(ipw.HBox):
    remote_data_uuid = tl.Unicode(allow_none=True)

    def __init__(self):
        self.rendered_images_widget = ipw.Select(
            description="Select an image:",
            layout=ipw.Layout(width="350px"),
            style={"description_width": "initial"},
        )
        self.rendered_images_widget.observe(self.show_image, names="value")
        self.image = ipw.Image(format="png", width=600, height=800)
        self.render_instructions_widget = ipw.Textarea(
            description="Render instructions:",
            style={"description_width": "initial"},
            layout=ipw.Layout(width="380px", height="300px"),
            disabled=True,
        )

        accordion = ipw.Accordion(
            children=[self.render_instructions_widget], layout=ipw.Layout(width="600px")
        )
        accordion.set_title(0, "Render instructions")
        accordion.selected_index = None
        super().__init__(
            [self.rendered_images_widget, ipw.VBox([accordion, self.image])]
        )

    def show_image(self, _=None):
        if not self.rendered_images_widget.value:
            self.render_instructions_widget.value = ""
            self.image.value = b""
            return
        fname, settings_uuid, folder_uuid = self.rendered_images_widget.value
        fd = orm.load_node(folder_uuid)
        settings = orm.load_node(settings_uuid)
        self.render_instructions_widget.value = toml.dumps(
            settings.get_dict()[self.rendered_images_widget.label]
        )
        self.image.value = fd.get_object_content(fname, mode="rb")

    @tl.observe("remote_data_uuid")
    def _observe_remote_data_uuid(self, _=None):
        if not self.remote_data_uuid:
            return
        query = orm.QueryBuilder()
        query.append(
            orm.RemoteData,
            filters={"uuid": self.remote_data_uuid},
            tag="original_cubes",
        )
        query.append(
            orm.CalcFunctionNode,
            filters={"label": "render_cube_file"},
            with_incoming="original_cubes",
            tag="render_calc",
        )
        query.append(orm.Dict, with_outgoing="render_calc", project="*")
        query.append(orm.FolderData, with_incoming="render_calc", project="uuid")

        self.rendered_images_widget.value = None
        select_image_list = []
        for dict_obj, folder_uuid in query.all():
            for name, value in dict_obj.items():
                select_image_list.append(
                    (name, (f"{name}.{value['format']}", dict_obj.uuid, folder_uuid))
                )
        self.rendered_images_widget.options = select_image_list


# ------------------------------ 2D slicer ---------------------------------


class CubePlaneCut2D(ipw.VBox):
    cube = tl.Instance(Cube, allow_none=True)

    def __init__(self):
        # free-text inputs (Cartesian Å)
        self.center_txt = ipw.Text(
            value="0.0 0.0 0.0",
            description="Center (Å)",
            placeholder="x y z",
            style={"description_width": "110px"},
            layout=ipw.Layout(width="420px"),
            tooltip="Enter Cartesian center, e.g. '9.0 9.1 8'",
        )
        self.normal_txt = ipw.Text(
            value="0 0 1",
            description="Normal",
            placeholder="nx ny nz",
            style={"description_width": "110px"},
            layout=ipw.Layout(width="420px"),
            tooltip="Enter Cartesian normal, e.g. '1 0 0'",
        )
        self.autonorm = ipw.Checkbox(value=True, description="Normalize normal")
        self.offset_txt = ipw.Text(
            value="0.0",
            description="Offset (Å)",
            placeholder="0.0",
            style={"description_width": "110px"},
            layout=ipw.Layout(width="220px"),
            tooltip="Shift the plane along the normal by this distance",
        )

        # Plane size & sampling
        self.width = ipw.FloatText(
            value=10.0,
            description="Width (Å)",
            style={"description_width": "110px"},
            layout=ipw.Layout(width="220px"),
        )
        self.height = ipw.FloatText(
            value=10.0,
            description="Height (Å)",
            style={"description_width": "110px"},
            layout=ipw.Layout(width="220px"),
        )
        self.res = ipw.IntSlider(
            value=256,
            min=32,
            max=1024,
            step=32,
            description="Resolution",
            continuous_update=False,
            layout=ipw.Layout(width="360px"),
        )

        # Plot options
        self.show_contours = ipw.Checkbox(value=False, description="Contours")
        self.vmin = ipw.FloatText(
            value=None,
            description="vmin",
            style={"description_width": "60px"},
            layout=ipw.Layout(width="150px"),
        )
        self.vmax = ipw.FloatText(
            value=None,
            description="vmax",
            style={"description_width": "60px"},
            layout=ipw.Layout(width="150px"),
        )
        self.update_btn = ipw.Button(
            description="Update 2D slice",
            button_style="success",
            layout=ipw.Layout(width="180px"),
        )

        # Info / error / output
        self.info = ipw.HTML("")
        self.error = ipw.HTML("")
        self._out = ipw.Output(
            layout=ipw.Layout(width="560px", height="560px", border="1px solid #ddd")
        )

        # Layout
        header = ipw.HTML("<b>2D plane cut (Cartesian point + normal, PBC)</b>")
        row_center = ipw.HBox([self.center_txt])
        row_normal = ipw.HBox([self.normal_txt, self.autonorm])
        row_offset = ipw.HBox([self.offset_txt])
        row_size = ipw.HBox([self.width, self.height, self.res])
        row_opts = ipw.HBox([self.show_contours, self.vmin, self.vmax, self.update_btn])

        super().__init__(
            [
                header,
                row_center,
                row_normal,
                row_offset,
                row_size,
                row_opts,
                self.info,
                self.error,
                self._out,
            ]
        )

        # Wiring
        self.update_btn.on_click(lambda _: self.plot_slice())
        for w in (
            self.center_txt,
            self.normal_txt,
            self.offset_txt,
            self.autonorm,
            self.width,
            self.height,
            self.res,
            self.show_contours,
            self.vmin,
            self.vmax,
        ):
            w.observe(self._on_any_change, names="value")

    # -------- auto-update guard (only when inputs are valid) ----------------

    def _vec3_is_valid(self, text: str) -> bool:
        toks = [t for t in re.split(r"[,\s]+", str(text).strip()) if t]
        if len(toks) != 3:
            return False
        try:
            _ = [float(t) for t in toks]
        except Exception:  # noqa: BLE001
            return False
        return True

    def _inputs_valid(self) -> bool:
        if not self.cube:
            return False
        if not self._vec3_is_valid(self.center_txt.value):
            return False
        if not self._vec3_is_valid(self.normal_txt.value):
            return False
        try:
            # offset, width/height/res (coerce to numbers)
            _ = float(self._parse_float(self.offset_txt.value))
            _ = float(self.width.value)
            _ = float(self.height.value)
            _ = int(self.res.value)
        except Exception:  # noqa: BLE001
            return False
        return True

    def _on_any_change(self, _=None):
        if self._inputs_valid():
            self.error.value = ""
            self.plot_slice()
        else:
            # while typing, keep UI calm (no red error spam)
            self.error.value = ""
            with self._out:
                self._out.clear_output(wait=True)

    # ------------------------------ helpers --------------------------------

    @staticmethod
    def _parse_vec3(text):
        toks = [t for t in re.split(r"[,\s]+", str(text).strip()) if t]
        if len(toks) != 3:
            raise ValueError("expected 3 numbers")
        return np.array([float(t) for t in toks], dtype=float)

    @staticmethod
    def _parse_float(text):
        s = str(text).strip()
        if s == "" or s.lower() == "none":
            return 0.0
        return float(s)

    @staticmethod
    def _orthonormal_basis_from_normal(n):
        n = np.asarray(n, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            n = np.array([0.0, 0.0, 1.0])
            n_norm = 1.0
        n_unit = n / n_norm
        t = (
            np.array([1.0, 0.0, 0.0])
            if abs(n_unit[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = np.cross(n_unit, t)
        u /= np.linalg.norm(u)
        v = np.cross(n_unit, u)
        v /= np.linalg.norm(v)
        return n_unit, u, v

    @staticmethod
    def _trilinear_sample_pbc(data, coords):
        nx, ny, nz = data.shape
        idx = np.mod(coords, np.array([nx, ny, nz]))
        i0 = np.floor(idx).astype(int)
        di = idx - i0
        i1 = (i0[..., 0] + 1) % nx
        j1 = (i0[..., 1] + 1) % ny
        k1 = (i0[..., 2] + 1) % nz
        i0x = i0[..., 0]
        j0y = i0[..., 1]
        k0z = i0[..., 2]
        c000 = data[i0x, j0y, k0z]
        c100 = data[i1, j0y, k0z]
        c010 = data[i0x, j1, k0z]
        c110 = data[i1, j1, k0z]
        c001 = data[i0x, j0y, k1]
        c101 = data[i1, j0y, k1]
        c011 = data[i0x, j1, k1]
        c111 = data[i1, j1, k1]
        tx = di[..., 0]
        ty = di[..., 1]
        tz = di[..., 2]
        c00 = c000 * (1 - tx) + c100 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    @tl.observe("cube")
    def _on_cube(self, _):
        if self.cube is None:
            self.info.value = ""
            self.error.value = ""
            with self._out:
                self._out.clear_output()
            return
        s = np.array(self.cube.ase_atoms.cell).T
        lengths = np.linalg.norm(s, axis=0)
        mean_len = float(np.mean(lengths))
        self.width.value = mean_len
        self.height.value = mean_len
        nx, ny, nz = self.cube.data.shape
        dmin = float(np.nanmin(self.cube.data))
        dmax = float(np.nanmax(self.cube.data))
        self.info.value = (
            f"Grid: {nx}×{ny}×{nz} | data ∈ [{dmin:.3e}, {dmax:.3e}] | "
            f"|a|,|b|,|c| ≈ {lengths[0]:.2f}, {lengths[1]:.2f}, {lengths[2]:.2f} Å"
        )
        self.plot_slice()

    def _build_plane_grid(self):
        atoms = self.cube.ase_atoms
        s = np.array(atoms.cell).T  # columns a, b, c

        center = self._parse_vec3(self.center_txt.value)  # Å
        n = self._parse_vec3(self.normal_txt.value)  # direction
        offset = self._parse_float(self.offset_txt.value)

        if self.autonorm.value:
            n_hat, u_hat, v_hat = self._orthonormal_basis_from_normal(n)
        else:
            n_hat_tmp, u_hat, v_hat = self._orthonormal_basis_from_normal(n)
            n_hat = (
                n / (np.linalg.norm(n) + 1e-12)
                if np.linalg.norm(n) > 1e-12
                else n_hat_tmp
            )

        center_shifted = center + offset * n_hat

        w = float(self.width.value)
        h = float(self.height.value)
        m = int(self.res.value)
        npts = int(self.res.value)

        us = np.linspace(-w / 2.0, w / 2.0, npts)
        vs = np.linspace(-h / 2.0, h / 2.0, m)
        uu, vv = np.meshgrid(us, vs)
        r = (
            center_shifted[None, None, :]
            + uu[..., None] * u_hat[None, None, :]
            + vv[..., None] * v_hat[None, None, :]
        )
        extent = [-w / 2.0, w / 2.0, -h / 2.0, h / 2.0]
        return r, extent, s

    def _cart_to_indexcoords(self, r, s):
        nx, ny, nz = self.cube.data.shape
        sinv = np.linalg.inv(s)
        rflat = r.reshape(-1, 3).T
        f = (sinv @ rflat).T
        idx = np.empty_like(f)
        idx[:, 0] = f[:, 0] * nx
        idx[:, 1] = f[:, 1] * ny
        idx[:, 2] = f[:, 2] * nz
        return idx.reshape(r.shape)

    def plot_slice(self):
        if self.cube is None:
            return
        try:
            r, extent, s = self._build_plane_grid()
            idxcoords = self._cart_to_indexcoords(r, s)
            vals = self._trilinear_sample_pbc(self.cube.data, idxcoords).reshape(
                r.shape[:2]
            )
            vmin = (
                self.vmin.value
                if self.vmin.value not in (None, "")
                else float(np.nanmin(vals))
            )
            vmax = (
                self.vmax.value
                if self.vmax.value not in (None, "")
                else float(np.nanmax(vals))
            )
        except Exception as exc:  # noqa: BLE001
            self.error.value = f"<span style='color:#b00'>Error: {exc}</span>"
            with self._out:
                self._out.clear_output(wait=True)
            return

        self.error.value = ""
        with self._out:
            self._out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            im = ax.imshow(
                vals,
                origin="lower",
                extent=extent,
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("u (Å)")
            ax.set_ylabel("v (Å)")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Value")
            if self.show_contours.value:
                try:
                    cs = ax.contour(
                        vals,
                        levels=8,
                        linewidths=0.8,
                        alpha=0.7,
                        origin="lower",
                        extent=extent,
                    )
                    ax.clabel(cs, inline=True, fontsize=8)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("contour overlay failed: %s", exc)
            plt.show()

    def current_plane_params(self):
        center = self._parse_vec3(self.center_txt.value)
        normal = self._parse_vec3(self.normal_txt.value)
        offset = self._parse_float(self.offset_txt.value)
        w = float(self.width.value)
        h = float(self.height.value)
        return center, normal, offset, w, h
