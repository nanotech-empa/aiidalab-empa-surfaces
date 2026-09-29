import copy
import logging
import re
import tempfile
from pathlib import PurePosixPath

import aiidalab_widgets_base as awb
import ase.io.cube
import ipywidgets as ipw
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nglview
import numpy as np
import toml
import traitlets as tl
from aiida import orm, plugins
from cubehandler import Cube
from PIL import ImageColor
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)

CubeHandlerCalculation = plugins.CalculationFactory("nanotech_empa.cubehandler")
Cp2kOrbitalsWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.orbitals")
Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")
Cp2kFragmentSeparationWorkChain = plugins.WorkflowFactory(
    "nanotech_empa.cp2k.fragment_separation"
)
ISO_OPTION_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*:\s*(#[0-9A-Fa-f]{6})\s*$"
)
ERR_EXPECT_3_FLOATS = "Expected 3 floats"


def _orthonormal_basis_from_normal(normal):
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm < 1.0e-12:
        n_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n_hat = normal / norm

    trial = (
        np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(n_hat[0]) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=float)
    )
    u_hat = np.cross(n_hat, trial)
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(n_hat, u_hat)
    v_hat /= np.linalg.norm(v_hat)
    return n_hat, u_hat, v_hat


def make_plane_mesh(
    center, normal, width, height, offset=0.0, colors_2d=None, max_grid_n=64
):
    n_hat, u_hat, v_hat = _orthonormal_basis_from_normal(normal)
    center = np.asarray(center, dtype=float) + float(offset) * n_hat
    width = float(width)
    height = float(height)

    if colors_2d is None:
        grid_n = 50
        colors = None
    else:
        grid_n = min(int(max_grid_n), colors_2d.shape[0] - 1)
        color_i = np.linspace(0, colors_2d.shape[0] - 1, grid_n, dtype=int)
        color_j = np.linspace(0, colors_2d.shape[1] - 1, grid_n, dtype=int)
        colors = colors_2d[np.ix_(color_i, color_j)]

    us = np.linspace(-width / 2, width / 2, grid_n + 1)
    vs = np.linspace(-height / 2, height / 2, grid_n + 1)
    vertices = []
    mesh_colors = []

    for i in range(grid_n):
        for j in range(grid_n):
            u0, u1 = us[i], us[i + 1]
            v0, v1 = vs[j], vs[j + 1]
            p00 = center + u0 * u_hat + v0 * v_hat
            p10 = center + u1 * u_hat + v0 * v_hat
            p11 = center + u1 * u_hat + v1 * v_hat
            p01 = center + u0 * u_hat + v1 * v_hat
            color = (0.6, 0.6, 0.6) if colors is None else colors[j, i]

            for point in (p00, p10, p11, p00, p11, p01):
                vertices.extend(point.tolist())
                mesh_colors.extend(color)

    return vertices, mesh_colors


def _normalize_color(color):
    rgb = ImageColor.getrgb(color)
    return "#{:02X}{:02X}{:02X}".format(*rgb[:3])


def _normalize_orientation(camera_orientation):
    orientation = [float(value) for value in camera_orientation]
    if len(orientation) != 16:
        err_msg = f"Camera orientation must contain exactly 16 numbers, but {len(orientation)} were provided."
        raise ValueError(err_msg)
    return orientation


def _normalize_isovalue_pairs(children):
    return [
        [float(child.children[0].value), _normalize_color(child.children[1].value)]
        for child in children
    ]


def _format_iso_option(isovalue, color):
    return f"{float(isovalue):.16g}:{_normalize_color(color)}"


def _render_name_from_output(output_path):
    output_name = PurePosixPath(output_path).name
    suffix = PurePosixPath(output_name).suffix
    return output_name[: -len(suffix)] if suffix else output_name


def _instructions_from_parameters(parameters):
    instructions = {}
    for step in parameters.get("steps", []):
        if step.get("command") != "render":
            continue

        options = step.get("options", {})
        args = step.get("args", [])
        output_path = options.get("output", "")
        render_name = _render_name_from_output(output_path)
        image_format = options.get("format", "png").lower()
        iso_pairs = []
        for iso_item in options.get("iso", []):
            match = ISO_OPTION_PATTERN.match(iso_item)
            if match is None:
                continue
            iso_pairs.append([float(match.group(1)), match.group(2).upper()])

        instructions[render_name] = {
            "cube": PurePosixPath(args[0]).name if args else "",
            "camera_orientation": [
                float(value) for value in options.get("orientation", [])
            ],
            "isovalues": iso_pairs,
            "format": image_format,
        }
    return instructions


class OneIsovalue(ipw.HBox):
    def __init__(self, structure=None):
        self.isovalue_widget = ipw.BoundedFloatText(
            value=1e-3,
            min=-1e-1,
            max=1e-1,
            step=1e-5,
            description="Isovalue",
        )
        self.color_widget = ipw.ColorPicker(
            concise=False, description="Pick a color", value="#00FFFF", disabled=False
        )
        super().__init__([self.isovalue_widget, self.color_widget])

    # proxy: così min/max/value su OneIsovalue agiscono sul BoundedFloatText
    @property
    def isomin(self):
        return self.isovalue_widget.min

    @isomin.setter
    def isomin(self, v):
        self.isovalue_widget.min = v

    @property
    def isomax(self):
        return self.isovalue_widget.max

    @isomax.setter
    def isomax(self, v):
        self.isovalue_widget.max = v

    @property
    def value(self):
        return self.isovalue_widget.value

    @value.setter
    def value(self, v):
        self.isovalue_widget.value = v


class IsovaluesWidget(ipw.VBox):
    iso_min = tl.Float(1.0e-5, allow_none=True)
    iso_max = tl.Float(1.0e-1, allow_none=True)

    def __init__(self):
        self.isovalues = ipw.VBox()

        # Add isovalue button.
        self.info = ipw.HTML(f"min: {self.iso_min:.1e}, max: {self.iso_max:.1e}")
        self.add_isovalue_button = ipw.Button(
            description="Add isovalue",
            layout={"width": "initial"},
            button_style="success",
        )
        self.add_isovalue_button.on_click(self.add_isovalue)

        # Remove isovalue button.
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
                    [
                        self.info,
                        self.add_isovalue_button,
                        self.remove_isovalue_button,
                    ]
                ),
            ]
        )

    def return_isovalues_and_colors(self):
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
            if not (vmin <= isovalue.children[0].value <= vmax):
                isovalue.children[0].value = default

    @tl.observe("iso_min", "iso_max")
    def _observe_range(self, _=None):
        self.set_range(vmin=self.iso_min, vmax=self.iso_max)

    def add_isovalue(self, _=None):
        new = OneIsovalue()
        new.isomin = self.iso_min
        new.isomax = self.iso_max
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


class CubePlaneCut2D(ipw.VBox):
    """2D plane-cut viewer for cube data."""

    cube = tl.Instance(Cube, allow_none=True)

    def __init__(self):
        self._last_r = None
        self._last_vals = None
        self._last_rgb = None
        self._sinv = None
        self._updating_color_range = False

        self.center_txt = ipw.Text(
            value="0.0 0.0 0.0",
            description="Center (A)",
            placeholder="x y z",
            layout=ipw.Layout(width="420px"),
            style={"description_width": "110px"},
        )
        self.normal_txt = ipw.Text(
            value="0 0 1",
            description="Normal",
            placeholder="nx ny nz",
            layout=ipw.Layout(width="420px"),
            style={"description_width": "110px"},
        )
        self.offset_txt = ipw.Text(
            value="0.0",
            description="Offset (A)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )
        self.width = ipw.FloatText(
            value=10.0,
            description="Width (A)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )
        self.height = ipw.FloatText(
            value=10.0,
            description="Height (A)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )
        self.res = ipw.IntSlider(
            value=64,
            min=32,
            max=256,
            step=32,
            description="2D slice res.",
            continuous_update=False,
            layout=ipw.Layout(width="360px"),
            style={"description_width": "90px"},
        )
        self.show_contours = ipw.Checkbox(value=False, description="2D contours")
        self.auto_color_range = ipw.Checkbox(
            value=True,
            description="Auto color range",
        )
        self.auto_color_range.observe(self._on_auto_color_range_toggled, names="value")
        self.vmin = ipw.FloatText(
            value=0.0,
            description="vmin",
            disabled=True,
            layout=ipw.Layout(width="150px"),
            style={"description_width": "60px"},
        )
        self.vmax = ipw.FloatText(
            value=0.0,
            description="vmax",
            disabled=True,
            layout=ipw.Layout(width="150px"),
            style={"description_width": "60px"},
        )
        self.show_2d_checkbox = ipw.Checkbox(value=True, description="Show 2D plot")
        self.show_2d_checkbox.observe(self._on_show_2d_toggled, names="value")
        self.plane_preset = ipw.Dropdown(
            options=[
                ("XY", "0 0 1"),
                ("XZ", "0 1 0"),
                ("YZ", "1 0 0"),
            ],
            description="Plane:",
            layout=ipw.Layout(width="180px"),
            style={"description_width": "55px"},
        )
        self.apply_plane_preset = ipw.Button(
            description="Apply",
            layout=ipw.Layout(width="80px"),
        )
        self.apply_plane_preset.on_click(self._on_apply_plane_preset)
        self.center_on_atoms = ipw.Button(
            description="Center on atoms",
            layout=ipw.Layout(width="140px"),
        )
        self.center_on_atoms.on_click(self._on_center_on_atoms)
        self.info = ipw.HTML("")
        self.error = ipw.HTML("")
        self._out = ipw.Output(
            layout=ipw.Layout(width="560px", height="560px", border="1px solid #ddd")
        )

        super().__init__(
            [
                ipw.HTML("<b>2D plane cut (Cartesian point + normal, PBC)</b>"),
                ipw.HBox([self.center_txt]),
                ipw.HBox([self.normal_txt]),
                ipw.HBox(
                    [
                        self.plane_preset,
                        self.apply_plane_preset,
                        self.center_on_atoms,
                    ]
                ),
                ipw.HBox([self.offset_txt]),
                ipw.HBox([self.width, self.height, self.res]),
                ipw.HBox(
                    [self.show_contours, self.auto_color_range, self.vmin, self.vmax]
                ),
                self.show_2d_checkbox,
                self.info,
                self.error,
                self._out,
            ]
        )

        for widget in (
            self.center_txt,
            self.normal_txt,
            self.offset_txt,
            self.width,
            self.height,
            self.res,
            self.show_contours,
            self.auto_color_range,
            self.vmin,
            self.vmax,
        ):
            widget.observe(self._on_any_change, names="value")

    def _vec3_is_valid(self, text):
        tokens = [token for token in re.split(r"[,\s]+", text.strip()) if token]
        if len(tokens) != 3:
            return False
        try:
            [float(token) for token in tokens]
        except ValueError:
            return False
        return True

    def _inputs_valid(self):
        if self.cube is None:
            return False
        if not self._vec3_is_valid(self.center_txt.value):
            return False
        if not self._vec3_is_valid(self.normal_txt.value):
            return False
        try:
            float(self.offset_txt.value)
            float(self.width.value)
            float(self.height.value)
            int(self.res.value)
        except (TypeError, ValueError):
            return False
        return True

    def _on_any_change(self, _=None):
        if self._updating_color_range:
            return
        if self._inputs_valid():
            self.error.value = ""
            self.plot_slice()
        else:
            self.error.value = ""
            with self._out:
                self._out.clear_output(wait=True)

    def _on_show_2d_toggled(self, change):
        if change["new"]:
            self._out.layout.display = ""
            if self._inputs_valid():
                self.plot_slice()
        else:
            self._out.clear_output()
            self._out.layout.display = "none"
            if self._inputs_valid():
                self.plot_slice()

    def _on_apply_plane_preset(self, _=None):
        self.normal_txt.value = self.plane_preset.value

    def _on_center_on_atoms(self, _=None):
        if self.cube is None:
            return
        center = np.mean(self.cube.ase_atoms.get_positions(), axis=0)
        self.center_txt.value = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"

    def _on_auto_color_range_toggled(self, change):
        use_auto = change["new"]
        self.vmin.disabled = use_auto
        self.vmax.disabled = use_auto
        if self._inputs_valid():
            self.plot_slice()

    @staticmethod
    def _parse_vec3(text):
        values = np.fromstring(text.replace(",", " "), sep=" ", dtype=float)
        if values.size != 3:
            raise ValueError(ERR_EXPECT_3_FLOATS)
        return values

    @tl.observe("cube")
    def _on_cube(self, _=None):
        if self.cube is None:
            self._sinv = None
            self.info.value = ""
            with self._out:
                self._out.clear_output()
            return

        cell = np.array(self.cube.ase_atoms.cell).T
        self._sinv = np.linalg.inv(cell)
        center = np.mean(self.cube.ase_atoms.get_positions(), axis=0)
        self.center_txt.value = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"

        nx, ny, nz = self.cube.data.shape
        dmin = float(np.min(self.cube.data))
        dmax = float(np.max(self.cube.data))
        lengths = np.linalg.norm(cell, axis=0)
        self.info.value = (
            f"Grid: {nx} x {ny} x {nz} | data in [{dmin:.3e}, {dmax:.3e}] "
            f"| cell lengths: {lengths[0]:.2f}, {lengths[1]:.2f}, "
            f"{lengths[2]:.2f} A"
        )
        mean_len = float(np.mean(lengths))
        self.width.value = mean_len
        self.height.value = mean_len
        self.plot_slice()

    def _build_plane_grid(self):
        center = self._parse_vec3(self.center_txt.value)
        normal = self._parse_vec3(self.normal_txt.value)
        n_hat, u_hat, v_hat = _orthonormal_basis_from_normal(normal)

        offset = float(self.offset_txt.value)
        center_shifted = center + offset * n_hat
        width = float(self.width.value)
        height = float(self.height.value)
        resolution = int(self.res.value)

        us = np.linspace(-width / 2, width / 2, resolution)
        vs = np.linspace(-height / 2, height / 2, resolution)
        uu, vv = np.meshgrid(us, vs)
        grid = (
            center_shifted[None, None, :]
            + uu[..., None] * u_hat
            + vv[..., None] * v_hat
        )
        extent = [-width / 2, width / 2, -height / 2, height / 2]
        return grid, extent

    def _cart_to_indexcoords(self, grid):
        nx, ny, nz = self.cube.data.shape
        fractional = (self._sinv @ grid.reshape(-1, 3).T).T
        indexes = np.empty_like(fractional)
        indexes[:, 0] = fractional[:, 0] * nx
        indexes[:, 1] = fractional[:, 1] * ny
        indexes[:, 2] = fractional[:, 2] * nz
        return indexes.reshape(grid.shape)

    def _compute_slice_values(self):
        grid, extent = self._build_plane_grid()
        index_coords = self._cart_to_indexcoords(grid)
        coords = np.stack(
            [index_coords[..., 0], index_coords[..., 1], index_coords[..., 2]], axis=0
        )
        values = map_coordinates(
            self.cube.data, coords, order=1, mode="wrap", prefilter=False
        )
        return grid, extent, values

    def _effective_color_range(self, values):
        if self.auto_color_range.value:
            return float(np.min(values)), float(np.max(values))
        return float(self.vmin.value), float(self.vmax.value)

    def _store_slice_cache(self, grid, values, vmin, vmax):
        normalizer = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("viridis")
        self._last_r = grid
        self._last_vals = values
        self._last_rgb = cmap(normalizer(values))[..., :3]

    def plot_slice(self):
        if self.cube is None:
            return

        try:
            grid, extent, values = self._compute_slice_values()
            vmin, vmax = self._effective_color_range(values)
        except Exception:
            logger.debug("cube plane slice failed", exc_info=True)
            self.error.value = "<span style='color:#b00'>Error computing slice</span>"
            with self._out:
                self._out.clear_output()
            return

        if self.auto_color_range.value:
            self._updating_color_range = True
            try:
                self.vmin.value = vmin
                self.vmax.value = vmax
            finally:
                self._updating_color_range = False

        self._store_slice_cache(grid, values, vmin, vmax)
        if not self.show_2d_checkbox.value:
            self._out.layout.display = "none"
            return

        self._out.layout.display = ""
        with self._out:
            self._out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
            image = ax.imshow(
                values,
                origin="lower",
                extent=extent,
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("u (A)")
            ax.set_ylabel("v (A)")
            fig.colorbar(image, ax=ax)

            if self.show_contours.value and not np.isclose(vmin, vmax):
                contour_levels = np.linspace(vmin, vmax, 9)[1:-1]
                ax.contour(
                    values,
                    levels=contour_levels,
                    origin="lower",
                    extent=extent,
                    colors="black",
                    linewidths=1.2,
                    alpha=0.65,
                )
                ax.contour(
                    values,
                    levels=contour_levels,
                    origin="lower",
                    extent=extent,
                    colors="white",
                    linewidths=0.55,
                    alpha=0.95,
                )

            plt.show()
            plt.close(fig)

    def current_plane_params(self):
        center = self._parse_vec3(self.center_txt.value)
        normal = self._parse_vec3(self.normal_txt.value)
        n_hat, _, _ = _orthonormal_basis_from_normal(normal)
        return (
            center,
            n_hat,
            float(self.offset_txt.value),
            float(self.width.value),
            float(self.height.value),
        )

    def return_settings(self):
        center, normal, offset, width, height = self.current_plane_params()
        return {
            "center": center.tolist(),
            "normal": normal.tolist(),
            "offset": offset,
            "width": width,
            "height": height,
            "resolution": int(self.res.value),
            "show_2d": bool(self.show_2d_checkbox.value),
            "show_contours": bool(self.show_contours.value),
            "auto_color_range": bool(self.auto_color_range.value),
            "vmin": float(self.vmin.value),
            "vmax": float(self.vmax.value),
        }


class CubeArrayData3dViewerWidget(ipw.VBox):
    """Widget to view cube data in 3D and overlay a slicing plane."""

    cube = tl.Instance(Cube, allow_none=True)

    def __init__(self, **kwargs):
        self.structure = None
        self.viewer = nglview.NGLWidget()
        self.isovalues = IsovaluesWidget()
        self._loaded_cube = None
        self._structure_component = None
        self._cube_component = None
        self._plane_component = None
        self.slice2d = None
        self._external_plane_sync_cb = None

        self.show_isosurfaes_button = ipw.Button(
            description="Update isosurfaces",
            layout={"width": "initial"},
            button_style="success",
        )
        self.show_isosurfaes_button.on_click(lambda _: self.apply_isosurfaces())
        self.show_plane_checkbox = ipw.Checkbox(
            value=True, description="Show slicing plane in 3D"
        )
        self.show_plane_checkbox.observe(self._on_plane_toggle, names="value")
        self.plane_resolution = ipw.BoundedIntText(
            value=48,
            min=16,
            max=128,
            step=16,
            description="3D plane res.:",
            layout=ipw.Layout(width="180px"),
            style={"description_width": "100px"},
        )

        super().__init__(
            [
                self.viewer,
                self.isovalues,
                ipw.HBox(
                    [
                        self.show_isosurfaes_button,
                        self.show_plane_checkbox,
                        self.plane_resolution,
                    ]
                ),
            ],
            **kwargs,
        )

    @tl.observe("cube")
    def on_observe_cube(self, _=None):
        """Update object attributes when cube trait is modified."""
        if self.cube is None:
            self.structure = None
            self._loaded_cube = None
            self._remove_component("_structure_component")
            self._remove_component("_cube_component")
            self.hide_and_remove_plane()
            return

        self.isovalues.set_range(
            vmin=float(np.min(self.cube.data)), vmax=float(np.max(self.cube.data))
        )
        self.structure = self.cube.ase_atoms
        self.update_plot()

    def _remove_component(self, attr):
        component = getattr(self, attr, None)
        if component is None:
            return
        try:
            self.viewer.remove_component(component.id)
        except Exception:
            logger.debug("remove_component failed", exc_info=True)
        setattr(self, attr, None)

    def update_plot(self):
        """Update the 3D plot."""
        if self._loaded_cube is self.cube and self._cube_component is not None:
            self.apply_isosurfaces()
            if self._external_plane_sync_cb:
                self._external_plane_sync_cb()
            return

        self._remove_component("_structure_component")
        self._remove_component("_cube_component")
        if self.structure is None or self.cube is None:
            return

        self._structure_component = self.viewer.add_component(
            nglview.ASEStructure(self.structure)
        )
        with tempfile.NamedTemporaryFile(mode="w") as tempf:
            ase.io.cube.write_cube(tempf, self.structure, self.cube.data)
            self._cube_component = self.viewer.add_component(tempf.name, ext="cube")
            self._cube_component.clear()
        self._loaded_cube = self.cube
        self.apply_isosurfaces()

        if self._external_plane_sync_cb:
            self._external_plane_sync_cb()

    def apply_isosurfaces(self):
        try:
            isovalues, colors = self.isovalues.return_isovalues_and_colors()
            self.set_cube_isosurf(isovalues, colors)
        except ValueError:
            pass
        except Exception:
            logger.debug("set_cube_isosurf failed", exc_info=True)

    def set_cube_isosurf(self, isovals, colors):
        """Set cube isosurfaces."""
        if self._cube_component is None:
            return
        self._cube_component.clear()
        for isov, col in zip(isovals, colors):
            try:
                self._cube_component.add_surface(
                    color=col, isolevelType="value", isolevel=isov
                )
            except Exception:
                logger.debug("add_surface failed", exc_info=True)

    def _capture_newest_component_as_plane(self, before_attrs):
        after = {name for name in dir(self.viewer) if name.startswith("component_")}
        new_attrs = sorted(after - before_attrs)
        if not new_attrs:
            return
        try:
            self._plane_component = getattr(self.viewer, new_attrs[-1])
        except Exception:
            logger.debug("capturing plane component failed", exc_info=True)

    def hide_and_remove_plane(self):
        if self._plane_component is None:
            return
        try:
            self.viewer.remove_component(self._plane_component.id)
        except Exception:
            logger.debug("remove plane component failed", exc_info=True)
        self._plane_component = None

    def _on_plane_toggle(self, change):
        if self._external_plane_sync_cb and change["new"]:
            self._external_plane_sync_cb()
            return
        if self._plane_component is None:
            return
        try:
            if change["new"]:
                self._plane_component.show()
            else:
                self._plane_component.hide()
        except Exception:
            logger.debug("plane show/hide failed", exc_info=True)

    def update_plane_mesh(self, *, center, normal, width, height, offset=0.0):
        if not self.show_plane_checkbox.value:
            self.hide_and_remove_plane()
            return

        self.hide_and_remove_plane()
        before = {name for name in dir(self.viewer) if name.startswith("component_")}
        colors_2d = getattr(self.slice2d, "_last_rgb", None)

        try:
            vertices, colors = make_plane_mesh(
                center,
                normal,
                width,
                height,
                offset,
                colors_2d=colors_2d,
                max_grid_n=self.plane_resolution.value,
            )
            self.viewer.shape.add_mesh(vertices, colors)
        except Exception:
            logger.debug("plane mesh update failed", exc_info=True)
            return

        self._capture_newest_component_as_plane(before)


def render_cube_file(settings, code_uuid, render_source_remote_uuid, remote_data_uuid):
    code = orm.load_code(code_uuid)
    builder = code.get_builder()
    builder.parameters = orm.Dict(
        dict={
            "steps": [
                {
                    "command": "render",
                    "args": [f"cube_source/{value['cube'].replace('reduced_', '', 1)}"],
                    "options": {
                        "orientation": _normalize_orientation(
                            value["camera_orientation"]
                        ),
                        "iso": [
                            _format_iso_option(isovalue, color)
                            for isovalue, color in value["isovalues"]
                        ],
                        "format": value["format"],
                        "output": f"out_cubes/{name}.{value['format']}",
                    },
                }
                for name, value in settings.items()
            ]
        }
    )
    builder.parent_folders = {
        "cube_source": orm.load_node(remote_data_uuid),
    }
    builder.metadata.label = "cube-render"
    builder.metadata.description = (
        f"Render {len(settings)} cube view(s) from {render_source_remote_uuid}"
    )
    builder.metadata.options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        },
        "max_wallclock_seconds": 600,
    }
    return builder


class HandleCubeFiles(ipw.VBox):
    node_pk = tl.Int(None, allow_none=True)
    nel_up = tl.Int(None, allow_none=True)
    nel_dw = tl.Int(None, allow_none=True)
    uks = tl.Bool(False)
    render_instructions = tl.Dict({}, allow_none=True)
    remote_data_uuid = tl.Unicode(allow_none=True)
    render_source_remote_uuid = tl.Unicode(allow_none=True)

    def __init__(self):
        self.node = None
        self._cube_cache = {}
        self._viewer = CubeArrayData3dViewerWidget(layout=ipw.Layout(width="550px"))
        self.slice2d = CubePlaneCut2D()
        self._viewer.slice2d = self.slice2d
        self._cube_sync = tl.dlink((self._viewer, "cube"), (self.slice2d, "cube"))

        def _sync_plane(_=None):
            try:
                if self.slice2d.cube is None or not self.slice2d._inputs_valid():
                    self._viewer.hide_and_remove_plane()
                    return
                center, normal, offset, width, height = (
                    self.slice2d.current_plane_params()
                )
                self._viewer.update_plane_mesh(
                    center=center,
                    normal=normal,
                    width=width,
                    height=height,
                    offset=offset,
                )
            except Exception:
                logger.debug("sync plane failed", exc_info=True)

        for widget in (
            self.slice2d.center_txt,
            self.slice2d.normal_txt,
            self.slice2d.offset_txt,
            self.slice2d.width,
            self.slice2d.height,
            self.slice2d.res,
            self.slice2d.show_contours,
            self.slice2d.auto_color_range,
            self.slice2d.vmin,
            self.slice2d.vmax,
            self._viewer.plane_resolution,
        ):
            widget.observe(_sync_plane, names="value")

        self._viewer._external_plane_sync_cb = _sync_plane
        self._sync_plane = _sync_plane
        self.cube_selector = ipw.Select(
            options=[],
            description="Cube files:",
            layout=ipw.Layout(width="400px"),
            style={"description_width": "initial"},
        )
        self.cube_selector.observe(self.show_selected_cube)
        self.dict_cube_files = {}
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
        self.render_wip_message = ipw.HTML(
            "<div style='border-left: 4px solid #f0ad4e; padding: 6px 10px; "
            "background: #fff8e8; max-width: 380px;'>"
            "<strong>&#9888; High-resolution rendering is work in progress.</strong><br>"
            "Cube file visualization is available now; render submission will be "
            "enabled once cubehandler rendering support is merged."
            "</div>"
        )
        self.error_message = ipw.HTML()
        self.render_instructions_widget = ipw.Textarea(
            description="Render instructions:",
            style={"description_width": "initial"},
            layout=ipw.Layout(width="380px", height="300px"),
        )

        self.render_instructions_widget.disabled = True

        self.cubehandler_code_widget = awb.ComputationalResourcesWidget(
            description="Cubehandler code:",
            default_calc_job_plugin="nanotech_empa.cubehandler",
        )
        self.render_submit = awb.SubmitButtonWidget(
            CubeHandlerCalculation,
            self.prepare_render_submission,
            description="Render all",
            disable_after_submit=False,
            append_output=False,
        )
        self.render_submit.on_submitted(self._refresh_render_results)

        # self.select_calculation()
        # The render submission controls require cubehandler's render command, which
        # is not available in the currently pinned cubehandler dependency.
        super().__init__(
            [
                # self.select_calc_widget,
                ipw.HBox(
                    [
                        ipw.VBox([self._viewer, self.slice2d]),
                        ipw.VBox(
                            [
                                self.cube_selector,
                                self.render_wip_message,
                                self.error_message,
                            ]
                        ),
                    ]
                ),
            ]
        )

    def show_selected_cube(self, _=None):

        if not self.cube_selector.value or self.node is None:
            return
        cube_name = self.cube_selector.value
        try:
            if cube_name not in self._cube_cache:
                self._cube_cache[cube_name] = Cube.from_content(
                    self.node.get_object_content(f"out_cubes/{cube_name}")
                )
            self._viewer.cube = self._cube_cache[cube_name]
            self._sync_plane()
        except FileNotFoundError:
            self.error_message.value = (
                "<span style='color:red'>Selected cube file is no longer available "
                "in the retrieved out_cubes folder.</span>"
            )

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
            CubeHandlerCalculation,
            filters={
                "label": {"in": ["cube-shrink", "charge-lowres", "ChargeDiff-lowres"]}
            },
            project="uuid",
            with_incoming="orbitals_wc",
        )
        return query.all()

    def select_calculation(self, _=None):
        # if not self.select_calc_widget.value:
        #    return
        # calc = orm.load_node(self.select_calc_widget.value)
        if not self.node_pk:
            return
        calc = orm.load_node(self.node_pk)
        self.node = calc.outputs.retrieved
        self._cube_cache = {}
        self.render_source_remote_uuid = calc.outputs.remote_folder.uuid

        parent_folders = getattr(calc.inputs, "parent_folders", None)
        if parent_folders and "folder1" in parent_folders:
            self.remote_data_uuid = parent_folders["folder1"].uuid
        else:
            try:
                self.remote_data_uuid = calc.inputs.nodes.remote_previous_job.uuid
            except Exception:
                self.remote_data_uuid = calc.outputs.remote_folder.uuid

        self.cubehandler_code_widget.value = calc.inputs.code.uuid

        orb_options = []
        pattern = re.compile(
            r"WFN_(\d+)_([12])\-"
        )  # captures orbital number and spin (1 or 2)

        try:
            cube_names = self.node.list_object_names("out_cubes")
        except (FileNotFoundError, NotADirectoryError):
            self.cube_selector.value = None
            self.cube_selector.options = []
            self.error_message.value = (
                f"<span style='color:red'>Node {self.node_pk} has no retrieved "
                "out_cubes folder. Select a finished cubehandler calculation "
                "created with the current nanotech_empa.cubehandler workflow, "
                "or rerun cubehandler for this calculation.</span>"
            )
            return

        self.error_message.value = ""
        for name in cube_names:
            label = None
            if "WFN" in name:
                m = pattern.search(name)
                if not m:
                    continue  # skip unexpected names

                n_orb = int(m.group(1))  # e.g. 00002 -> 2
                spin_index = int(m.group(2))

                # choose the correct HOMO index to compare against
                if self.uks:
                    n_morb = self.nel_up if spin_index == 1 else self.nel_dw
                    spin = "UP-" if spin_index == 1 else "DW-"
                else:
                    spin = ""
                    n_morb = self.nel_up  # only spin-1 files exist in RKS

                # assign label
                if n_orb < n_morb:
                    label = f"{spin}HOMO - {n_morb - n_orb}"
                elif n_orb == n_morb:
                    label = f"{spin}HOMO"
                elif n_orb == n_morb + 1:
                    label = f"{spin}LUMO"
                else:
                    label = f"{spin}LUMO +{n_orb - n_morb - 1}"

            elif "ELECTRON" in name:
                label = "CHARGE-DENSITY"
            elif "SPIN" in name:
                label = "SPIN-DENSITY"
            elif "ChargeDiff" in name:
                label = "ChargeDifference"
            else:
                label = None
            if label is not None:
                orb_options.append((label, name))

        self.cube_selector.value = None
        self.cube_selector.options = orb_options

    def append_render_instructions(self, _=None):
        if not self.render_name.value:
            self.error_message.value = "Please provide a render name"
            return
        if not self.cube_selector.value:
            self.error_message.value = "Please select a cube file"
            return
        else:
            self.error_message.value = ""
        isovalue_pairs = _normalize_isovalue_pairs(
            self._viewer.isovalues.isovalues.children
        )
        if not isovalue_pairs:
            self.error_message.value = "Please add at least one isovalue"
            return
        new_dict = copy.deepcopy(self.render_instructions)
        new_dict[self.render_name.value] = {
            "cube": self.cube_selector.value,
            "camera_orientation": _normalize_orientation(
                self._viewer.viewer._camera_orientation
            ),
            "isovalues": isovalue_pairs,
            "format": "png",
        }
        self.render_instructions = new_dict

    @tl.observe("node_pk")
    def _observe_node_pk(self, _=None):
        if self.node_pk:
            self.select_calculation()
        else:
            self.node = None
            self._cube_cache = {}
            self.cube_selector.options = []
            self._viewer.cube = None
            self.render_instructions = {}
            self.remote_data_uuid = None
            self.render_source_remote_uuid = None
            self._viewer.hide_and_remove_plane()

    @tl.observe("render_instructions")
    def _observe_render_instructions(self, _=None):
        # Format the render instructions using toml and display them in the widget
        self.render_instructions_widget.value = toml.dumps(self.render_instructions)

    def prepare_render_submission(self):
        if not self.render_instructions:
            self.error_message.value = "Please add at least one render instruction"
            return None
        if not self.cubehandler_code_widget.value:
            self.error_message.value = "Please select a cubehandler code"
            return None
        if not self.render_source_remote_uuid:
            self.error_message.value = "No render source remote folder was found"
            return None
        if not self.remote_data_uuid:
            self.error_message.value = "No source remote folder was found"
            return None
        if any(not value["isovalues"] for value in self.render_instructions.values()):
            self.error_message.value = "Each render needs at least one isovalue"
            return None

        self.error_message.value = ""
        return render_cube_file(
            self.render_instructions,
            self.cubehandler_code_widget.value,
            self.render_source_remote_uuid,
            self.remote_data_uuid,
        )

    def render_all(self, _=None):
        self.render_submit.on_btn_submit_press()
        return self.render_submit.process

    def _refresh_render_results(self, _=None):
        if not self.remote_data_uuid:
            return
        remote_data_uuid = self.remote_data_uuid
        self.remote_data_uuid = None
        self.remote_data_uuid = remote_data_uuid


class DisplayRenderedCubes(ipw.HBox):
    remote_data_uuid = tl.Unicode(allow_none=True)

    def __init__(self):
        self._render_records = {}
        self.rendered_images_widget = ipw.Select(
            description="Select an image:",
            layout=ipw.Layout(width="350px"),
            style={"description_width": "initial"},
        )
        self.rendered_images_widget.observe(self.show_image)
        self.refresh_button = ipw.Button(
            description="Refresh renders", layout=ipw.Layout(width="160px")
        )
        self.refresh_button.on_click(self.refresh)
        self.image = ipw.Image(
            format="png",
            width=600,
            height=800,
        )
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
            [
                ipw.VBox([self.refresh_button, self.rendered_images_widget]),
                ipw.VBox([accordion, self.image]),
            ]
        )

    def show_image(self, _=None):
        if not self.rendered_images_widget.value:
            self.render_instructions_widget.value = ""
            self.image.value = b""
            return
        calc_uuid, fname, render_name = self.rendered_images_widget.value
        calc = orm.load_node(calc_uuid)
        instruction = _instructions_from_parameters(
            calc.inputs.parameters.get_dict()
        ).get(render_name, {})
        self.render_instructions_widget.value = toml.dumps({render_name: instruction})
        image_format = PurePosixPath(fname).suffix.lower().lstrip(".")
        if image_format == "jpg":
            image_format = "jpeg"
        self.image.format = image_format or "png"
        self.image.value = calc.outputs.retrieved.get_object_content(fname, mode="rb")

    def refresh(self, _=None):
        self._observe_remote_data_uuid()

    @tl.observe("remote_data_uuid")
    def _observe_remote_data_uuid(self, _=None):
        if not self.remote_data_uuid:
            self.rendered_images_widget.options = []
            self.rendered_images_widget.value = None
            return
        query = orm.QueryBuilder()
        query.append(
            orm.RemoteData,
            filters={"uuid": self.remote_data_uuid},
            tag="original_cubes",
        )
        query.append(
            CubeHandlerCalculation,
            filters={"label": "cube-render"},
            with_incoming="original_cubes",
            tag="render_calc",
            project="*",
        )
        query.order_by({CubeHandlerCalculation: {"ctime": "desc"}})

        self.rendered_images_widget.value = None
        select_image_list = []
        self._render_records = {}
        for calc in query.all(flat=True):
            if not calc.is_finished_ok or "retrieved" not in calc.outputs:
                continue
            instructions = _instructions_from_parameters(
                calc.inputs.parameters.get_dict()
            )
            retrieved = calc.outputs.retrieved
            object_names = set(retrieved.list_object_names("out_cubes"))
            for render_name, value in instructions.items():
                file_name = f"{render_name}.{value['format']}"
                if file_name not in object_names:
                    continue
                label = f"{render_name} (pk: {calc.pk})"
                repo_path = f"out_cubes/{file_name}"
                self._render_records[label] = (calc.uuid, repo_path, render_name)
                select_image_list.append((label, (calc.uuid, repo_path, render_name)))
        self.rendered_images_widget.options = select_image_list
