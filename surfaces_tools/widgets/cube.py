import copy
import logging
import re
import tempfile

import ase.io.cube
import ipywidgets as ipw
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nglview
import numpy as np
import toml
import traitlets as tl
from aiida import engine, orm, plugins
from aiida_shell import ShellJob
from cubehandler import Cube
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)

ERR_EXPECT_3_FLOATS = "Expected 3 floats"


class Vec3ParseError(ValueError):
    """Expected 3 numbers for a 3-vector."""


# Optional (only used by get_calcs/select_calculation)
Cp2kOrbitalsWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.orbitals")
Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")
Cp2kFragmentSeparationWorkChain = plugins.WorkflowFactory(
    "nanotech_empa.cp2k.fragment_separation"
)

# ============================================================
# Helpers to construct an orthonormal frame and a plane mesh
# ============================================================


def _orthonormal_basis_from_normal(
    n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given an arbitrary normal vector `n`, construct a right-handed,
    orthonormal basis (n_hat, u_hat, v_hat):

        - n_hat = normalized normal
        - u_hat = any unit vector perpendicular to n_hat
        - v_hat = n_hat × u_hat

    This is needed to build a 3D plane oriented according to `normal`.
    """
    # Convert to float array
    n = np.asarray(n, dtype=float)

    # --- Normalize the normal vector ---
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        # Degenerate case: if the normal is zero, just use +z.
        n_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        n_hat = n / n_norm

    # --- Choose an auxiliary vector not parallel to n_hat ---
    # If the x-component is small, use x-axis; otherwise use y-axis.
    # This avoids n_hat × t ≈ 0 (unstable).
    if abs(n_hat[0]) < 0.9:
        t = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        t = np.array([0.0, 1.0, 0.0], dtype=float)

    # --- Build u_hat as a vector perpendicular to n_hat ---
    u = np.cross(n_hat, t)
    u /= np.linalg.norm(u)

    # --- Build v_hat as the third axis of the right-handed system ---
    v = np.cross(n_hat, u)
    v /= np.linalg.norm(v)

    return n_hat, u, v


def make_plane_mesh(center, normal, width, height, offset=0.0, colors_2d=None):
    import numpy as np

    # 1) Build orthonormal basis
    n_hat, u_hat, v_hat = _orthonormal_basis_from_normal(normal)

    # 2) Shift the plane
    c = np.asarray(center, float) + offset * n_hat
    w, h = float(width), float(height)

    # 3) Determine grid resolution
    if colors_2d is None:
        grid_n = 50
        colors = None
    else:
        grid_n = colors_2d.shape[0] - 1
        colors = colors_2d

    us = np.linspace(-w / 2, w / 2, grid_n + 1)
    vs = np.linspace(-h / 2, h / 2, grid_n + 1)

    verts = []
    cols = []

    for i in range(grid_n):
        for j in range(grid_n):

            u0, u1 = us[i], us[i + 1]
            v0, v1 = vs[j], vs[j + 1]

            p00 = c + u0 * u_hat + v0 * v_hat
            p10 = c + u1 * u_hat + v0 * v_hat
            p11 = c + u1 * u_hat + v1 * v_hat
            p01 = c + u0 * u_hat + v1 * v_hat

            quad = [p00, p10, p11, p00, p11, p01]

            # color from vals_2d (if provided)
            if colors is None:
                col = (0.6, 0.6, 0.6)
            else:
                v = colors[j, i]
                col = (v, v, v)
                col = colors_2d[j, i]  # (3,) RGB tuple floats

            for p in quad:
                verts.extend(p.tolist())
                cols.extend(col)

    return verts, cols


# ============================================================
# Isosurface controls (cleaned and documented)
# ============================================================


class OneIsovalue(ipw.HBox):
    """
    Widget holding a single isovalue + its color.
    Appears as a row in the IsovaluesWidget.
    """

    def __init__(self):
        # Numeric field for the isovalue
        self.isovalue_widget = ipw.BoundedFloatText(
            value=1e-3,
            isomin=1e-5,
            isomax=1e-1,
            step=1e-5,
            description="Isovalue",
        )

        # Color picker
        self.color_widget = ipw.ColorPicker(
            concise=False,
            description="Color",
            value="cyan",
        )

        super().__init__([self.isovalue_widget, self.color_widget])

    # Convenience properties
    @property
    def isomin(self):
        return self.isovalue_widget.isomin

    @isomin.setter
    def isomin(self, v):
        self.isovalue_widget.isomin = v

    @property
    def isomax(self):
        return self.isovalue_widget.isomax

    @isomax.setter
    def isomax(self, v):
        self.isovalue_widget.isomax = v

    @property
    def value(self):
        return self.isovalue_widget.value

    @value.setter
    def value(self, v):
        self.isovalue_widget.value = v


class IsovaluesWidget(ipw.VBox):
    """
    Holds multiple isovalues + colors.
    Provides UI to add/remove isovalues and to enforce a global min/max range.
    """

    iso_min = tl.Float(1e-5)
    iso_max = tl.Float(1e-1)

    def __init__(self):
        self.isovalues = ipw.VBox()

        # Range info label
        self.info = ipw.HTML(f"min: {self.iso_min:.1e}, max: {self.iso_max:.1e}")

        # Buttons to add/remove isovalues
        self.add_isovalue_button = ipw.Button(
            description="Add",
            button_style="success",
        )
        self.add_isovalue_button.on_click(self.add_isovalue)

        self.remove_isovalue_button = ipw.Button(
            description="Remove",
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
        """
        Return two lists: isovalues[], colors[].
        """
        pairs = [
            (row.children[0].value, row.children[1].value)
            for row in self.isovalues.children
        ]
        return zip(*pairs) if pairs else ([], [])

    def set_range(self, vmin=-0.001, vmax=0.001):
        """
        Update allowed range for all isovalue widgets.
        """
        self.info.value = f"min: {vmin:.1e}, max: {vmax:.1e}"
        self.iso_min = vmin
        self.iso_max = vmax

        # choose a reasonable default if needed
        default = min(0.001, vmax)
        if abs(vmin) > abs(vmax):
            default = max(-0.001, vmin)

        # enforce new range across existing widgets
        for row in self.isovalues.children:
            widget = row.children[0]
            widget.min = vmin
            widget.max = vmax
            if not (vmin <= widget.value <= vmax):
                widget.value = default

    @tl.observe("iso_min", "iso_max")
    def _observe_range(self, _):
        self.set_range(self.iso_min, self.iso_max)

    def add_isovalue(self, _=None):
        new = OneIsovalue()
        new.isomin = self.iso_min
        new.isomax = self.iso_max
        new.value = min(0.001, self.iso_max)
        self.isovalues.children += (new,)

    def remove_isovalue(self, _=None):
        self.isovalues.children = self.isovalues.children[:-1]


class CubePlaneCut2D(ipw.VBox):
    """
    2D slice viewer for a cube file.

    - Always normalizes the plane normal (no autonorm checkbox).
    - Always updates automatically when any input changes (no Update button).
    """

    cube = tl.Instance(object, allow_none=True)  # expects .data + .ase_atoms

    def __init__(self):
        self._last_r = None
        self._last_vals = None
        self._last_grid_n = None  # utile per sapere m

        # ---- UI fields ----
        self.center_txt = ipw.Text(
            value="0.0 0.0 0.0",
            description="Center (Å)",
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

        # Removed autonorm completely — always normalize

        self.offset_txt = ipw.Text(
            value="0.0",
            description="Offset (Å)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )

        self.width = ipw.FloatText(
            value=10.0,
            description="Width (Å)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )

        self.height = ipw.FloatText(
            value=10.0,
            description="Height (Å)",
            layout=ipw.Layout(width="220px"),
            style={"description_width": "110px"},
        )

        self.res = ipw.IntSlider(
            value=64,
            min=32,
            max=256,
            step=32,
            description="Resolution",
            continuous_update=False,
            layout=ipw.Layout(width="360px"),
        )

        self.show_contours = ipw.Checkbox(value=False, description="Contours")

        self.vmin = ipw.FloatText(
            value=None,
            description="vmin",
            layout=ipw.Layout(width="150px"),
            style={"description_width": "60px"},
        )

        self.vmax = ipw.FloatText(
            value=None,
            description="vmax",
            layout=ipw.Layout(width="150px"),
            style={"description_width": "60px"},
        )

        # Show 2D plot checkbox (not used currently)
        self.show_2d_checkbox = ipw.Checkbox(
            value=True,
            description="Show 2D plot",
        )
        self.show_2d_checkbox.observe(self._on_show_2d_toggled, names="value")

        self.info = ipw.HTML("")
        self.error = ipw.HTML("")

        self._out = ipw.Output(
            layout=ipw.Layout(width="560px", height="560px", border="1px solid #ddd")
        )

        # ---- Layout ----
        super().__init__(
            [
                ipw.HTML("<b>2D plane cut (Cartesian point + normal, PBC)</b>"),
                ipw.HBox([self.center_txt]),
                ipw.HBox([self.normal_txt]),
                ipw.HBox([self.offset_txt]),
                ipw.HBox([self.width, self.height, self.res]),
                ipw.HBox([self.show_contours, self.vmin, self.vmax]),
                self.show_2d_checkbox,
                self.info,
                self.error,
                self._out,
            ]
        )

        # ---- Automatic updates ----
        for w in (
            self.center_txt,
            self.normal_txt,
            self.offset_txt,
            self.width,
            self.height,
            self.res,
            self.show_contours,
            self.vmin,
            self.vmax,
        ):
            w.observe(self._on_any_change, names="value")

        self._sinv = None

    # ------------- Validation / Parsing -------------
    def _vec3_is_valid(self, text: str) -> bool:
        toks = [t for t in re.split(r"[,\s]+", text.strip()) if t]
        if len(toks) != 3:
            return False
        try:
            [float(t) for t in toks]
        except ValueError:
            return False
        return True

    def _inputs_valid(self) -> bool:
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
        except Exception:
            return False
        return True

    def _on_any_change(self, _=None):
        """Auto-update the slice when inputs change."""
        if self._inputs_valid():
            self.error.value = ""
            self.plot_slice()
        else:
            self.error.value = ""
            with self._out:
                self._out.clear_output(wait=True)

    def _on_show_2d_toggled(self, change):
        if change["new"] is False:
            # hide the whole output widget
            self._out.clear_output()
            self._out.layout.display = "none"
        else:
            # show widget again
            self._out.layout.display = ""

            # redraw only if inputs valid
            if self._inputs_valid():
                self.plot_slice()

    @staticmethod
    def _parse_vec3(text):
        arr = np.fromstring(text.replace(",", " "), sep=" ", dtype=float)
        if arr.size != 3:
            raise ValueError(ERR_EXPECT_3_FLOATS)
        return arr

    # ------------- Cube update listener -------------
    @tl.observe("cube")
    def _on_cube(self, _=None):
        if self.cube is None:
            self._sinv = None
            self.info.value = ""
            with self._out:
                self._out.clear_output()
            return

        # Precompute inverse cell for fractional coordinates
        cell = np.array(self.cube.ase_atoms.cell).T
        self._sinv = np.linalg.inv(cell)

        # Auto-set center = mean atomic coordinates
        center = np.mean(self.cube.ase_atoms.get_positions(), axis=0)
        self.center_txt.value = f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"

        # Update stats
        nx, ny, nz = self.cube.data.shape
        dmin, dmax = float(np.min(self.cube.data)), float(np.max(self.cube.data))
        lengths = np.linalg.norm(cell, axis=0)

        self.info.value = (
            f"Grid: {nx}×{ny}×{nz} | data ∈ [{dmin:.3e}, {dmax:.3e}] "
            f"| cell lengths: {lengths[0]:.2f}, {lengths[1]:.2f}, {lengths[2]:.2f} Å"
        )

        # Auto-set width/height
        mean_len = float(np.mean(lengths))
        self.width.value = mean_len
        self.height.value = mean_len

        self.plot_slice()

    # ------------- Geometry construction -------------
    def _build_plane_grid(self):
        center = self._parse_vec3(self.center_txt.value)
        normal = self._parse_vec3(self.normal_txt.value)

        # Always normalize the normal
        n_norm = np.linalg.norm(normal)
        n_hat = normal / n_norm if n_norm > 1e-12 else np.array([0, 0, 1], float)

        # Orthonormal basis
        n_hat, u_hat, v_hat = _orthonormal_basis_from_normal(n_hat)

        offset = float(self.offset_txt.value)
        center_shifted = center + offset * n_hat

        w, h, m = float(self.width.value), float(self.height.value), int(self.res.value)

        # Plane mesh grid
        us = np.linspace(-w / 2, w / 2, m)
        vs = np.linspace(-h / 2, h / 2, m)
        uu, vv = np.meshgrid(us, vs)

        r = (
            center_shifted[None, None, :]
            + uu[..., None] * u_hat
            + vv[..., None] * v_hat
        )

        extent = [-w / 2, w / 2, -h / 2, h / 2]
        return r, extent

    def _cart_to_indexcoords(self, r):
        nx, ny, nz = self.cube.data.shape
        f = (self._sinv @ r.reshape(-1, 3).T).T
        idx = np.empty_like(f)
        idx[:, 0] = f[:, 0] * nx
        idx[:, 1] = f[:, 1] * ny
        idx[:, 2] = f[:, 2] * nz
        return idx.reshape(r.shape)

    # ------------- Plotting -------------
    def plot_slice(self):
        if self.cube is None:
            return

        if not self.show_2d_checkbox.value:
            # Don't display plot area
            self._out.layout.display = "none"
        else:
            # Ensure the plot area is visible when plotting
            self._out.layout.display = ""

        if not self.show_2d_checkbox.value:
            # We still compute slice (needed for 3D plane), but do not plot it
            r, extent = self._build_plane_grid()
            idxcoords = self._cart_to_indexcoords(r)
            coords = np.stack(
                [idxcoords[..., 0], idxcoords[..., 1], idxcoords[..., 2]], axis=0
            )
            vals = map_coordinates(
                self.cube.data, coords, order=1, mode="wrap", prefilter=False
            )

            # store everything for 3D plane
            user_vmin = self.vmin.value
            user_vmax = self.vmax.value
            vmin = (
                float(user_vmin)
                if user_vmin not in (None, "", 0, "0")
                else np.min(vals)
            )
            vmax = (
                float(user_vmax)
                if user_vmax not in (None, "", 0, "0")
                else np.max(vals)
            )

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap("viridis")
            rgb = cmap(norm(vals))[..., :3]

            self._last_r = r
            self._last_vals = vals
            self._last_rgb = rgb
            self._last_grid_n = vals.shape[0]

            return

        # ----------------------------------------------------
        # 1) Compute slice
        # ----------------------------------------------------
        try:
            r, extent = self._build_plane_grid()
            idxcoords = self._cart_to_indexcoords(r)

            coords = np.stack(
                [idxcoords[..., 0], idxcoords[..., 1], idxcoords[..., 2]], axis=0
            )

            vals = map_coordinates(
                self.cube.data, coords, order=1, mode="wrap", prefilter=False
            )

            # --- interpret user inputs ---
            user_vmin = self.vmin.value
            user_vmax = self.vmax.value

            # user 0 or empty → treat as "auto"
            vmin = float(user_vmin) if user_vmin not in (None, "", 0, "0") else None
            vmax = float(user_vmax) if user_vmax not in (None, "", 0, "0") else None

        except Exception:
            self.error.value = "<span style='color:#b00'>Error computing slice</span>"
            with self._out:
                self._out.clear_output()
            return

        # ----------------------------------------------------
        # 2) Determine actual vmin/vmax WITHOUT drawing into widget output
        #    (this avoids breaking layout!)
        # ----------------------------------------------------
        tmp_fig, tmp_ax = plt.subplots()
        tmp_im = tmp_ax.imshow(vals, vmin=vmin, vmax=vmax)
        real_vmin = tmp_im.norm.vmin
        real_vmax = tmp_im.norm.vmax
        plt.close(tmp_fig)  # IMPORTANT: avoid ghost figures

        # Update widgets ONLY if user left them empty or zero
        if user_vmin in (None, "", 0, "0"):
            self.vmin.value = real_vmin
        if user_vmax in (None, "", 0, "0"):
            self.vmax.value = real_vmax

        # ----------------------------------------------------
        # 3) Now do the REAL drawing
        # ----------------------------------------------------
        with self._out:
            self._out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

            im = ax.imshow(
                vals,
                origin="lower",
                extent=extent,
                aspect="equal",
                vmin=real_vmin,
                vmax=real_vmax,
            )

            ax.set_xlabel("u (Å)")
            ax.set_ylabel("v (Å)")
            fig.colorbar(im, ax=ax)

            if self.show_contours.value:
                ax.contour(
                    vals, levels=8, origin="lower", extent=extent, linewidths=0.8
                )

            # ------------------------------------------------
            # Save data for 3D colored plane
            # ------------------------------------------------
            norm = mcolors.Normalize(vmin=real_vmin, vmax=real_vmax)
            cmap = cm.get_cmap("viridis")
            rgb = cmap(norm(vals))[..., :3]

            self._last_r = r
            self._last_vals = vals
            self._last_rgb = rgb
            self._last_grid_n = vals.shape[0]

            plt.show()

    # ------------- Parameters for 3D plane sync -------------
    def current_plane_params(self):
        center = self._parse_vec3(self.center_txt.value)
        normal = self._parse_vec3(self.normal_txt.value)

        # Normalization always on
        n_norm = np.linalg.norm(normal)
        n_hat = normal / n_norm if n_norm > 1e-12 else np.array([0, 0, 1], float)

        offset = float(self.offset_txt.value)
        w, h = float(self.width.value), float(self.height.value)

        return center, n_hat, offset, w, h


# ----------------------------- 3D viewer ----------------------------------
class CubeArrayData3dViewerWidget(ipw.VBox):
    """Structure + cube + isosurfaces + slicing-plane overlay as a mesh component.
    We keep at most ONE plane component. Updating the plane removes the old component
    and creates a new one. The show checkbox toggles visibility with .show()/.hide().
    """

    cube = tl.Instance(object, allow_none=True)

    def __init__(self, **kwargs):
        self.structure = None
        self.viewer = nglview.NGLWidget()
        self.isovalues = IsovaluesWidget()

        self.show_isosurfaes_button = ipw.Button(
            description="Show isosurfaces",
            layout={"width": "initial"},
            button_style="success",
        )
        self.show_isosurfaes_button.on_click(lambda _: self.update_plot())

        self.show_plane_checkbox = ipw.Checkbox(
            value=True, description="Show slicing plane in 3D"
        )

        # handle to the shape component that holds our mesh
        self._plane_component = None

        super().__init__(
            [
                self.viewer,
                self.isovalues,
                ipw.HBox([self.show_isosurfaes_button, self.show_plane_checkbox]),
            ],
            **kwargs,
        )

        # external callback (assigned by HandleCubeFiles) to (re)build the plane
        self._external_plane_sync_cb = None

        # connect checkbox
        self.show_plane_checkbox.observe(self._on_plane_toggle, names="value")

    # ---------------- lifecycle ----------------
    @tl.observe("cube")
    def on_observe_cube(self, _=None):
        if self.cube is None:
            return
        self.isovalues.set_range(
            vmin=float(np.min(self.cube.data)),
            vmax=float(np.max(self.cube.data)),
        )
        self.structure = self.cube.ase_atoms
        self.update_plot()

    def update_plot(self):
        # remove prior structure/cube components
        for comp_attr in ("_structure_component", "_cube_component"):
            comp = getattr(self, comp_attr, None)
            if comp is not None:
                try:
                    self.viewer.remove_component(comp.id)
                except Exception:
                    logger.debug("remove_component failed", exc_info=True)
                setattr(self, comp_attr, None)

        # do not touch the plane here; it is managed by the sync callback
        if self.structure is None or self.cube is None:
            return

        self._structure_component = self.viewer.add_component(
            nglview.ASEStructure(self.structure)
        )
        with tempfile.NamedTemporaryFile(mode="w") as tempf:
            ase.io.cube.write_cube(tempf, self.structure, self.cube.data)
            self._cube_component = self.viewer.add_component(tempf.name, ext="cube")
            self._cube_component.clear()

        # isosurfaces
        try:
            isovals, colors = self.isovalues.return_values()
            self.set_cube_isosurf(isovals, colors)
        except ValueError:
            # no isovalues defined yet
            pass
        except Exception:
            logger.debug("set_cube_isosurf failed", exc_info=True)

        # rebuild plane last
        if self._external_plane_sync_cb:
            try:
                self._external_plane_sync_cb()
            except Exception:
                logger.debug("external plane sync failed", exc_info=True)

    # ---------------- isosurfaces ----------------
    def set_cube_isosurf(self, isovals, colors):
        if getattr(self, "_cube_component", None) is None:
            return
        self._cube_component.clear()
        for isov, col in zip(isovals, colors):
            try:
                self._cube_component.add_surface(
                    color=col, isolevelType="value", isolevel=isov
                )
            except Exception:
                logger.debug("add_surface failed", exc_info=True)

    # ---------------- plane mesh (component-based) ----------------
    def _capture_newest_component_as_plane(self, before_attrs: set[str]):
        """Capture the newest component (compared to 'before_attrs') as the plane component."""
        after = {n for n in dir(self.viewer) if n.startswith("component_")}
        new_attrs = sorted(after - before_attrs)
        if new_attrs:
            try:
                self._plane_component = getattr(self.viewer, new_attrs[-1])
            except Exception:
                logger.debug("capturing plane component failed", exc_info=True)
        elif self._plane_component is None:
            try:
                comps = sorted(
                    [n for n in dir(self.viewer) if n.startswith("component_")]
                )
                if comps:
                    self._plane_component = getattr(self.viewer, comps[-1])
            except Exception:
                logger.debug(
                    "fallback capture of plane component failed", exc_info=True
                )

    def _remove_plane_component(self):
        """Remove the current plane component from the viewer (if any)."""
        if self._plane_component is not None:
            try:
                self.viewer.remove_component(self._plane_component.id)
            except Exception:
                logger.debug("remove plane component failed", exc_info=True)
            self._plane_component = None

    def _on_plane_toggle(self, change):
        """Checkbox toggles visibility only (no geometry changes)."""
        comp = self._plane_component
        if comp is None:
            return
        try:
            if change["new"] is False:
                comp.hide()
            else:
                comp.show()
        except Exception:
            logger.debug("plane show/hide failed", exc_info=True)

    def update_plane_mesh(self, *, center, normal, width, height, offset=0.0):

        if not self.show_plane_checkbox.value:
            self._remove_plane_component()
            return

        self._remove_plane_component()

        before = {n for n in dir(self.viewer) if n.startswith("component_")}

        getattr(self.slice2d, "_last_vals", None)
        rgb = getattr(self.slice2d, "_last_rgb", None)

        try:
            # ---- Build mesh (vertices + colors) ----
            verts, cols = make_plane_mesh(
                center, normal, width, height, offset, colors_2d=rgb
            )

            # ---- Add mesh to viewer ----
            self.viewer.shape.add_mesh(verts, cols)

        except Exception as e:
            print("Plane mesh update failed:", e)
            return

        # Capture plane component only if everything succeeded
        self._capture_newest_component_as_plane(before_attrs=before)

    # convenience used by HandleCubeFiles when inputs are invalid or node cleared
    def hide_and_remove_plane(self):
        self._remove_plane_component()


# ----------------------- render helper (AiiDA) -----------------------------
@engine.calcfunction
def render_cube_file(settings, remote_folder):
    fd = orm.FolderData()
    for name, value in settings.items():
        full_name = f"{name}.{value['format']}"
        with open(full_name, "rb") as file_h:
            fd.put_object_from_bytes(file_h.read(), path=full_name)
    return {"rendered_images": fd}


# ------------------------ Main handle widget -------------------------------
class HandleCubeFiles(ipw.VBox):
    node_pk = tl.Int(None, allow_none=True)
    nel_up = tl.Int(None, allow_none=True)
    nel_dw = tl.Int(None, allow_none=True)
    uks = tl.Bool(False)
    render_instructions = tl.Dict({}, allow_none=True)
    remote_data_uuid = tl.Unicode(allow_none=True)

    def __init__(self):
        self.node = None

        # selector first so observers can safely reference it
        self.cube_selector = ipw.Select(
            options=[],
            description="Cube files:",
            layout=ipw.Layout(width="400px"),
            style={"description_width": "initial"},
        )
        self.cube_selector.observe(self.show_selected_cube, names="value")

        # 3D viewer + 2D slicer
        self._viewer = CubeArrayData3dViewerWidget(layout=ipw.Layout(width="550px"))
        self.slice2d = CubePlaneCut2D()
        self._viewer.slice2d = self.slice2d

        # Keep the cubes in sync (3D viewer ↔ 2D slicer)
        self._cube_sync = tl.dlink((self._viewer, "cube"), (self.slice2d, "cube"))

        # slicer → viewer plane sync
        def _sync_plane(_=None):
            try:
                if self.slice2d.cube is None or not self.slice2d._inputs_valid():
                    self._viewer.hide_and_remove_plane()
                    return
                center, normal, offset, w, h = self.slice2d.current_plane_params()
                self._viewer.update_plane_mesh(
                    center=center,
                    normal=normal,
                    width=w,
                    height=h,
                    offset=offset,
                )
            except Exception:
                logger.debug("sync plane failed", exc_info=True)

        # re-sync on slicer changes
        for w in (
            self.slice2d.center_txt,
            self.slice2d.normal_txt,
            # self.slice2d.autonorm,
            self.slice2d.offset_txt,
            self.slice2d.width,
            self.slice2d.height,
            self.slice2d.res,
        ):
            w.observe(_sync_plane, names="value")

        # Allow the viewer to call us after it rebuilds components
        self._viewer._external_plane_sync_cb = _sync_plane
        self._sync_plane = _sync_plane

        # Camera orientation snapshot
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

        # Render controls
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

        # Layout
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
        if not self.cube_selector.value or self.node is None:
            return
        cube_obj = Cube.from_content(
            self.node.get_object_content(f"out_cubes/{self.cube_selector.value}")
        )
        self._viewer.cube = cube_obj
        self._sync_plane()  # draw current plane immediately

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
        self.node = calc.outputs.retrieved
        try:
            self.remote_data_uuid = calc.inputs.nodes.remote_previous_job.uuid
        except Exception:
            logger.debug("fallback to outputs.remote_folder", exc_info=True)
            self.remote_data_uuid = calc.outputs.remote_folder.uuid

        orb_options = []
        pattern = re.compile(r"WFN_(\d+)_([12])\-")  # orbital index + spin

        for name in self.node.list_object_names("out_cubes"):
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
        except Exception:
            logger.debug("no isovalues to capture", exc_info=True)

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
            self.remote_data_uuid = None
            try:
                self._viewer.hide_and_remove_plane()
            except Exception:
                logger.debug("hide_and_remove_plane failed", exc_info=True)

    @tl.observe("render_instructions")
    def _observe_render_instructions(self, _=None):
        if hasattr(self, "render_instructions_widget"):
            self.render_instructions_widget.value = toml.dumps(self.render_instructions)

    def render_all(self, _=None):
        if not self.remote_data_uuid:
            self.error_message.value = "No remote_data_uuid set."
            return
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
