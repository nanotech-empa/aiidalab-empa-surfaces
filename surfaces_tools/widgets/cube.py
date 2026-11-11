"""Surfaces provenance and submission adapters for the maintained shared viewer."""

from fnmatch import fnmatch
from html import escape
import re
from pathlib import Path, PurePosixPath

import aiidalab_widgets_base as awb
import ipywidgets as ipw
import numpy as np
import toml
import traitlets as tl
from aiida import orm, plugins
from aiida.common.exceptions import NotExistent
from cubehandler import Cube
from cubehandler.cube import ANG_TO_BOHR
from PIL import ImageColor
from aiidalab_widgets_empa.cube import (
    CubeViewer,
    CubeArrayData3dViewerWidget,
    CubePlaneCut2D,
    OneIsovalue,
    IsovaluesWidget,
    make_plane_mesh,
)
from aiidalab_widgets_empa.cube_views import SavedCubeViews
from ..utils import cube_render, cube_results

__all__ = [
    "CubeViewer",
    "CubeArrayData3dViewerWidget",
    "CubePlaneCut2D",
    "OneIsovalue",
    "IsovaluesWidget",
    "make_plane_mesh",
    "HandleCubeFiles",
    "DisplayRenderedCubes",
    "render_cube_file",
]
CubeHandlerCalculation = plugins.CalculationFactory("nanotech_empa.cubehandler")
ISO_OPTION_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*:\s*(#[0-9A-Fa-f]{6})\s*$"
)


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


def _load_cube_preview(calc_uuid, cube_name):
    """Load retained data, deriving a labelled preview only from qualified source."""
    calc = orm.load_node(calc_uuid)
    preview = Cube.from_content(
        calc.outputs.retrieved.get_object_content(f"out_cubes/{cube_name}")
    )
    # Follow the recorded shrink input, never an unrelated cube with a similar name.
    matches = []
    for step in calc.inputs.parameters.get_dict().get("steps", []):
        options = step.get("options", {})
        prefix = options.get("prefix", "reduced_")
        if step.get("command") != "shrink" or not cube_name.startswith(prefix):
            continue
        original_name = cube_name[len(prefix) :]
        for arg in step.get("args", []):
            path = PurePosixPath(arg)
            if (
                not path.is_absolute()
                and ".." not in path.parts
                and fnmatch(original_name, path.name)
            ):
                matches.append(
                    (path.parent / original_name, options.get("method", "slicing"))
                )
    if not matches:
        return preview, ""
    unavailable = (
        "Retained reduced preview. Its grid could not be checked against the full "
        "cube. Spatial alignment is unverified."
    )
    if len(matches) != 1 or calc.computer.transport_type != "core.local":
        return preview, unavailable
    source_path, method = matches[0]
    source_path = Path(calc.get_remote_workdir()) / str(source_path)
    if not source_path.is_file():
        return preview, unavailable
    source = Cube.from_file(source_path, read_data=False)
    if (
        not np.array_equal(source.ase_atoms.numbers, preview.ase_atoms.numbers)
        or not np.allclose(
            source.ase_atoms.positions, preview.ase_atoms.positions, atol=1e-5, rtol=0
        )
        or not np.allclose(source.origin, preview.origin, atol=1e-6, rtol=0)
    ):
        raise ValueError(
            "Full and reduced cube structures or origins differ. Preview correction was refused."
        )
    shape = np.asarray(preview.data.shape)
    source_shape = np.asarray(source.cell_n)
    cell_matches = np.allclose(source.cell, preview.cell, atol=2e-4, rtol=0)
    if cell_matches and np.all(source_shape % shape == 0):
        return preview, "Retained preview. Cell and sampling match the full cube grid."
    if method != "slicing":
        raise ValueError(
            "This reduced grid needs correction. Automatic reconstruction supports slicing only."
        )
    # Reject a full cube whose cell already disagrees with its calculation model.
    caller = calc.caller
    if caller is None or "structure" not in caller.inputs:
        raise ValueError(
            "Cannot qualify the full cube against its source calculation cell."
        )
    if not np.allclose(
        source.cell / ANG_TO_BOHR, caller.inputs.structure.cell, atol=2e-4, rtol=0
    ):
        raise ValueError(
            "The full cube cell differs from the source calculation. Preview correction was refused."
        )
    source = Cube.from_file(source_path)
    # Qualify the retained values against the known legacy slicing algorithm.
    legacy_voxels = source.cell / source_shape / ANG_TO_BOHR
    strides = np.rint(1.0 / (2 * np.linalg.norm(legacy_voxels, axis=1))).astype(int)
    if np.any(strides < 1):
        raise ValueError(
            "The retained preview does not match the supported legacy slicing path."
        )
    samples = source.data[:: strides[0], :: strides[1], :: strides[2]]
    tolerance = abs(preview.scaling_factor) * 0.000501
    if samples.shape != preview.data.shape or not np.allclose(
        samples, preview.data, atol=tolerance, rtol=1e-6
    ):
        raise ValueError(
            "Retained preview values do not match the full cube. Preview correction was refused."
        )
    legacy_cell = source.cell / shape * shape[:, None]
    if not cell_matches and not np.allclose(
        legacy_cell, preview.cell, atol=2e-4, rtol=0
    ):
        raise ValueError("The reduced cell does not match the known legacy grid error.")
    source.reduce_data_density_slicing()
    return source, (
        "Derived preview reconstructed from the original full cube with cell-aligned "
        "sampling. Stored cubes are unchanged. Use the full grid for density integrals."
    )


class HandleCubeFiles(ipw.VBox):
    """AiiDA navigation/persistence around the maintained, data-only viewer."""

    node_pk = tl.Int(None, allow_none=True)
    nel_up = tl.Int(None, allow_none=True)
    nel_dw = tl.Int(None, allow_none=True)
    uks = tl.Bool(False)
    remote_data_uuid = tl.Unicode(None, allow_none=True)
    render_source_remote_uuid = tl.Unicode(None, allow_none=True)

    def __init__(self):
        self._helper_uuid = None
        self.retrieved_uuid = None
        self._cube_cache = {}
        self._reviewed_render = None
        self._model = None
        self._shared_viewer = None
        self._view_observers = []
        self._viewer_box = ipw.VBox()
        self.error_message = ipw.HTML()
        self.model_message = ipw.HTML()
        self.render_review = ipw.HTML()
        self.cube_selector = ipw.Select(
            options=[], description="Cube files:", layout=ipw.Layout(width="90%")
        )
        self.cube_selector.observe(self.show_selected_cube, names="value")
        self.cubehandler_code_widget = awb.ComputationalResourcesWidget(
            description="Cubehandler code:",
            default_calc_job_plugin="nanotech_empa.cubehandler",
        )
        self.cubehandler_code_widget.observe(self._invalidate_render, names="value")
        self.render_submit = ipw.Button(
            description="Submit reviewed render",
            disabled=True,
            layout=ipw.Layout(width="auto"),
        )
        self.render_submit.on_click(self._submit_render)
        self.saved = SavedCubeViews(
            save=self._save_view,
            list_views=self._list_views,
            load=self._load_view,
            remove=self._remove_view,
            render=self._review_saved_view,
        )
        self.saved.children[0].value = (
            "<h4>Saved visualizations</h4><p>Save this cube's camera, isosurfaces, "
            "slice and rendering settings. Review one saved view at a time for rendering.</p>"
        )
        self.saved.render_button.description = "Review selected view"
        super().__init__(
            [
                self.cube_selector,
                self.model_message,
                self.error_message,
                self._viewer_box,
                self.saved,
                self.cubehandler_code_widget,
                self.render_review,
                self.render_submit,
            ]
        )

    @property
    def _viewer(self):
        return self._shared_viewer.viewer3d if self._shared_viewer is not None else None

    @property
    def slice2d(self):
        return self._shared_viewer.slice2d if self._shared_viewer is not None else None

    def _invalidate_render(self, _=None):
        self._reviewed_render = None
        self.render_submit.disabled = True
        self.render_review.value = ""

    def _unbind_view_observers(self):
        for widget, names in self._view_observers:
            widget.unobserve(self._view_changed, names=names)
        self._view_observers = []

    def _bind_view_observers(self):
        self._unbind_view_observers()
        seen = set()

        def visit(widget):
            if id(widget) in seen:
                return
            seen.add(id(widget))
            traits = widget.traits()
            names = []
            if "value" in traits and not isinstance(widget, (ipw.HTML, ipw.HTMLMath)):
                names.append("value")
            if "_camera_orientation" in traits:
                names.append("_camera_orientation")
            if "children" in traits:
                names.append("children")
            if names:
                widget.observe(self._view_changed, names=names)
                self._view_observers.append((widget, names))
            for child in getattr(widget, "children", ()):
                visit(child)

        if self._shared_viewer is not None:
            visit(self._shared_viewer)

    def _view_changed(self, change):
        self._invalidate_render()
        if change["name"] == "children":
            self._bind_view_observers()

    def _clear_source(self):
        self._invalidate_render()
        self._unbind_view_observers()
        self._helper_uuid = self.retrieved_uuid = None
        self.remote_data_uuid = self.render_source_remote_uuid = None
        self._cube_cache = {}
        self._model = None
        self.cube_selector.options = []
        self.model_message.value = self.error_message.value = ""
        if self._shared_viewer is not None:
            self._shared_viewer.close()
        self._shared_viewer = None
        self._viewer_box.children = ()
        self.saved.refresh()

    @tl.observe("node_pk")
    def _observe_node_pk(self, _=None):
        self.select_calculation()

    def select_calculation(self, _=None):
        self._clear_source()
        if not self.node_pk:
            return
        try:
            node = orm.load_node(self.node_pk)
            if node.process_type != cube_results.CUBE_TYPE or not node.is_finished_ok:
                raise ValueError("Select a completed cubehelper result.")
            names = node.outputs.retrieved.list_object_names("out_cubes")
            model = cube_results.cube_boundary(node.uuid)
            if model["periodic"] not in ("XYZ", "NONE"):
                raise ValueError(
                    f"Effective {model['dimension']}D {model['periodic']} model. "
                    "This viewer currently supports XYZ or NONE sampling. "
                    "Partial-axis visualization is unavailable."
                )
            self._helper_uuid = node.uuid
            self.retrieved_uuid = node.outputs.retrieved.uuid
            remote = getattr(node.outputs, "remote_folder", None)
            self.render_source_remote_uuid = remote.uuid if remote is not None else None
            folders = getattr(node.inputs, "parent_folders", {})
            original = folders.get("folder1", remote)
            self.remote_data_uuid = original.uuid if original is not None else None
            self._model = model
            code_uuid = node.inputs.code.uuid
            self.cubehandler_code_widget.value = (
                code_uuid
                if code_uuid in self.cubehandler_code_widget.codes.values()
                else None
            )
            self._shared_viewer = CubeViewer(
                periodic=model["periodic"] == "XYZ", on_render=self._review_render
            )
            self._shared_viewer.render_button.description = "Review render"
            self._viewer_box.children = (self._shared_viewer,)
            self._bind_view_observers()
            self.model_message.value = (
                f"<p>Effective calculation model: {model['dimension']}D {model['periodic']}. "
                f"Sampling follows {escape(model['source'])}. "
                "The preview may be reduced. Rendering uses the original full grid.</p>"
            )
            options = []
            for name in sorted(names):
                if not name.endswith(".cube"):
                    continue
                label = name
                match = re.search(r"WFN_(\d+)_([12])-", name)
                if match and self.nel_up is not None:
                    index, spin = map(int, match.groups())
                    homo = self.nel_dw if self.uks and spin == 2 else self.nel_up
                    if homo is not None:
                        offset = index - homo
                        label = ("UP-" if spin == 1 else "DW-") if self.uks else ""
                        label += (
                            f"HOMO {offset}"
                            if offset < 0
                            else "HOMO"
                            if offset == 0
                            else "LUMO"
                            if offset == 1
                            else f"LUMO +{offset - 1}"
                        )
                elif "ChargeDiff" in name:
                    label = "ChargeDifference"
                elif "ELECTRON" in name:
                    label = "CHARGE-DENSITY"
                elif "SPIN" in name:
                    label = "SPIN-DENSITY"
                options.append((label, name))
            self.cube_selector.options = options
            if not options:
                raise ValueError("This completed helper has no retrieved cube files.")
            self.saved.refresh()
        except (
            NotExistent,
            AttributeError,
            KeyError,
            ValueError,
            FileNotFoundError,
            NotADirectoryError,
        ) as exc:
            self._clear_source()
            self.error_message.value = (
                f"<p style='color:#a02020'>{escape(str(exc))}</p>"
            )

    def show_selected_cube(self, _=None):
        self._invalidate_render()
        if not self.cube_selector.value or not self._helper_uuid:
            return
        name = self.cube_selector.value
        try:
            if name not in self._cube_cache:
                self._cube_cache[name] = _load_cube_preview(self._helper_uuid, name)
            preview, notice = self._cube_cache[name]
            self._shared_viewer.set_cube(preview)
            self.error_message.value = f"<p>{escape(notice)}</p>"
        except (AttributeError, ValueError, FileNotFoundError) as exc:
            self._shared_viewer.set_cube(None)
            self.error_message.value = (
                f"<p style='color:#a02020'>{escape(str(exc))}</p>"
            )

    def _review_render(self, settings):
        self._invalidate_render()
        try:
            plan = cube_render.prepare_render(
                self._helper_uuid,
                self.cube_selector.value,
                self.cubehandler_code_widget.value,
                settings,
            )
        except (NotExistent, AttributeError, ValueError, OSError) as exc:
            self.render_review.value = (
                f"<p style='color:#a02020'>{escape(str(exc))}</p>"
            )
            return
        self._reviewed_render = plan
        source = plan["source"]
        self.render_review.value = (
            f"<p>Full-grid render: {escape(source['cube_name'])}, "
            f"{escape(str(source['full_shape']))}. Effective "
            f"{source['model']['dimension']}D {source['model']['periodic']}. "
            "One local CPU process, one thread, 600 seconds. "
            "The current camera, signed isosurfaces, visible plane, colors and lighting are included.</p>"
        )
        self.render_submit.disabled = False

    def _submit_render(self, _=None):
        self.render_submit.disabled = True
        try:
            plan = self._reviewed_render
            if plan is None:
                raise ValueError("Review the current render before submitting.")
            if (
                plan["source"]["helper_uuid"] != self._helper_uuid
                or plan["source"]["cube_name"] != self.cube_selector.value
                or plan["code_uuid"] != self.cubehandler_code_widget.value
                or plan["settings"]
                != cube_render.validate_settings(
                    self._shared_viewer.render_settings(),
                    periodic=self._model["periodic"] == "XYZ",
                )
            ):
                raise ValueError("The view or source changed. Review the render again.")
            self._reviewed_render = None
            result = cube_render.submit_reviewed_render(plan)
            action = "Existing render" if result["reused"] else "Render submitted"
            self.render_review.value = f"<p>{action}: {escape(result['uuid'])}. Refresh renders after completion.</p>"
            original = self.remote_data_uuid
            self.remote_data_uuid = None
            self.remote_data_uuid = original
        except Exception as exc:
            self._reviewed_render = None
            self.render_review.value = (
                f"<p style='color:#a02020'>{escape(str(exc))}</p>"
            )

    def _list_views(self):
        return cube_render.saved_views(self._helper_uuid) if self._helper_uuid else []

    def _save_view(self, name, identifier=None):
        if not self._helper_uuid or not self.cube_selector.value:
            raise ValueError("Load a cube before saving its visualization.")
        return cube_render.save_view(
            self._helper_uuid,
            self.cube_selector.value,
            name,
            self._shared_viewer.view_settings(),
            identifier,
        )

    def _load_view(self, identifier):
        self._invalidate_render()
        record = cube_render.saved_view(self._helper_uuid, identifier)
        if record["helper_uuid"] != self._helper_uuid:
            raise ValueError("The saved view belongs to another cube result.")
        self.cube_selector.value = record["cube_name"]
        self.show_selected_cube()
        self._shared_viewer.apply_view_settings(record["visualization"])

    def _remove_view(self, identifier):
        cube_render.remove_view(self._helper_uuid, identifier)

    def _review_saved_view(self, identifiers):
        if len(identifiers) != 1:
            raise ValueError("Select one saved view. Local renders run one at a time.")
        self._load_view(identifiers[0])
        self._review_render(self._shared_viewer.render_settings())

    def close(self):
        if hasattr(self, "_view_observers"):
            self._unbind_view_observers()
        if getattr(self, "_shared_viewer", None) is not None:
            self._shared_viewer.close()
            self._shared_viewer = None
        super().close()


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
