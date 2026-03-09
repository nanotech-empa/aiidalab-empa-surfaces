import copy
import re
import tempfile
from pathlib import PurePosixPath

import aiidalab_widgets_base as awb
import ase.io.cube
import ipywidgets as ipw
import nglview
import numpy as np
import toml
import traitlets as tl
from aiida import orm, plugins
from cubehandler import Cube
from PIL import ImageColor

CubeHandlerCalculation = plugins.CalculationFactory("nanotech_empa.cubehandler")
Cp2kOrbitalsWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.orbitals")
Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")
Cp2kFragmentSeparationWorkChain = plugins.WorkflowFactory(
    "nanotech_empa.cp2k.fragment_separation"
)
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


class CubeArrayData3dViewerWidget(ipw.VBox):
    """Widget to View 3-dimensional AiiDA ArrayData object in 3D."""

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

        super().__init__(
            [self.viewer, self.isovalues, self.show_isosurfaes_button], **kwargs
        )

    @tl.observe("cube")
    def on_observe_cube(self, _=None):
        """Update object attributes when cube trait is modified."""
        self.isovalues.set_range(
            vmin=np.min(self.cube.data), vmax=np.max(self.cube.data)
        )
        self.structure = self.cube.ase_atoms
        self.update_plot()

    def update_plot(self):
        """Update the 3D plot."""
        while hasattr(self.viewer, "component_0"):
            self.viewer.component_0.clear_representations()
            self.viewer.remove_component(self.viewer.component_0.id)
        self.setup_cube_plot()
        try:
            isovalues, colors = self.isovalues.return_isovalues_and_colors()
            self.set_cube_isosurf(
                isovalues,  # [-0.001, 0.001],
                colors,  # ["red", "blue"],
            )
        except ValueError:
            pass

    def setup_cube_plot(self):
        """Setup cube plot."""
        self.viewer.add_component(nglview.ASEStructure(self.structure))
        with tempfile.NamedTemporaryFile(mode="w") as tempf:
            ase.io.cube.write_cube(tempf, self.structure, self.cube.data)
            c_2 = self.viewer.add_component(tempf.name, ext="cube")
            c_2.clear()

    def set_cube_isosurf(self, isovals, colors):
        """Set cube isosurface."""
        if hasattr(self.viewer, "component_1"):
            c_2 = self.viewer.component_1
            c_2.clear()
            for isov, col in zip(isovals, colors):
                c_2.add_surface(color=col, isolevelType="value", isolevel=isov)


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
        self._viewer = CubeArrayData3dViewerWidget(layout=ipw.Layout(width="550px"))
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
        super().__init__(
            [
                # self.select_calc_widget,
                ipw.HBox(
                    [
                        self._viewer,
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
                                self.cubehandler_code_widget,
                                self.render_submit,
                            ]
                        ),
                    ]
                ),
            ]
        )

    def show_selected_cube(self, _=None):

        if not self.cube_selector.value:
            return
        self._viewer.cube = Cube.from_content(
            self.node.get_object_content(f"out_cubes/{self.cube_selector.value}")
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
        label = None
        pattern = re.compile(
            r"WFN_(\d+)_([12])\-"
        )  # captures orbital number and spin (1 or 2)

        for name in self.node.list_object_names("out_cubes"):
            if "WFN" in name:

                m = pattern.search(name)
                if not m:
                    continue  # skip unexpected names

                n_orb = int(m.group(1))  # e.g. 00002 -> 2
                spin = "UP-"
                if int(m.group(2)) == 2:
                    spin = "DW-"

                # choose the correct HOMO index to compare against
                if self.uks:
                    n_morb = self.nel_up if spin == "UP" else self.nel_dw
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
            self.cube_selector.options = []
            self._viewer.cube = None
            self.render_instructions = {}
            self.remote_data_uuid = None
            self.render_source_remote_uuid = None

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
