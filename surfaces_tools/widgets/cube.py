import copy
import re
import tempfile

import ase.io.cube
import ipywidgets as ipw
import nglview
import numpy as np
import toml
import traitlets as tl
from aiida import engine, orm, plugins
from aiida_shell import ShellJob
from cubehandler import Cube

Cp2kOrbitalsWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.orbitals")
Cp2kGeoOptWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.geo_opt")
Cp2kFragmentSeparationWorkChain = plugins.WorkflowFactory(
    "nanotech_empa.cp2k.fragment_separation"
)


class OneIsovalue(ipw.HBox):
    def __init__(self, structure=None):
        self.isovalue_widget = ipw.BoundedFloatText(
            value=1e-3,
            min=1e-5,
            max=1e-1,
            step=1e-5,
            description="Isovalue",
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

        # Add constraint button.
        self.info = ipw.HTML(f"min: {self.iso_min:.1e}, max: {self.iso_max:.1e}")
        self.add_isovalue_button = ipw.Button(
            description="Add isovalue",
            layout={"width": "initial"},
            button_style="success",
        )
        self.add_isovalue_button.on_click(self.add_isovalue)

        # Remove constraint button.
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
        if np.abs(vmin) > np.abs(vmax):
            default = max(-0.001, vmin)

        for isovalue in self.isovalues.children:
            isovalue.children[0].min = vmin
            isovalue.children[0].max = vmax
            isovalue.children[0].value = default

    @tl.observe("iso_min", "iso_max")
    def _observe_range(self, _=None):
        self.set_range(vmin=self.iso_min, vmax=self.iso_max)

    def add_isovalue(self, b=None):
        new_isovalue = OneIsovalue()
        new_isovalue.min = self.iso_min
        new_isovalue.max = self.iso_max
        new_isovalue.value = min(0.001, self.iso_max)
        if self.iso_max < 1.0e-8:
            new_isovalue.value = max(-0.001, self.iso_min)
        self.isovalues.children += (new_isovalue,)

    def remove_isovalue(self, b=None):
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
            isovalues, colors = self.isovalues.return_values()
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


@engine.calcfunction
def render_cube_file(settings, remote_folder):
    fd = orm.FolderData()
    for name, value in settings.items():
        full_name = f"{name}.{value['format']}"
        file = open(full_name, "rb")
        image = file.read()
        fd.put_object_from_bytes(image, path=full_name)
    return {"rendered_images": fd}


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

        self.render_all_button = ipw.Button(description="Render all")
        self.render_all_button.on_click(self.render_all)

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
            ShellJob,
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
        try:
            self.remote_data_uuid = calc.inputs.nodes.remote_previous_job.uuid
        except Exception:
            self.remote_data_uuid = calc.outputs.remote_folder.uuid

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
        else:
            self.error_message.value = ""
        new_dict = copy.deepcopy(self.render_instructions)
        new_dict[self.render_name.value] = {
            "cube": self.cube_selector.value,
            "camera_orientation": self._viewer.viewer._camera_orientation,
            "isovalue": self._viewer.orb_isosurf_slider.value,
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
        # Format the render instructions using toml and display them in the widget
        self.render_instructions_widget.value = toml.dumps(self.render_instructions)

    def render_all(self, _=None):
        render_cube_file(self.render_instructions, orm.load_node(self.remote_data_uuid))


class DisplayRenderedCubes(ipw.HBox):
    remote_data_uuid = tl.Unicode(allow_none=True)

    def __init__(self):
        self.rendered_images_widget = ipw.Select(
            description="Select an image:",
            layout=ipw.Layout(width="350px"),
            style={"description_width": "initial"},
        )
        self.rendered_images_widget.observe(self.show_image)
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
                self.rendered_images_widget,
                ipw.VBox([accordion, self.image]),
            ]
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
