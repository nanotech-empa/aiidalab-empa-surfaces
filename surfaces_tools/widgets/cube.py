import copy
import tempfile

import ase.io.cube
import ipywidgets as ipw
import nglview
import toml
import traitlets as tl
from aiida import engine, orm, plugins
from aiida_shell import ShellJob
from cubehandler import Cube

Cp2kOrbitalsWorkChain = plugins.WorkflowFactory("nanotech_empa.cp2k.orbitals")


class CubeArrayData3dViewerWidget(ipw.VBox):
    """Widget to View 3-dimensional AiiDA ArrayData object in 3D."""

    cube = tl.Instance(Cube, allow_none=True)

    def __init__(self, **kwargs):

        self.structure = None
        self.viewer = nglview.NGLWidget()
        self.orb_isosurf_slider = ipw.FloatSlider(
            continuous_update=False,
            value=1e-3,
            min=1e-3,
            max=1e-1,
            step=1e-4,
            description="Isovalue",
            readout_format=".1e",
        )
        self.orb_isosurf_slider.observe(
            lambda c: self.set_cube_isosurf([c["new"], -c["new"]], ["red", "blue"]),
            names="value",
        )
        super().__init__([self.viewer, self.orb_isosurf_slider], **kwargs)

    @tl.observe("cube")
    def on_observe_cube(self, _=None):
        """Update object attributes when cube trait is modified."""
        self.structure = self.cube.ase_atoms
        self.update_plot()

    def update_plot(self):
        """Update the 3D plot."""
        while hasattr(self.viewer, "component_0"):
            self.viewer.component_0.clear_representations()
            self.viewer.remove_component(self.viewer.component_0.id)
        self.setup_cube_plot()
        self.set_cube_isosurf(
            [
                self.orb_isosurf_slider.value,
                -self.orb_isosurf_slider.value,
            ],
            ["red", "blue"],
        )

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
        self.select_calc_widget = ipw.Dropdown(
            description="Calculation:", options=self.get_calcs()
        )

        tl.dlink(
            (self, "node_pk"),
            (self.select_calc_widget, "value"),
            transform=lambda x: orm.load_node(x).uuid if x else None,
        )
        self.select_calc_widget.observe(self.select_calculation)
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

        self.select_calculation()
        super().__init__(
            [
                self.select_calc_widget,
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
            self.node.get_object_content(self.cube_selector.value)
        )

    def get_calcs(self):
        query = orm.QueryBuilder()
        query.append(Cp2kOrbitalsWorkChain, tag="orbitals_wc", project="description")
        query.append(
            ShellJob,
            filters={"label": "cube-shrink"},
            project="uuid",
            with_incoming="orbitals_wc",
        )
        return query.all()

    def select_calculation(self, _=None):
        if not self.select_calc_widget.value:
            return
        calc = orm.load_node(self.select_calc_widget.value)
        self.node = calc.outputs.out_cubes
        self.remote_data_uuid = calc.inputs.nodes.remote_previous_job.uuid

        # TODO: number of occupied orbitals is hardcoded to 4:
        n_morb = 4
        orb_options = []
        for name in self.node.list_object_names():
            if "WFN" not in name:
                continue
            n_orb = int(name[18:23])
            if n_orb < n_morb:
                label = f"HOMO - {n_morb - n_orb}"
            elif n_orb == n_morb:
                label = "HOMO"
            elif n_orb == n_morb + 1:
                label = "LUMO"
            else:
                label = f"LUMO +{n_orb - n_morb - 1}"
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
