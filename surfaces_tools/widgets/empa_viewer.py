import aiidalab_widgets_base as awb
import nglview
import numpy as np
import traitlets as tr

from .analyze_structure import StructureAnalyzer


# Custom set for visualization.
def default_vis_func(structure):
    # from struct details['cases'] we receive a list of atom types in the structure:
    # 'a' adatoms
    # 'b' bulk
    # 'm' molecule
    # 's' surface
    # 'u' unknown
    # 'w' wire
    if structure is None:
        return {}, {}
    an = StructureAnalyzer()
    # an.set_trait('structure',structure)
    an.structure = structure
    details = an.details
    if not details:
        return {}, {}

    vis_dict = {}
    std_dict = {
        "a": {
            "ids": "set_ids",
            "aspectRatio": 8,
            "highlight_aspectRatio": 8.1,
            "highlight_color": "red",
            "highlight_opacity": 0.6,
            "name": "adatoms",
            "type": "ball+stick",
        },
        "b": {
            "ids": "set_ids",
            "aspectRatio": 4,
            "highlight_aspectRatio": 4.1,
            "highlight_color": "green",
            "highlight_opacity": 0.6,
            "name": "bulk",
            "type": "ball+stick",
        },
        "m": {
            "ids": "set_ids",
            "aspectRatio": 3.5,
            "highlight_aspectRatio": 3.6,
            "highlight_color": "red",
            "highlight_opacity": 0.6,
            "name": "molecule",
            "type": "ball+stick",
        },
        "s": {
            "ids": "set_ids",
            "aspectRatio": 10,
            "highlight_aspectRatio": 10.1,
            "highlight_color": "green",
            "highlight_opacity": 0.6,
            "name": "surface",
            "type": "ball+stick",
        },
        "u": {
            "ids": "set_ids",
            "aspectRatio": 5,
            "highlight_aspectRatio": 5.1,
            "highlight_color": "red",
            "highlight_opacity": 0.6,
            "name": "unknown",
            "type": "ball+stick",
        },
        "w": {
            "ids": "set_ids",
            "aspectRatio": 5.5,
            "highlight_aspectRatio": 5.6,
            "highlight_color": "red",
            "highlight_opacity": 0.6,
            "name": "wire",
            "type": "ball+stick",
        },
    }
    current_rep = 0
    # ['b','w','s','m','a','u'] order matters
    if "b" in details["cases"]:
        vis_dict[current_rep] = std_dict["b"]
        ids = list(details["bulkatoms"])
        vis_dict[current_rep]["ids"] = awb.utils.list_to_string_range(ids, shift=0)
    if "w" in details["cases"]:
        vis_dict[current_rep] = std_dict["w"]
        ids = list(details["wireatoms"])
        vis_dict[current_rep]["ids"] = awb.utils.list_to_string_range(ids, shift=0)
    if "s" in details["cases"]:
        vis_dict[current_rep] = std_dict["s"]
        ids = list(details["slabatoms"] + details["bottom_H"])
        vis_dict[current_rep]["ids"] = awb.utils.list_to_string_range(ids, shift=0)
        current_rep += 1
    if "m" in details["cases"]:
        vis_dict[current_rep] = std_dict["m"]
        ids = [item for sublist in details["all_molecules"] for item in sublist]
        vis_dict[current_rep]["ids"] = awb.utils.list_to_string_range(ids, shift=0)
        current_rep += 1
    if "a" in details["cases"]:
        vis_dict[current_rep] = std_dict["a"]
        ids = list(details["adatoms"])
        vis_dict[current_rep]["ids"] = awb.utils.list_to_string_range(ids, shift=0)
        current_rep += 1
    if "u" in details["cases"]:
        vis_dict[current_rep] = std_dict["u"]
        ids = list(details["unclassified"])
        vis_dict[current_rep]["ids"] = awb.utils.list_to_string_range(ids, shift=0)

    return vis_dict, details


class EmpaStructureViewer(awb.structures.StructureDataViewer):
    DEFAULT_SELECTION_OPACITY = 0.2
    DEFAULT_SELECTION_RADIUS = 6
    DEFAULT_SELECTION_COLOR = "green"
    vis_dict = tr.Dict()
    details = tr.Dict()
    sys_type = tr.Unicode()

    def __init__(self, vis_func=default_vis_func, **kwargs):
        """
        vis_func : function to assign visualization. Gets self.displayed_structure as input
        and must return a dictionary where keys are integers 0,1,2,.. for the components of the
        representation and ids cover all atom indexes Supported: {'ball+stick','licorice','hyperball'}
        Example of dictionary returned by a possible vis_func with two components:
                vis_dict={
            1 : {
                'ids' : '0 3..40',
                'aspectRatio' : 3 ,
                'highlight_aspectRatio' : 3.3 ,
                'highlight_color' : 'red',
                'highlight_opacity' : 1,
                'name' : 'molecule',
                'type' : 'ball+stick'
            },
            0 : {
                'ids' : '2 41..49',
                'aspectRatio' : 10 ,
                'highlight_aspectRatio' : 10.3 ,
                'highlight_color' : 'green',
                'highlight_opacity' :1,
                'name' : 'substrate',
                'type' : 'ball+stick'
            },
        }
        """

        # Needed to display info about selected atoms e.g. distance, angle.
        self.selection_dict = {}
        self.selection_info = ""
        self.vis_func = vis_func
        super().__init__(**kwargs)

    def _gen_translation_indexes(self):
        """Transfromation of indexes in case of multiple representations
        dictionaries for  back and forth transformations."""

        self._translate_i_glob_loc = {}
        self._translate_i_loc_glob = {}
        for component in range(len(self.vis_dict.keys())):
            comp_i = 0
            ids = list(
                awb.utils.string_range_to_list(
                    self.vis_dict[component]["ids"], shift=0
                )[0]
            )
            for i_g in ids:
                self._translate_i_glob_loc[i_g] = (component, comp_i)
                self._translate_i_loc_glob[(component, comp_i)] = i_g
                comp_i += 1

    def _translate_glob_loc(self, indexes):
        """From global index to indexes of different components."""
        all_comp = [[] for i in range(len(self.vis_dict.keys()))]
        for i_g in indexes:
            i_c, i_a = self._translate_i_glob_loc[i_g]
            all_comp[i_c].append(i_a)

        return all_comp

    def _on_atom_click(self, change=None):  # pylint:disable=unused-argument
        """Update selection when clicked on atom."""

        if "atom1" not in self._viewer.picked.keys():
            return  # did not click on atom
        index = self._viewer.picked["atom1"]["index"]

        displayed_selection = self.displayed_selection.copy()

        if self.vis_dict:
            component = self._viewer.picked["component"]
            index = self._translate_i_loc_glob[(component, index)]

        if displayed_selection:
            if index not in displayed_selection:
                self.selection_dict[index] = self._viewer.picked["atom1"]
                displayed_selection.append(index)
            else:
                displayed_selection.remove(index)
                self.selection_dict.pop(index, None)
        else:
            self.selection_dict[index] = self._viewer.picked["atom1"]
            displayed_selection = [index]

        self.displayed_selection = displayed_selection

        return

        # First update selection_dict then update selection.
        selection_tmp = self.selection.difference({index})

        self.selection = selection_tmp

    def highlight_atoms(
        self,
        vis_list,
        color=DEFAULT_SELECTION_COLOR,
        size=DEFAULT_SELECTION_RADIUS,
        opacity=DEFAULT_SELECTION_OPACITY,
    ):
        """Highlighting atoms according to the provided list."""

        if not hasattr(self._viewer, "component_0"):
            return

        if self.vis_dict is None:
            self._viewer._remove_representations_by_name(repr_name="selected_atoms")
            self._viewer.add_ball_and_stick(
                name="selected_atoms",
                selection=[] if vis_list is None else vis_list,
                color=color,
                aspectRatio=size,
                opacity=opacity,
            )
        else:
            ncomponents = len(self.vis_dict.keys())
            for component in range(ncomponents):
                name = "highlight_" + self.vis_dict[component]["name"]
                self._viewer._remove_representations_by_name(
                    repr_name=name, component=component
                )
                color = self.vis_dict[component]["highlight_color"]
                aspect_ratio = self.vis_dict[component]["highlight_aspectRatio"]
                opacity = self.vis_dict[component]["highlight_opacity"]
                if vis_list is None:
                    self._viewer.add_ball_and_stick(
                        name=name,
                        selection=[],
                        color=color,
                        aspectRatio=aspect_ratio,
                        opacity=opacity,
                        component=component,
                    )
                else:
                    all_comp = self._translate_glob_loc(vis_list)
                    selection = all_comp[component]
                    self._viewer.add_ball_and_stick(
                        name=name,
                        selection=selection,
                        color=color,
                        aspectRatio=aspect_ratio,
                        opacity=opacity,
                        component=component,
                    )

    def custom_vis(self, c=None):
        """use cutom visualization if available and if compatible"""
        if self.vis_func is None:
            return

        else:
            vis_dict, self.details = self.vis_func(self.structure)

            # Keys must be integers 0, 1, 2...
            if vis_dict:
                self.sys_type = self.details["system_type"]
                types = {vis_dict[i]["type"] for i in range(len(vis_dict))}

                # Only {'ball+stick','licorice'} implemented
                # Atom pick very difficult with 'licorice' and 'hyperball'.
                if not types.issubset({"ball+stick", "licorice", "hyperball"}):
                    print("type unknown")
                    self.vis_func = None
                    return

                # delete all old components
                while hasattr(self._viewer, "component_0"):
                    self._viewer.component_0.clear_representations()
                    cid = self._viewer.component_0.id
                    self._viewer.remove_component(cid)

                self.vis_dict = vis_dict
                for component in range(len(self.vis_dict)):
                    rep_indexes = list(
                        awb.utils.string_range_to_list(
                            self.vis_dict[component]["ids"], shift=0
                        )[0]
                    )
                    if rep_indexes:
                        mol = self.structure[rep_indexes]

                        self._viewer.add_component(
                            nglview.ASEStructure(mol), default_representation=False
                        )

                        if self.vis_dict[component]["type"] == "ball+stick":
                            aspect_ratio = self.vis_dict[component]["aspectRatio"]
                            self._viewer.add_ball_and_stick(
                                aspectRatio=aspect_ratio,
                                opacity=1.0,
                                component=component,
                            )
                        elif self.vis_dict[component]["type"] == "licorice":
                            self._viewer.add_licorice(opacity=1.0, component=component)
                        elif self.vis_dict[component]["type"] == "hyperball":
                            self._viewer.add_hyperball(opacity=1.0, component=component)
            self._gen_translation_indexes()
            self._viewer.add_unitcell()
            self._viewer.center()

    def orient_z_up(self, _=None):
        try:
            cell_z = self.structure.cell[2, 2]
            com = self.structure.get_center_of_mass()
            self._viewer._camera_orientation
            top_z_orientation = [
                1.0,
                0.0,
                0.0,
                0,
                0.0,
                1.0,
                0.0,
                0,
                0.0,
                0.0,
                -np.max([cell_z, 30.0]),
                0,
                -com[0],
                -com[1],
                -com[2],
                1,
            ]
            self._viewer._set_camera_orientation(top_z_orientation)
        except Exception:
            return

    @tr.observe("structure")
    def _observe_structure(self, change):
        super()._observe_structure(change=change)
        with self.hold_trait_notifications():
            if self.vis_func:
                self.custom_vis()
        self.orient_z_up()
