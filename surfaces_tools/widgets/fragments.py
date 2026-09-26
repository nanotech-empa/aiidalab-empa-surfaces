import aiidalab_widgets_base as awb
import aiidalab_widgets_empa as awe
import ipywidgets as ipw
import traitlets
from IPython.display import clear_output, display

from surfaces_tools.utils.atom_indices import string_range_to_list

from .analyze_structure import StructureAnalyzer
from ._lifecycle import close_owned_widgets

STYLE = {"description_width": "100px"}
BOX_LAYOUT = ipw.Layout(
    display="flex-wrap",
    flex_flow="row wrap",
    justify_content="space-between",
)


class Fragment(ipw.VBox):
    uks = traitlets.Bool(False)
    master_class = None

    def __init__(self, indices="1..2", name="no-name"):
        self._closed = False
        self._links = []
        self.label = ipw.HTML("<b>Fragment</b>")
        self.name = ipw.Text(description="Name", value=name, style=STYLE)
        self.indices = ipw.Text(
            value=indices, description="Indices", disabled=False, style=STYLE
        )
        self.charge = ipw.IntText(description="Charge", value=0, style=STYLE)
        self.multiplicity = ipw.IntText(
            description="Multiplicity", value=1, style=STYLE
        )

        self._links.append(
            ipw.dlink(
                (self.name, "value"),
                (self.label, "value"),
                transform=lambda x: f"<b>Fragment: {x}</b>",
            )
        )

        self.output = ipw.Output()
        self._multiplicity_box = ipw.VBox([self.multiplicity])

        # Delete button.
        self.delete_button = ipw.Button(description="Delete", button_style="danger")
        self.delete_button.on_click(self.delete_myself)

        # Resources.
        self.resources = awe.ProcessResourcesWidget()
        self.structure_analyzer = StructureAnalyzer()
        self.resources_estimator = awe.ResourcesEstimatorWidget()
        self.resources_estimator.link_to_resources_widget(self.resources)
        self._links.append(ipw.dlink((self, "uks"), (self.resources_estimator, "uks")))

        super().__init__(
            children=[
                self.label,
                ipw.HTML("<hr>"),
                self.name,
                self.indices,
                self.charge,
                self.output,
                ipw.HTML("Resources:"),
                self.resources,
                self.delete_button,
            ],
            layout={"width": "330px"},
        )

    @traitlets.observe("uks")
    def _observe_uks(self, change):
        with self.output:
            clear_output()
            if change["new"]:
                display(self._multiplicity_box)

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for link in getattr(self, "_links", ()):
            link.unlink()
        self._links = []
        if hasattr(self, "delete_button"):
            self.delete_button.on_click(self.delete_myself, remove=True)
        self.master_class = None
        close_owned_widgets(
            *self.children,
            getattr(self, "_multiplicity_box", None),
            getattr(self, "resources_estimator", None),
            self.layout,
        )
        self.children = ()
        super().close()

    def delete_myself(self, _):
        self.master_class.delete_fragment(self)

    def estimate_computational_resources(self, whole_structure, selected_code):
        self.structure_analyzer.structure = whole_structure[
            string_range_to_list(self.indices.value)[0]
        ]
        self.resources_estimator.details = self.structure_analyzer.details
        self.resources_estimator.selected_code = selected_code
        self.resources_estimator.estimate_resources()


class FragmentList(ipw.VBox):
    fragments = traitlets.List()
    selection_string = traitlets.Unicode()
    uks = traitlets.Bool(False)

    def __init__(self, add_fragment_visibility="visible"):
        self._closed = False
        self._fragment_links = {}
        # Fragment selection.
        self.new_fragment_name = ipw.Text(
            value="",
            description="Fragment name",
            style={"description_width": "initial"},
        )
        self.new_fragment_indices = ipw.HTML(
            value="", description="Selected indices:", style=STYLE
        )
        self._selection_link = ipw.dlink(
            (self, "selection_string"), (self.new_fragment_indices, "value")
        )
        self.add_new_fragment_button = ipw.Button(
            description="Add fragment", button_style="info"
        )
        self.add_new_fragment_button.on_click(self.add_fragment)

        # Outputs.
        self.fragment_add_message = awb.utils.StatusHTML()
        self.fragment_output = ipw.Box(layout=BOX_LAYOUT)
        super().__init__(
            children=[
                ipw.HBox(
                    [
                        self.new_fragment_name,
                        self.new_fragment_indices,
                        self.add_new_fragment_button,
                    ],
                    layout={"visibility": add_fragment_visibility},
                ),
                self.fragment_add_message,
                self.fragment_output,
            ]
        )

        self.fragment_output.children = self.fragments

    def add_fragment(self, _):
        """Add a fragment to the list of fragments."""
        if not self.selection_string:
            self.fragment_add_message.message = """<span style="color:red"> Error:</span> Please select a fragment first."""
            return
        if not self.new_fragment_name.value:
            self.fragment_add_message.message = """<span style="color:red">Error:</span> Please enter a name for the fragment."""
            return
        self.fragment_add_message.message = f"""<span style="color:blue">Info:</span> Adding {self.new_fragment_name.value} ({self.selection_string}) to the fragment list."""
        self.fragments = self.fragments + [
            Fragment(indices=self.selection_string, name=self.new_fragment_name.value)
        ]
        self.new_fragment_name.value = ""

    def delete_fragment(self, fragment):
        try:
            index = self.fragments.index(fragment)
        except ValueError:
            self.fragment_add_message.message = f"""<span style="color:red">Error:</span> Fragment {fragment} not found."""
            return

        self.fragment_add_message.message = f"""<span style="color:blue">Info:</span> Removing {fragment.name.value} ({fragment.indices.value}) from the fragment list."""
        self.fragments = self.fragments[:index] + self.fragments[index + 1 :]

    @traitlets.observe("fragments")
    def _observe_fragments(self, change):
        """Update the list of fragments."""
        current = set(change["new"])
        for fragment in list(self._fragment_links):
            if fragment not in current:
                self._fragment_links.pop(fragment).unlink()
                fragment.master_class = None
                fragment.close()
        for fragment in current:
            if fragment not in self._fragment_links:
                self._fragment_links[fragment] = ipw.dlink(
                    (self, "uks"), (fragment, "uks")
                )
                fragment.master_class = self
        self.fragment_output.children = change["new"]
        if not current:
            self.fragment_add_message.message = ""

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.fragments = []
        if hasattr(self, "_selection_link"):
            self._selection_link.unlink()
        if hasattr(self, "add_new_fragment_button"):
            self.add_new_fragment_button.on_click(self.add_fragment, remove=True)
        close_owned_widgets(*self.children, self.layout, shared=(BOX_LAYOUT,))
        self.children = ()
        super().close()
