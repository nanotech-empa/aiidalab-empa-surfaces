from functools import reduce

import ipywidgets as ipw
import numpy as np
import traitlets as tr
from aiida import orm
from ase import Atoms
from IPython.display import clear_output, display

from ..utils.atom_indices import string_range_to_list
from ..utils.cp2k_input_validity import validate_input
from ..utils.neb_resources import at_least
from . import analyze_structure, constraints, stack


def cp2k_bool(value):
    return ".TRUE." if value else ".FALSE."


STYLE = {"description_width": "120px"}
LAYOUT = {"width": "70%"}
LAYOUT2 = {"width": "35%"}

# Shared by every NEB band setting, so they line up in one column instead of
# being spread across two half-width boxes with the labels pulled apart.
SETTING_STYLE = {"description_width": "150px"}
SETTING_LAYOUT = {"width": "330px"}


class InputDetails(ipw.VBox):
    structure = tr.Instance(Atoms, allow_none=True)  # needed for colvars
    structure_manager = tr.Any(allow_none=True)  # the structure browser, set by the app
    neb_state = tr.Dict()  # NEB replica setup, survives section rebuilds
    selected_code = tr.Union([tr.Unicode(), tr.Instance(orm.Code)], allow_none=True)
    details = tr.Dict()
    protocol = tr.Unicode()
    to_fix = tr.List()
    do_cell_opt = tr.Bool()
    uks = tr.Bool()
    net_charge = tr.Int()
    neb = tr.Bool()  # Set by app in case of neb calculation, to be linked to resources.
    replica = tr.Bool()  # Set by app in case of replica chain calculation.
    phonons = tr.Bool()  # Set by app in case of phonons calculation
    n_replica_trait = (
        tr.Int()
    )  # To be linked to resources used only if neb = True or phonons = True
    nproc_replica_trait = tr.Int()  # To be linked from resources to input_details  used only if neb = True or phonons = True
    n_replica_per_group_trait = (
        tr.Int()
    )  # To be linked to resources used only if neb = True

    def __init__(
        self,
    ):
        """
        Arguments:
            sections(list): list of tuples each containing the displayed name of an input section and the
                section object. Each object should containt 'structure' trait pointing to the imported
                structure. The trait will be linked to 'structure' trait of this class.
        """
        # Displaying input sections.
        self.output = ipw.Output()
        self.displayed_sections = []
        # self.neb = False

        self.strusture_analyzer = analyze_structure.StructureAnalyzer()
        tr.dlink((self, "structure"), (self.strusture_analyzer, "structure"))
        tr.dlink((self.strusture_analyzer, "details"), (self, "details"))

        super().__init__(children=[self.output])

    @tr.default("neb")
    def _default_neb(self):
        return False

    @tr.default("phonons")
    def _default_phonons(self):
        return False

    @tr.default("n_replica_trait")
    def _default_n_proc_replica(self):
        if self.neb:
            # Two: the initial and the last replica, an empty NebWidget's floor.
            # This value is pushed into the section when its traits are linked,
            # and traitlets' link suppresses the section's correction while that
            # first propagation is in flight - so a default below the floor
            # would stick here and reach the resource arithmetic.
            return 2
        return 1

    @tr.default("n_replica_per_group_trait")
    def _default_n_replica_per_group_trait(self):
        return 1

    @tr.observe("details", "neb", "replica", "phonons")
    def _observe_details(self, _=None):
        self.to_fix = []
        self.net_charge = 0
        self.do_cell_opt = False
        self.uks = False
        with self.output:
            clear_output()
            if self.details:
                sys_type = self.details["system_type"]
                if self.neb:
                    sys_type = "Neb"
                if self.replica:
                    sys_type = "Replica"
                if self.phonons:
                    sys_type = "Phonons"
            else:
                sys_type = "None"

            self.displayed_sections = []
            add_children = []
            for sec in SECTIONS_TO_DISPLAY[sys_type]:
                section = sec()
                if hasattr(section, "traits_to_link"):
                    for trait in section.traits_to_link():
                        tr.link((self, trait), (section, trait))
                self.displayed_sections.append(section)
            display(ipw.VBox(add_children + self.displayed_sections))

    def return_final_dictionary(self):
        final_dictionary = {
            "elements": self.details["all_elements"],
            "system_type": self.details["system_type"],
        }

        # Retrieve all widgets values.
        for section in self.displayed_sections:
            try:
                to_add = section.return_dict()
            except ValueError as exc:
                return False, str(exc), final_dictionary
            if to_add:
                for key in to_add.keys():
                    if key in final_dictionary.keys():
                        final_dictionary[key].update(to_add[key])
                    else:
                        final_dictionary[key] = to_add[key]

        # If its a molecule, make non-periodic.
        if (
            self.details["system_type"] == "Molecule"
            and "forceperiodic" not in final_dictionary["dft_params"].keys()
        ):
            final_dictionary["dft_params"]["periodic"] = "NONE"

        # Check input validity.
        can_submit, error_msg = validate_input(self.details, final_dictionary)

        # print(self.final_dictionary)
        return can_submit, error_msg, final_dictionary


class DescriptionWidget(ipw.Text):
    def __init__(self):
        super().__init__(
            description="Process description: ",
            value="",
            placeholder="Type the name here.",
            style={"description_width": "120px"},
            layout={"width": "70%"},
        )

    def return_dict(self):
        return {"description": self.value}


class StructureInfoWidget(ipw.Accordion):
    details = tr.Dict()

    def __init__(self):
        self.info = ipw.Output()

        super().__init__(children=[ipw.VBox([self.info])], selected_index=None)
        self.set_title(0, "Structure details")

    @tr.observe("details")
    def _observe_details(self, _=None):
        if self.details is None:
            return
        else:
            self.children = [ipw.VBox([self.info])]
            with self.info:
                clear_output()
                print(self.details["summary"])

    def traits_to_link(self):
        return ["details"]

    def return_dict(self):
        return {}


class VdwSelectorWidget(ipw.Checkbox):
    def __init__(self):
        super().__init__(
            value=True,
            description="Dispersion Corrections",
            tooltip="VDW_POTENTIAL",
            style={"description_width": "120px"},
        )

    def return_dict(self):
        return {"dft_params": {"vdw": self.value}}

    def traits_to_link(self):
        return []


class ForcePeriodicWidget(ipw.Checkbox):
    def __init__(self):
        super().__init__(
            value=False,
            description="Keep periodic",
            # style={"description_width": "120px"},
        )

    def return_dict(self):
        if self.value:
            return {"dft_params": {"forceperiodic": self.value}}
        else:
            return {}

    def traits_to_link(self):
        return []


class ReplicaWidget(ipw.VBox):
    def __init__(self):
        self.restart_from = ipw.Text(
            description="Restart from PK:",
            value="",
            style={"description_width": "120px"},
            layout={"width": "340px"},
        )
        info1 = ipw.HTML(
            value="""If you want to restart from a previous Replica calculation, please enter the PK of the replica calculation.<br>
            Define all CVs using <strong style='color: green;'>'Add constraint'</strong> at least one has to be defined. Check that the automatically provided value for the CVs matches the selected geometry<br>
            Define the final value that each CV should reach<br>
            Define the increment for each CV (e.g. 0.05 to increase a bond length)<br>
            If you do not want a CV to evolve set 0.0 as increment, still adjustments in the CV value may occurr<br>""",
            layout={"width": "70%"},
        )
        self.CVs_targets = ipw.Text(
            description="CVs targets",
            value="",
            style={"description_width": "120px"},
            layout={"width": "70%"},
        )
        self.CVs_increments = ipw.Text(
            description="CVs increments",
            value="",
            style={"description_width": "120px"},
            layout={"width": "70%"},
        )
        super().__init__(
            children=[info1, self.CVs_targets, self.CVs_increments, self.restart_from],
        )

    def return_dict(self):
        the_dict = {}
        if self.restart_from.value:
            the_dict["restart_from"] = orm.load_node(self.restart_from.value).uuid
        the_dict["sys_params"] = {
            "colvars_targets": [float(i) for i in self.CVs_targets.value.split()],
            "colvars_increments": [float(i) for i in self.CVs_increments.value.split()],
        }
        return the_dict

    def traits_to_link(self):
        return []


def replica_group_divisors(n_replica):
    """Divisors of ``n_replica``, sorted - the valid "# rep / group" choices.

    A count below 1 cannot be produced by the form, which floors it at the
    number of replicas provided. It is pinned to ``[1]`` rather than left to
    raise, because the trait can be written from outside the widget and the
    generator below is empty for 0.
    """
    if n_replica < 1:
        return [1]
    return sorted(
        {
            divisor
            for i in range(1, int(n_replica**0.5) + 1)
            if n_replica % i == 0
            for divisor in (i, n_replica // i)
        }
    )


def validate_replica_pair(previous, current):
    """Check that two consecutive replicas describe the same atoms in the same order.

    Takes ``ase.Atoms`` rather than AiiDA nodes so it is testable without a
    profile. Returns ``(ok, message)``; ``message`` is empty when ``ok``.
    """
    if len(previous) != len(current):
        return False, f"Atom count changed: {len(previous)} -> {len(current)}."

    previous_symbols = previous.get_chemical_symbols()
    current_symbols = current.get_chemical_symbols()
    for index, (previous_symbol, current_symbol) in enumerate(
        zip(previous_symbols, current_symbols)
    ):
        if previous_symbol != current_symbol:
            return (
                False,
                f"Atom order changed at index {index}: "
                f"{previous_symbol} -> {current_symbol}.",
            )
    return True, ""


def compare_replica_cells(previous, current):
    """Describe how two replicas' cells differ; empty string when they match."""
    differences = []
    if list(previous.pbc) != list(current.pbc):
        differences.append("PBC differs")
    if not np.allclose(previous.cell.array, current.cell.array):
        differences.append("cell differs")
    return ", ".join(differences)


def replica_distance(previous, current):
    """Cartesian norm between two replicas' positions, in Angstrom."""
    return float(np.linalg.norm(previous.positions - current.positions))


def interpolate_replicas(first, last, coefficient):
    """Linearly interpolate a geometry between two endpoints.

    ``coefficient`` 0 reproduces ``first`` and 1 reproduces ``last``. Cell and
    symbols are taken from ``first``, which is only meaningful for a pair that
    passed :func:`validate_replica_pair`.
    """
    interpolated = first.copy()
    interpolated.positions = (
        1.0 - coefficient
    ) * first.positions + coefficient * last.positions
    return interpolated


def _colored(text, color):
    return f"<span style='color:{color}'>{text}</span>"


def _load_replica_node(pk):
    """Load a replica by PK, returning ``(node, problem)``.

    A mistyped PK is a normal thing for a user to do and must not blow up the
    table rendering, so the failure is returned rather than raised.
    """
    if not pk:
        return None, ""
    try:
        return orm.load_node(int(pk)), ""
    except Exception as exc:  # noqa: BLE001 - NotExistent, ValueError, ...
        return None, f"Cannot load PK {pk}: {exc}"


class NebReplicaRow(ipw.VBox):
    """One replica: its PK, the ways to fill it in, and its status.

    Every replica is entered the same way. The endpoints differ only in that
    they carry fewer buttons: nothing is inserted after the last replica, and
    neither endpoint can be removed.
    """

    def __init__(
        self, parent, name, pk=0, can_insert=True, can_remove=True, can_interpolate=True
    ):
        self.parent = parent
        self.pk = ipw.IntText(
            value=pk,
            style={"description_width": "120px"},
            # 120px of that is the description; the rest holds nine digits plus
            # the spinner, which is more PK than this database will ever reach.
            layout={"width": "220px"},
        )
        self.from_current = ipw.Button(
            description="From viewer",
            tooltip="Take the structure currently shown in the structure "
            "browser, storing it first if it is not stored yet",
            layout={"width": "110px"},
        )
        self.show = ipw.Button(description="Show", layout={"width": "70px"})
        self.insert = ipw.Button(
            icon="plus",
            button_style="success",
            tooltip="Insert an empty replica below this one",
            layout={"width": "45px"},
        )
        self.remove = ipw.Button(
            icon="times",
            button_style="danger",
            tooltip="Remove this replica",
            layout={"width": "45px"},
        )
        self.factor = ipw.FloatText(
            value=0.5,
            step=0.1,
            description="Factor:",
            tooltip=(
                "Where this replica sits between the ones above and below: "
                "0 is the one above, 1 the one below, 0.5 the midpoint. "
                "Outside 0-1 it lands beyond an endpoint instead."
            ),
            style={"description_width": "55px"},
            layout={"width": "130px"},
        )
        self.interpolate = ipw.Button(
            description="Interpolate",
            tooltip="Build this replica between the ones above and below",
            layout={"width": "110px"},
        )
        self.interpolation_box = (
            ipw.HBox([self.factor, self.interpolate]) if can_interpolate else None
        )
        # Ends the line: distance to the replica above, and anything wrong with
        # this one. Replaces the summary table, which repeated the PK and the
        # name already shown on the row.
        self.info = ipw.HTML(layout={"width": "auto"})
        # Second line, for the rare failure. Collapsed while empty so a healthy
        # row is exactly one line tall.
        self.status = ipw.HTML(layout={"width": "95%", "display": "none"})
        self.rename(name)

        # Observers are attached last so that the values passed in above do not
        # fire them while the parent is still rebuilding its row list.
        self.pk.observe(self.parent.update_replica_info, "value")
        for button, action in (
            (self.from_current, self.parent.set_row_from_current),
            (self.show, self.parent.show_row),
            (self.insert, self.parent.insert_row_below),
            (self.remove, self.parent.remove_row),
            (self.interpolate, self.parent.interpolate_row),
        ):
            button.on_click(
                lambda _, action=action: self.parent.run_handler(
                    self, lambda: action(self)
                )
            )

        buttons = [self.from_current, self.show]
        if can_insert:
            buttons.append(self.insert)
        if can_remove:
            buttons.append(self.remove)
        if self.interpolation_box is not None:
            buttons.append(self.interpolation_box)
        # The buttons sit in a fixed-width box so that the info at the end of
        # the line starts at the same place on every row. Rows carry different
        # buttons - no remove on an endpoint, no interpolation once filled in -
        # and without this the info column would step in and out with them.
        super().__init__(
            children=[
                ipw.HBox(
                    [self.pk, ipw.HBox(buttons, layout={"width": "540px"}), self.info],
                    layout={"align_items": "center"},
                ),
                self.status,
            ]
        )

    def rename(self, name):
        """Positions shift as replicas are inserted and removed."""
        self.name = name
        self.pk.description = f"{name} PK:"

    def set_info(self, text, color):
        self.info.value = _colored(text, color) if text else ""

    def set_status(self, text, color):
        self.status.value = _colored(text, color) if text else ""
        self.status.layout.display = "block" if text else "none"

    def show_interpolation(self, visible, missing):
        """Offer interpolation only while this replica is still empty.

        Once it has a PK - interpolated, typed, or taken from the browser -
        the controls go away rather than sit there offering to overwrite it.
        They stay visible but dead while a neighbour is missing, and the
        button's tooltip then names what is missing, so a dead button is not
        left unexplained. ``missing`` holds the names of the undefined ends.
        """
        if self.interpolation_box is None:
            return
        self.interpolation_box.layout.display = "flex" if visible else "none"
        self.factor.disabled = self.interpolate.disabled = bool(missing)
        self.interpolate.tooltip = (
            f"Inactive: needs a defined replica above and below - "
            f"{' and '.join(missing)} still missing."
            if missing
            else "Build this replica between the ones above and below"
        )

    def get_node(self):
        return _load_replica_node(self.pk.value)


class NebWidget(ipw.VBox):
    structure_manager = tr.Any(allow_none=True)
    neb_state = tr.Dict()
    n_replica_trait = tr.Int()
    nproc_replica_trait = tr.Int()
    n_replica_per_group_trait = tr.Int()

    def __init__(self):
        self.restart_from = ipw.Text(
            description="Restart from PK:",
            value="",
            # Resolving the node on every keystroke would hunt for "1", "12",
            # "123" on the way to 1234 and flash an error for each.
            continuous_update=False,
            style={"description_width": "150px"},
            layout={"width": "90%"},
        )
        # Says what a restart PK does; the band-building guidance lives inside
        # the box below, so it disappears together with what it describes.
        self.restart_info = ipw.HTML(layout={"width": "90%"})
        # The endpoints are ordinary replica rows, built once and outliving the
        # intermediate ones. Nothing is inserted after the last replica, neither
        # endpoint can be removed, and neither can be interpolated - an endpoint
        # has nothing beyond it to interpolate against.
        self.initial_row = NebReplicaRow(
            self, "Initial", can_remove=False, can_interpolate=False
        )
        self.last_row = NebReplicaRow(
            self, "Last", can_insert=False, can_remove=False, can_interpolate=False
        )
        self.rows_box = ipw.VBox()

        self.align_frames = ipw.Checkbox(
            description="Align Frames",
            value=False,
            indent=True,
            style=SETTING_STYLE,
            layout=SETTING_LAYOUT,
        )
        self.rotate_frames = ipw.Checkbox(
            description="Rotate Frames",
            value=False,
            indent=True,
            style=SETTING_STYLE,
            layout=SETTING_LAYOUT,
        )
        self.optimize_endpoints = ipw.Checkbox(
            description="Optimize Endpoints",
            value=False,
            indent=True,
            style=SETTING_STYLE,
            layout=SETTING_LAYOUT,
        )
        self.band_type = ipw.Dropdown(
            options=["CI-NEB"],
            description="Band Type",
            value="CI-NEB",
            style=SETTING_STYLE,
            layout=SETTING_LAYOUT,
        )
        self.k_spring = ipw.Text(
            description="Spring constant",
            value="0.05",
            style=SETTING_STYLE,
            layout=SETTING_LAYOUT,
        )
        self.nproc_rep = ipw.HTML(
            description="# processors / rep",
            value="324",
            style=SETTING_STYLE,
            layout=SETTING_LAYOUT,
        )
        # CP2K's NUMBER_OF_REPLICA may exceed the number of &REPLICA sections
        # given: it fills the remainder by repeatedly bisecting the largest gap
        # in the band. It may never be smaller, so `min` tracks the rows below
        # and BoundedIntText raises the value when rows are added.
        self.n_replica = ipw.BoundedIntText(
            description="# of replica",
            value=2,
            min=2,
            max=1000,
            style={"description_width": "initial"},
            layout={"width": "200px"},
        )
        self.n_replica_info = ipw.HTML(layout={"width": "70%"})
        # Options are deliberately left empty here: they are derived from the
        # replica count by _update_replica_per_group_options, and a literal list
        # would be a second, immediately-overwritten statement of the same fact.
        self.n_replica_per_group = ipw.Dropdown(
            description="# rep / group",
            options=[],
            style=SETTING_STYLE,
            layout=SETTING_LAYOUT,
        )
        self.nsteps_it = ipw.Text(
            description="Steps before CI",
            value="5",
            style=SETTING_STYLE,
            layout=SETTING_LAYOUT,
        )

        self.replica_rows = []
        self._updating_from_state = False
        # How many replicas a restart brings with it, cached so the floor below
        # does not reload the node on every redraw. None when not restarting.
        self._restart_replica_count = None

        # Everything that defines the band, so a restart can hide it in one go.
        # The rows that build a band from scratch. A restart supplies the band
        # instead, so this hides - but the count below it does not, because
        # CP2K will still interpolate above whatever it inherits.
        self.replica_box = ipw.VBox(
            [
                # Sits inside the box that a restart hides, so the alternative
                # disappears along with the thing it is an alternative to.
                ipw.HTML(
                    "<div style='display:flex; align-items:center; gap:8px;"
                    " color:gray; width:90%;'>"
                    "<hr style='flex:1; border:none; border-top:1px solid #ddd;'>"
                    "OR build the band below"
                    "<hr style='flex:1; border:none; border-top:1px solid #ddd;'>"
                    "</div>"
                ),
                self.initial_row,
                self.rows_box,
                self.last_row,
            ]
        )
        # The band size belongs with the replicas that define it, not in the
        # settings tab: it is the control that says how many more CP2K should
        # interpolate.
        self.count_box = ipw.HBox([self.n_replica, self.n_replica_info])

        self.n_replica.observe(self.on_n_replica_change, "value")
        self.n_replica_per_group.observe(self.on_n_replica_per_group_change, "value")
        self.restart_from.observe(self.on_restart_change, "value")
        for widget in (
            self.restart_from,
            self.align_frames,
            self.rotate_frames,
            self.optimize_endpoints,
            self.band_type,
            self.k_spring,
            self.nsteps_it,
        ):
            widget.observe(self._observe_state_value, "value")

        # Two tabs: what the band is made of, and how it is run. The restart
        # field belongs with the replicas because that is what it replaces -
        # entering a PK takes the band from a previous calculation instead.
        self.tabs = ipw.Tab(
            children=[
                ipw.VBox(
                    [
                        self.restart_from,
                        self.restart_info,
                        self.replica_box,
                        self.count_box,
                    ]
                ),
                # One column, in the order the settings are reasoned about:
                # what kind of band, how it is optimised, how it is split over
                # the allocation.
                ipw.VBox(
                    [
                        self.band_type,
                        self.k_spring,
                        self.nsteps_it,
                        self.optimize_endpoints,
                        self.align_frames,
                        self.rotate_frames,
                        self.n_replica_per_group,
                        self.nproc_rep,
                    ]
                ),
            ]
        )
        self.tabs.set_title(0, "Replicas")
        self.tabs.set_title(1, "Advanced settings")

        super().__init__(children=[self.tabs])
        self._refresh_rows()
        self.on_restart_change()

    # ------------------------------------------------------------------
    # Handler boundary
    # ------------------------------------------------------------------

    def run_handler(self, row, action):
        """Run a click handler, reporting any failure on ``row``.

        ipywidgets swallows exceptions raised inside ``on_click``, so without
        this a failed click does nothing at all with no explanation. Errors are
        raised where they are detected and rendered here.
        """
        try:
            row.set_status("", "")
            action()
        except Exception as exc:  # noqa: BLE001 - anything reaching here is user-facing
            row.set_status(exc, "red")

    # ------------------------------------------------------------------
    # Restarting a previous NEB
    # ------------------------------------------------------------------

    def _restart_source(self):
        """The calculation being restarted: its structure and its replica count.

        A restart reuses that calculation's optimised replicas, so the band is
        not built here at all - but the workchain still needs ``structure``,
        which supplies the tags written into every replica file. Taking it from
        the calculation being restarted keeps it consistent with those replicas.
        """
        try:
            node = orm.load_node(self.restart_from.value)
            return (
                node,
                node.inputs.structure,
                node.inputs.neb_params["number_of_replica"],
            )
        except Exception as exc:  # noqa: BLE001 - NotExistent, AttributeError, KeyError
            raise ValueError(
                f"Cannot restart from PK {self.restart_from.value}: {exc}"
            ) from exc

    def on_restart_change(self, _=None):
        """Hide the band builder while a restart PK is present."""
        restarting = bool(self.restart_from.value.strip())
        self.replica_box.layout.display = "none" if restarting else "flex"
        if not restarting:
            self._restart_replica_count = None
            self.restart_info.value = ""
            self.update_replica_info()
            return
        try:
            _, structure, n_replica = self._restart_source()
        except ValueError as exc:
            self._restart_replica_count = None
            self.restart_info.value = _colored(exc, "red")
            self.update_replica_info()
            return
        # Becomes the floor for "# of replica": those replicas are written out
        # as &REPLICA sections, and CP2K cannot run fewer than it is given.
        self._restart_replica_count = n_replica
        self.restart_info.value = _colored(
            f"Restarting from PK {self.restart_from.value}: {n_replica} replicas, "
            f"structure PK {structure.pk}. Its optimised replicas are reused, so "
            "there is no band to define here - but you can ask for more below, "
            "and CP2K will bisect the widest gaps to reach that many.",
            "gray",
        )
        self.update_replica_info()

    def _replica_floor(self):
        """Fewest replicas the band can have: one per &REPLICA section written.

        Those come from the restart when there is one, and from the rows
        otherwise - the rows are still there while restarting, just hidden and
        not submitted, so they must not set the floor.
        """
        if self._restart_replica_count is not None:
            return self._restart_replica_count
        return len(self.all_rows())

    # ------------------------------------------------------------------
    # Structure browser
    # ------------------------------------------------------------------

    def _store_current_structure(self):
        """Store and return the structure currently shown in the browser."""
        if self.structure_manager is None:
            raise ValueError("No structure browser is connected to this form.")
        node = self.structure_manager.structure_node
        if node is not None and node.is_stored:
            return node
        if self.structure_manager.structure is None:
            raise ValueError("No structure is currently visualized.")
        return orm.StructureData(ase=self.structure_manager.structure).store()

    def _show_node(self, node):
        if self.structure_manager is None:
            raise ValueError("No structure browser is connected to this form.")
        self.structure_manager.input_structure = node

    def set_row_from_current(self, row):
        row.pk.value = self._store_current_structure().pk

    def show_row(self, row):
        node, problem = row.get_node()
        if problem:
            raise ValueError(problem)
        if node is None:
            raise ValueError(f"{row.name} replica is not defined.")
        self._show_node(node)

    # ------------------------------------------------------------------
    # Replica sequence
    # ------------------------------------------------------------------

    def all_rows(self):
        """Every replica row in order: initial, the intermediates, last."""
        return [self.initial_row, *self.replica_rows, self.last_row]

    def _replica_sequence(self):
        """``(name, node, problem)`` for every replica, in order, initial first.

        Rows are addressed by their position in :meth:`all_rows`, which is the
        one indexing convention used throughout; ``node`` is ``None`` when a
        replica is undefined or its PK could not be loaded.
        """
        sequence = []
        for row in self.all_rows():
            node, problem = row.get_node()
            sequence.append((row.name, node, problem))
        return sequence

    def _neighbouring_nodes(self, row):
        """The nearest defined replicas either side of ``row``."""
        sequence = self._replica_sequence()
        position = self.all_rows().index(row)
        before = next(
            (node for _, node, _ in reversed(sequence[:position]) if node is not None),
            None,
        )
        after = next(
            (node for _, node, _ in sequence[position + 1 :] if node is not None),
            None,
        )
        return before, after

    def _set_intermediate_rows(self, pks):
        self.replica_rows = [NebReplicaRow(self, "", pk=pk) for pk in pks]
        self._refresh_rows()

    def _refresh_rows(self):
        """Renumber the intermediate rows and show them."""
        for position, row in enumerate(self.replica_rows, start=1):
            row.rename(f"Intermediate {position}")
        self.rows_box.children = self.replica_rows
        self.update_replica_info()

    def insert_row_below(self, row):
        """Add an empty replica below ``row``, for Interpolate or a PK to fill."""
        self.replica_rows.insert(self.all_rows().index(row), NebReplicaRow(self, ""))
        self._refresh_rows()

    def interpolate_row(self, row):
        before, after = self._neighbouring_nodes(row)
        # The button is dead without both neighbours; this is the backstop, and
        # it names the side still missing rather than saying "cannot".
        missing = [
            name
            for name, node in (("Initial", before), ("Last", after))
            if node is None
        ]
        if missing:
            raise ValueError(
                "Interpolation needs a defined replica on either side - "
                f"{' and '.join(missing)} still missing."
            )
        first, last = before.get_ase(), after.get_ase()
        ok, message = validate_replica_pair(first, last)
        if not ok:
            raise ValueError(message)

        factor = row.factor.value
        node = orm.StructureData(ase=interpolate_replicas(first, last, factor)).store()
        node.label = "NEB interpolated replica"
        row.pk.value = node.pk
        # No confirmation on success: the PK appears in the field and the
        # distance at the end of the row, which says it better than a sentence.
        # A factor outside (0, 1) is the one case worth a word - the field is
        # unbounded, so that lands the replica beyond an endpoint.
        if not 0.0 < factor < 1.0:
            row.set_status(
                f"Factor {factor:g} is outside 0-1: this replica was placed "
                "beyond an endpoint, not between them.",
                "orange",
            )

    def remove_row(self, row):
        self.replica_rows.remove(row)
        self._refresh_rows()

    def validate_replicas(self):
        """Return every replica node in order, raising if the chain is unusable."""
        names = []
        nodes = []
        for name, node, problem in self._replica_sequence():
            if problem:
                raise ValueError(f"{name} replica: {problem}")
            if node is None:
                raise ValueError(f"{name} replica is not defined.")
            names.append(name)
            nodes.append(node)

        for position in range(1, len(nodes)):
            ok, message = validate_replica_pair(
                nodes[position - 1].get_ase(), nodes[position].get_ase()
            )
            if not ok:
                raise ValueError(
                    f"{names[position]} vs {names[position - 1]}: {message}"
                )

        return nodes

    # ------------------------------------------------------------------
    # Row info and derived quantities
    # ------------------------------------------------------------------

    def update_replica_info(self, _=None):
        # Requesting fewer replicas than CP2K is handed is meaningless; anything
        # above the floor it interpolates itself.
        self.n_replica.min = self._replica_floor()
        self.n_replica_trait = self.n_replica.value

        sequence = self._replica_sequence()
        # Reuse the sequence rather than asking each row for its neighbours:
        # that would reload every node once per row.
        nodes = [node for _, node, _ in sequence]
        previous = None
        for position, (row, (_, node, problem)) in enumerate(
            zip(self.all_rows(), sequence)
        ):
            if problem:
                row.set_info(problem, "red")
            elif node is None:
                row.set_info("missing", "red")
            elif previous is None:
                # Nothing above it to measure from; a bare PK is the whole story.
                row.set_info("", "gray")
            else:
                previous_atoms, current_atoms = previous.get_ase(), node.get_ase()
                ok, message = validate_replica_pair(previous_atoms, current_atoms)
                if not ok:
                    row.set_info(message, "red")
                else:
                    warning = compare_replica_cells(previous_atoms, current_atoms)
                    distance = replica_distance(previous_atoms, current_atoms)
                    row.set_info(
                        f"&Delta; {distance:.3f} &#8491;"
                        + (f" &middot; {warning}" if warning else ""),
                        "orange" if warning else "gray",
                    )
            row.show_interpolation(
                visible=node is None,
                missing=[
                    name
                    for name, defined in (
                        ("Initial", any(o is not None for o in nodes[:position])),
                        ("Last", any(o is not None for o in nodes[position + 1 :])),
                    )
                    if not defined
                ],
            )
            if node is not None:
                previous = node

        provided = self._replica_floor()
        interpolated = self.n_replica.value - provided
        source = (
            "inherited from the restart"
            if self._restart_replica_count is not None
            else "defined above"
        )
        self.n_replica_info.value = _colored(
            f"{provided} {source}"
            + (
                f", {interpolated} interpolated by CP2K, which bisects the widest "
                "gap in the band until it has this many."
                if interpolated
                else " - raise this to have CP2K interpolate the rest."
            ),
            "gray",
        )

        self._sync_state()

    def _update_replica_per_group_options(self):
        """Rebuild the divisor options, keeping the selection when still valid."""
        divisors = replica_group_divisors(self.n_replica_trait)
        previous = self.n_replica_per_group.value
        self.n_replica_per_group.options = divisors
        self.n_replica_per_group.value = previous if previous in divisors else 1

    # ------------------------------------------------------------------
    # Traits
    # ------------------------------------------------------------------

    def return_dict(self):
        the_dict = {}
        if self.restart_from.value != "":
            # A restart reuses that calculation's replicas and its structure, so
            # the band builder is not read here - but its replicas still become
            # &REPLICA sections, so they still set the floor for the count.
            restart_node, structure, n_previous = self._restart_source()
            the_dict["restart_from"] = restart_node.uuid
            nodes = [structure]
            floor = n_previous
            floor_source = f"replicas inherited from PK {self.restart_from.value}"
        else:
            nodes = self.validate_replicas()
            floor = len(nodes)
            floor_source = "replicas provided"

        # CP2K interpolates the replicas beyond those it is given, but cannot
        # run fewer. The widget bounds this too; this is the backstop for a
        # count written straight into the trait.
        n_replica = self.n_replica.value
        if n_replica < floor:
            raise ValueError(
                f"# of replica is {n_replica}, below the {floor} {floor_source}. "
                "Raise it."
            )

        the_dict["initial_uuid"] = nodes[0].uuid
        the_dict["neb_params"] = {
            "align_frames": cp2k_bool(self.align_frames.value),
            "rotate_frames": cp2k_bool(self.rotate_frames.value),
            "band_type": self.band_type.value,
            "k_spring": self.k_spring.value,
            "nproc_rep": at_least(self.nproc_rep.value),
            "number_of_replica": n_replica,
            "nsteps_it": int(self.nsteps_it.value),
            "optimize_end_points": cp2k_bool(self.optimize_endpoints.value),
        }
        if len(nodes) > 1:
            the_dict["replica_uuids"] = [node.uuid for node in nodes[1:]]

        return the_dict

    @tr.observe("nproc_replica_trait")
    def _observe_nproc_replica_trait(self, _=None):
        self.nproc_rep.value = str(at_least(self.nproc_replica_trait))

    def on_n_replica_change(self, _=None):
        self.n_replica_trait = self.n_replica.value
        # Redraws the table, whose footer reports how many CP2K will add.
        self.update_replica_info()

    @tr.observe("n_replica_trait")
    def _observe_n_replica_trait(self, _=None):
        # An external write - InputDetails pushes its default in when the
        # section is linked - is clamped to the replicas actually provided
        # rather than displayed as a number the band cannot have.
        self.n_replica.value = self.n_replica_trait
        if self.n_replica_trait != self.n_replica.value:
            self.n_replica_trait = self.n_replica.value
            return
        self._update_replica_per_group_options()

    def on_n_replica_per_group_change(self, _=None):
        if self.n_replica_per_group.value is not None:
            self.n_replica_per_group_trait = self.n_replica_per_group.value

    # ------------------------------------------------------------------
    # State persistence across input-section rebuilds
    # ------------------------------------------------------------------

    def _current_state(self):
        """The replica setup, for round-tripping through InputDetails.

        InputDetails rebuilds every input section whenever `details`, `neb`,
        `replica` or `phonons` changes, which would otherwise discard the
        replica setup entered before that happened. Adding a field here means
        adding it to _observe_neb_state as well - the two lists must agree.
        """
        return {
            "restart_from": self.restart_from.value,
            "initial_pk": int(self.initial_row.pk.value or 0),
            "last_pk": int(self.last_row.pk.value or 0),
            "intermediate_pks": [int(row.pk.value or 0) for row in self.replica_rows],
            "n_replica": int(self.n_replica.value),
            "align_frames": bool(self.align_frames.value),
            "rotate_frames": bool(self.rotate_frames.value),
            "optimize_endpoints": bool(self.optimize_endpoints.value),
            "band_type": self.band_type.value,
            "k_spring": self.k_spring.value,
            "nsteps_it": self.nsteps_it.value,
        }

    def _sync_state(self):
        if not self._updating_from_state:
            self.neb_state = self._current_state()

    def _observe_state_value(self, _=None):
        self._sync_state()

    @tr.observe("neb_state")
    def _observe_neb_state(self, _=None):
        state = dict(self.neb_state or {})
        # _sync_state writes this trait, so most changes are the form's own echo.
        # Restoring one would rebuild the rows and discard the row being edited.
        if not state or state == self._current_state():
            return
        # Restoring sets widget values, which fire the observers that write the
        # state back; the guard keeps a restore from overwriting itself.
        self._updating_from_state = True
        try:
            self.restart_from.value = state.get("restart_from", "")
            self.initial_row.pk.value = int(state.get("initial_pk", 0) or 0)
            self.last_row.pk.value = int(state.get("last_pk", 0) or 0)
            self._set_intermediate_rows(
                [int(pk or 0) for pk in state.get("intermediate_pks", [])]
            )
            # After the rows: they set the lower bound this is clamped to.
            self.n_replica.value = int(state.get("n_replica", self.n_replica.value))
            self.align_frames.value = bool(
                state.get("align_frames", self.align_frames.value)
            )
            self.rotate_frames.value = bool(
                state.get("rotate_frames", self.rotate_frames.value)
            )
            self.optimize_endpoints.value = bool(
                state.get("optimize_endpoints", self.optimize_endpoints.value)
            )
            self.band_type.value = state.get("band_type", self.band_type.value)
            self.k_spring.value = state.get("k_spring", self.k_spring.value)
            self.nsteps_it.value = state.get("nsteps_it", self.nsteps_it.value)
        finally:
            self._updating_from_state = False
        self.update_replica_info()

    def traits_to_link(self):
        return [
            "structure_manager",
            "neb_state",
            "n_replica_trait",
            "nproc_replica_trait",
            "n_replica_per_group_trait",
        ]


class PhononsWidget(ipw.VBox):
    details = tr.Dict()
    n_replica_trait = tr.Int()
    nproc_replica_trait = tr.Int()

    def __init__(self):
        self.nproc_rep = ipw.HTML(
            description="# processors / rep",
            value="324",
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )
        self.n_replica = ipw.Dropdown(
            description="# of replica",
            value=3,
            options=[1, 3],
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )

        self.n_replica.observe(self.on_n_replica_change, "value")

        super().__init__(
            children=[
                self.nproc_rep,
                self.n_replica,
            ],
        )

    def return_dict(self):
        return {
            "phonons_params": {
                "nproc_rep": int(self.nproc_rep.value),
            }
        }

    @tr.observe("nproc_replica_trait")
    def _observe_nproc_replica_trait(self, _=None):
        self.nproc_rep.value = str(self.nproc_replica_trait)

    @tr.observe("details")
    def _observe_details(self, _=None):
        self.n_replica.value = 3
        three_times_natoms = self.details["numatoms"] * 3
        self.n_replica.options = set(
            reduce(
                list.__add__,
                (
                    [i, three_times_natoms // i]
                    for i in range(1, int(three_times_natoms**0.5) + 1)
                    if three_times_natoms % i == 0
                ),
            )
        )

    def on_n_replica_change(self, _=None):
        self.n_replica_trait = int(self.n_replica.value)

    def traits_to_link(self):
        return ["n_replica_trait", "nproc_replica_trait", "details"]


class UksSectionWidget(ipw.Accordion):
    details = tr.Dict()
    uks = tr.Bool()
    net_charge = tr.Int()

    def __init__(self, charge_visibility="visible", multiplicity_visibility="visible"):
        self.uks_toggle = ipw.Checkbox(
            value=False,
            description="UKS",
            tooltip="Activate UKS",
            style={"description_width": "initial"},
            layout={"width": "60px"},
        )
        tr.link((self, "uks"), (self.uks_toggle, "value"))

        # Spins
        class OneSpinWidget(stack.HorizontalItemWidget):
            def __init__(self):
                self.selection = ipw.Text(
                    description="Atoms indices:",
                    style={"description_width": "initial"},
                )
                self.starting_magnetization = ipw.IntText(
                    description="Magnetization value:",
                    style={"description_width": "initial"},
                )
                super().__init__(children=[self.selection, self.starting_magnetization])

        self.spins = stack.VerticalStackWidget(
            item_class=OneSpinWidget, add_button_text="Add spin set"
        )

        self.charge = ipw.IntText(
            value=0,
            description="Net charge:",
            style={"description_width": "initial"},
            layout={"width": "120px", "visibility": charge_visibility},
        )
        tr.link((self, "net_charge"), (self.charge, "value"))

        self.multiplicity = ipw.IntText(
            value=1,
            description="Multiplicity:",
            style={"description_width": "initial"},
            layout={"width": "140px", "visibility": multiplicity_visibility},
        )

        self.uks_box = [
            ipw.VBox(
                [
                    ipw.HBox(
                        [
                            self.uks_toggle,
                            self.charge,
                            self.multiplicity,
                        ]
                    ),
                    self.spins,
                ]
            )
        ]
        self.no_uks_box = [ipw.HBox([self.uks_toggle, self.charge])]

        super().__init__(selected_index=None)

        self.children = self.no_uks_box
        self.set_title(0, "Spin-polarized calculation")

    def return_dict(self):
        to_return = {
            "uks": self.uks,
            "charge": self.net_charge,
        }

        if self.uks:
            magnetization_per_site = np.zeros(self.details["numatoms"])
            for spinset in self.spins.items:
                atom_indices, is_valid = string_range_to_list(spinset.selection.value)
                if not is_valid:
                    raise ValueError(
                        f"Invalid spin atom indices: {spinset.selection.value!r}"
                    )
                if any(
                    atom_index < 0 or atom_index >= self.details["numatoms"]
                    for atom_index in atom_indices
                ):
                    raise ValueError(
                        "Spin atom indices must be between 1 and "
                        f"{self.details['numatoms']}."
                    )
                magnetization_per_site[atom_indices] = (
                    spinset.starting_magnetization.value
                )
            to_return.update(
                {
                    "multiplicity": self.multiplicity.value,
                    "magnetization_per_site": magnetization_per_site.astype(
                        np.int32
                    ).tolist(),
                }
            )
        return {"dft_params": to_return}

    @tr.observe("uks")
    def _observe_uks(self, value=None):
        self.children = self.uks_box if value["new"] else self.no_uks_box

    def traits_to_link(self):
        return ["details", "uks", "net_charge"]


class CellSectionWidget(ipw.Accordion):
    details = tr.Dict()
    do_cell_opt = tr.Bool()

    def __init__(self):
        self.cell_symmetry = ipw.Dropdown(
            description="Cell symmetry:",
            options=[
                "CUBIC",
                "HEXAGONL",
                "MONOCLINIC",
                "NONE",
                "ORTHORHOMBIC",
                "RHOMBOHEDRAL",
                "TETRAGONAL_AB",
                "TETRAGONAL_AC",
                "TETRAGONAL_BC",
                "TRICLINIC",
            ],
            value="ORTHORHOMBIC",
            style=STYLE,
        )

        self.cell_constraint = ipw.Dropdown(
            description="Cell constraints:",
            options=["XYZ", "NONE", "X", "XY", "XZ", "Y", "YZ", "Z"],
            value="NONE",
            style=STYLE,
        )

        self.cell_freedom = ipw.Dropdown(
            options=["FREE", "KEEP_SYMMETRY", "KEEP_ANGLES", "KEEP_SPACE_GROUP"],
            description="Cell freedom",
            value="KEEP_SYMMETRY",
            style=STYLE,
        )

        self.opt_cell = ipw.Checkbox(
            value=False,
            description="Optimize cell",
        )
        tr.link((self, "do_cell_opt"), (self.opt_cell, "value"))

        super().__init__(
            selected_index=None,
            children=[
                ipw.VBox(
                    [
                        self.cell_symmetry,
                        self.cell_freedom,
                        self.cell_constraint,
                        self.opt_cell,
                    ]
                )
            ],
        )
        self.set_title(0, "Cell optimization")

    def return_dict(self):
        sys_params = {"symmetry": self.cell_symmetry.value}
        if self.opt_cell.value:
            sys_params["cell_opt"] = ""
        if self.cell_constraint.value != "NONE":
            sys_params["cell_opt_constraint"] = self.cell_constraint.value
        if self.cell_freedom.value != "FREE":
            sys_params[self.cell_freedom.value.lower()] = ""

        return {"sys_params": sys_params}

    def traits_to_link(self):
        return ["details", "do_cell_opt"]


class DiagonalisationSmearingWidget(ipw.HBox):
    def __init__(self, **kwargs):
        self.enable_diagonalisation = ipw.Checkbox(
            value=False,
            description="Self-consistent diagonalisation",
            style={"description_width": "initial"},
            layout={"width": "240px"},
        )
        self.enable_diagonalisation.observe(
            self._observe_enable_diagonalisation, "value"
        )
        self.enable_diagonalisation.observe(self.enable_or_disable_widgets, "value")

        self.enable_smearing = ipw.ToggleButton(
            value=False,
            description="Enable Fermi-Dirac smearing",
            style={"description_width": "initial"},
            layout={"width": "450px"},
        )
        self.enable_smearing.observe(self.enable_or_disable_widgets, "value")

        self.smearing_temperature = ipw.FloatText(
            value=150.0,
            description="Temperature [K]",
            disabled=True,
            style={"description_width": "initial"},
            layout={"width": "200px"},
        )
        self.force_multiplicity = ipw.Checkbox(
            value=True, description="Force multiplicity", disabled=True
        )
        self.smearing_box = ipw.VBox(
            children=[
                self.enable_smearing,
                ipw.HBox(children=[self.smearing_temperature, self.force_multiplicity]),
            ]
        )

        super().__init__(
            children=[
                self.enable_diagonalisation,
            ],
            **kwargs,
        )

    def _observe_enable_diagonalisation(self, _=None):
        if self.enable_diagonalisation.value:
            self.children = [
                self.enable_diagonalisation,
                self.smearing_box,
            ]
        else:
            self.children = [self.enable_diagonalisation]

    def enable_or_disable_widgets(self, _=None):
        self.enable_smearing.disabled = not self.enable_diagonalisation.value
        self.smearing_temperature.disabled = not self.enable_smearing.value
        self.force_multiplicity.disabled = not self.enable_smearing.value

    @property
    def smearing_enabled(self):
        return self.enable_diagonalisation and self.enable_smearing.value


SECTIONS_TO_DISPLAY = {
    "None": [],
    "Wire": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
    ],
    "Bulk": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
        CellSectionWidget,
    ],
    "SlabXY": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
    ],
    "SlabYZ": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
    ],
    "SlabXZ": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
    ],
    "Molecule": [
        StructureInfoWidget,
        VdwSelectorWidget,
        UksSectionWidget,
        constraints.ConstraintsWidget,
        ForcePeriodicWidget,
    ],
    "Replica": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
        ReplicaWidget,
    ],
    "Neb": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
        NebWidget,
    ],
    "Phonons": [
        VdwSelectorWidget,
        UksSectionWidget,
        StructureInfoWidget,
        constraints.ConstraintsWidget,
        PhononsWidget,
    ],
}
