import traitlets as tr
import ipywidgets as ipw
from ase import Atoms
from aiidalab_widgets_base import viewers as awb_viewers
from aiidalab_widgets_base.utils import list_to_string_range

from .analyze_structure import StructureAnalyzer


def _unique_sorted(indices, natoms):
    return sorted({int(index) for index in indices if 0 <= int(index) < natoms})


def molecule_and_non_molecule_atoms(details, natoms):
    """Return molecule and non-molecule atom indices from structure details."""
    molecules_atoms = _unique_sorted(
        (index for molecule in details.get("all_molecules", []) for index in molecule),
        natoms,
    )
    non_molecule_atoms = [
        index for index in range(natoms) if index not in set(molecules_atoms)
    ]
    return molecules_atoms, non_molecule_atoms


def _make_representation(
    viewer,
    name,
    indices,
    representation_type,
    *,
    style_id=None,
    deletable=True,
    atom_show_threshold=1,
):
    representation = awb_viewers.NglViewerRepresentation(
        style_id=style_id or f"{viewer.REPRESENTATION_PREFIX}{name}",
        indices=indices,
        deletable=deletable,
        atom_show_threshold=atom_show_threshold,
    )
    representation.type.value = representation_type
    representation.size.value = 3
    representation.color.value = "element"
    representation.viewer_class = viewer
    return representation


def reset_default_representation(viewer):
    """Reset a StructureDataViewer to one default all-atom representation."""
    structure = viewer.structure
    if not isinstance(structure, Atoms):
        return

    default_representation = _make_representation(
        viewer,
        "default",
        list(range(len(structure))),
        "ball+stick",
        style_id=viewer.DEFAULT_REPRESENTATION,
        deletable=False,
        atom_show_threshold=0,
    )
    viewer._all_representations = [default_representation]

    for array in list(structure.arrays):
        if array.startswith(viewer.REPRESENTATION_PREFIX):
            del structure.arrays[array]

    viewer._apply_representations()


def apply_auto_representations(viewer, details):
    """Apply molecule/rest representations to a StructureDataViewer."""
    structure = viewer.structure
    if not isinstance(structure, Atoms):
        raise ValueError("No structure available for automatic representations.")

    molecules_atoms, non_molecule_atoms = molecule_and_non_molecule_atoms(
        details, len(structure)
    )
    if not molecules_atoms:
        raise ValueError("No molecule atoms were identified in the structure.")

    representations = [
        _make_representation(
            viewer,
            "molecules",
            molecules_atoms,
            "ball+stick",
            style_id=viewer.DEFAULT_REPRESENTATION,
            deletable=False,
            atom_show_threshold=0,
        )
    ]
    if non_molecule_atoms:
        representations.append(
            _make_representation(
                viewer,
                "non_molecule",
                non_molecule_atoms,
                "spacefill",
            )
        )

    viewer._all_representations = representations
    for representation in viewer._all_representations:
        representation.viewer_class = viewer

    representation_ids = {representation.style_id for representation in representations}
    for array in list(structure.arrays):
        if (
            array.startswith(viewer.REPRESENTATION_PREFIX)
            and array not in representation_ids
        ):
            del structure.arrays[array]

    viewer._apply_representations()

    return molecules_atoms, non_molecule_atoms


class AutoRepresentationWidget(ipw.VBox):
    """Button that assigns molecule/rest representations in a structure viewer."""

    structure = tr.Instance(Atoms, allow_none=True)
    details = tr.Dict(default_value={})

    def __init__(self, viewer, **kwargs):
        self.viewer = viewer
        self.button = ipw.Button(
            description="AutoRep",
            tooltip=(
                "Represent molecule atoms as ball+stick and all other atoms as "
                "spacefill."
            ),
            icon="magic",
        )
        self.button.on_click(self.apply)
        self.status = ipw.HTML()
        super().__init__(children=[self.button, self.status], **kwargs)
        self.observe(self._observe_structure, names="structure")

    def _observe_structure(self, _=None):
        if isinstance(getattr(self.viewer, "structure", None), Atoms):
            reset_default_representation(self.viewer)
        self.status.value = ""

    def _details(self):
        if self.details:
            return self.details
        structure = self.structure or getattr(self.viewer, "structure", None)
        if structure is None:
            return {}

        analyzer = StructureAnalyzer()
        analyzer.structure = structure.copy()
        return analyzer.details

    def apply(self, _=None):
        try:
            molecules_atoms, non_molecule_atoms = apply_auto_representations(
                self.viewer,
                self._details(),
            )
        except ValueError as exc:
            self.status.value = f'<span style="color:red;">AutoRep: {exc}</span>'
            return

        molecule_range = list_to_string_range(molecules_atoms, shift=1)
        non_molecule_range = list_to_string_range(non_molecule_atoms, shift=1)
        self.status.value = (
            '<span style="color:green;">AutoRep:</span> '
            f"molecule atoms {molecule_range}: ball+stick; "
            f"non-molecule atoms {non_molecule_range or 'none'}: spacefill."
        )
