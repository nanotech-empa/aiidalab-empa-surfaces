import aiidalab_widgets_base as awb
import ase
import ipywidgets as ipw
import numpy as np
import traitlets as tr
from aiida_nanotech_empa.utils import common_utils

from . import slabs
from .analyze_structure import StructureAnalyzer


class BuildSlab(ipw.VBox):
    structure = tr.Instance(ase.Atoms, allow_none=True)
    molecule = tr.Instance(ase.Atoms, allow_none=True)
    details = tr.Dict()
    slab = tr.Instance(ase.Atoms, allow_none=True)
    manager = tr.Instance(awb.StructureManagerWidget, allow_none=True)

    def __init__(self, title=""):
        self.title = title
        self._molecule = None
        self.drop_surface = ipw.Dropdown(
            description="Surface",
            options=[
                "Au(111)",
                "Au(110)2x1",
                "Au(110)3x1",
                "Au(110)4x1",
                "Cu(110)O-2x1",
                "Ag(111)",
                "Cu(111)",
                "NaCl(100)",
                "hBN",
                "PdGa_A_Pd1",
                "PdGa_A_Pd3",
            ],
            value="Au(111)",
        )
        self.nx_slider = ipw.IntSlider(
            description="nx", min=1, max=60, continuous_update=False
        )
        self.ny_slider = ipw.IntSlider(
            description="ny", min=1, max=30, continuous_update=False
        )
        self.nz_slider = ipw.IntSlider(
            description="nz", min=1, max=10, value=4, continuous_update=False
        )

        def on_slab_select(c):
            if self._molecule:
                self.nx_slider.value, self.ny_slider.value = slabs.guess_slab_size(
                    self._molecule, self.drop_surface.value
                )
            elif (
                self.molecule
                and self.details
                and self.details["system_type"] == "Molecule"
            ):
                self.nx_slider.value, self.ny_slider.value = slabs.guess_slab_size(
                    self.molecule, self.drop_surface.value
                )

        self.drop_surface.observe(on_slab_select)

        self.create_bttn = ipw.Button(description="Add slab")
        self.create_bttn.on_click(self.create_slab)
        self.info = ipw.HTML("")
        super().__init__(
            children=[
                self.drop_surface,
                ipw.HBox(
                    [self.nx_slider, self.ny_slider, self.nz_slider, self.create_bttn]
                ),
                self.info,
            ]
        )

    def create_slab(self, _=None):
        """Create slab and remember the last molecule used."""
        sa = StructureAnalyzer()
        sa.structure = self.molecule
        self.info.value = ""

        # Remembering the last used molecule.
        if sa.details and sa.details["system_type"] == "Molecule":
            self._molecule = self.molecule
            self.info.value = """<span style="color:green;">Info:</span> The molecule was remembered and will be used for the further slab builds."""
        elif sa.details:
            self.info.value = """<span style="color:green;">Info:</span> Using previously remembered molecule."""
        if not self._molecule:
            self.info.value = (
                """<span style="color:#FFCC00;">Warning:</span> No molecule defined."""
            )
            return
        nx = self.nx_slider.value
        ny = self.ny_slider.value
        nz = self.nz_slider.value
        which_surf = self.drop_surface.value
        self.slab = slabs.prepare_slab(
            self._molecule,
            dx=0.0,
            dy=0.0,
            dz=0.0,
            phi=0.0,
            nx=nx,
            ny=ny,
            nz=nz,
            which_surf=which_surf,
        )
        self.structure = self._molecule + self.slab

    @tr.observe("molecule", "details")
    def on_struct_change(self, change=None):
        """Selected molecule from structure."""

        if self.molecule and self.details and self.details["system_type"] == "Molecule":
            nx, ny = slabs.guess_slab_size(self.molecule, self.drop_surface.value)
            self.nx_slider.value = nx
            self.ny_slider.value = ny


class InsertStructureWidget(ipw.HBox):
    structure = tr.Instance(ase.Atoms, allow_none=True)
    structure_to_insert = tr.Instance(ase.Atoms, allow_none=True)
    input_selection = tr.List(tr.Int())

    def __init__(self, title=""):
        self.title = title

        self.thumbnail = ipw.HTML()
        self.location = ipw.Text(
            description="Place to:", value="0.0 0.0 0.0", layout={"width": "initial"}
        )
        self.add_molecule_button = ipw.Button(description="Add molecule")
        self.add_molecule_button.on_click(self.add_molecule)
        self.importer = awb.StructureBrowserWidget(title="AiiDA database")
        tr.dlink(
            (self.importer, "structure"),
            (self, "structure_to_insert"),
            transform=lambda x: x.get_ase() if x else None,
        )
        self._status_message = awb.utils.StatusHTML()
        super().__init__(
            children=[
                self.thumbnail,
                ipw.VBox(
                    [
                        self.importer,
                        ipw.HBox(
                            [
                                self.location,
                                self.add_molecule_button,
                            ]
                        ),
                        self._status_message,
                    ]
                ),
            ]
        )

    @tr.validate("structure_to_insert")
    def _validate_structure_to_insert(self, proposal):
        structure = proposal["value"]
        structure.pbc = False
        structure.cell = None
        geom_center = np.mean(structure.get_positions(), axis=0)
        structure.translate(-geom_center)
        return structure

    @tr.observe("structure_to_insert")
    def _observe_structure_to_insert(self, _=None):
        thumbnail = common_utils.thumbnail(ase_struc=self.structure_to_insert)
        self.thumbnail.value = (
            f"""<img width="200px" src="data:image/png;base64,{thumbnail}" title="">"""
        )

    def add_molecule(self, _=None):
        """Add molecule to the structure."""
        insert_structure = self.structure_to_insert.copy()
        insert_structure.translate([float(x) for x in self.location.value.split()])
        self.structure = self.structure.copy() + insert_structure
        self.input_selection = list(
            range(len(self.structure) - len(insert_structure), len(self.structure))
        )
