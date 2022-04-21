import itertools

import ase.neighborlist
import ipywidgets as ipw
import nglview
import numpy as np
import scipy.stats
from aiidalab_widgets_base import StructureManagerWidget
from apps.surfaces.widgets import slabs
from apps.surfaces.widgets.ANALYZE_structure import StructureAnalyzer
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from IPython.display import HTML, clear_output, display
from numpy.linalg import norm
from scipy.constants import physical_constants
from traitlets import Dict, HasTraits, Instance, Unicode, dlink, link, observe


class BuildSlab(ipw.VBox):
    structure = Instance(Atoms, allow_none=True)
    molecule = Instance(Atoms, allow_none=True)
    details = Dict()
    slab = Instance(Atoms, allow_none=True)
    manager = Instance(StructureManagerWidget, allow_none=True)

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
        # Remembering the last used molecule
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

    @observe("molecule", "details")
    def on_struct_change(self, change=None):
        """Selected molecule from structure."""

        if self.molecule and self.details and self.details["system_type"] == "Molecule":
            nx, ny = slabs.guess_slab_size(self.molecule, self.drop_surface.value)
            self.nx_slider.value = nx
            self.ny_slider.value = ny
