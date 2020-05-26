import numpy as np
from numpy.linalg import norm
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist
import scipy.stats
from scipy.constants import physical_constants
import itertools
from IPython.display import display, clear_output, HTML
import nglview
import ipywidgets as ipw

from apps.surfaces.widgets import slabs

from traitlets import HasTraits, Instance, Dict, Unicode, dlink, link, observe
from aiidalab_widgets_base import StructureManagerWidget
from apps.surfaces.widgets.ANALYZE_structure import StructureAnalyzer


class BuildSlab(ipw.VBox):
    structure = Instance(Atoms, allow_none=True)
    molecule = Instance(Atoms, allow_none=True)
    #sys_type = Unicode()
    details = Dict()
    slab = Instance(Atoms, allow_none=True)
    manager = Instance(StructureManagerWidget, allow_none=True)
    def __init__(self):

        self.drop_surface = ipw.Dropdown(description="Surface",
                                    options=["Au(111)","Ag(111)","Cu(111)","hBN","PdGa_A_Pd1","PdGa_A_Pd3"],
                                    value="Au(111)"
        #                           ,style=style, layout=layout
                                    )
        self.nx_slider = ipw.IntSlider(description="nx", min=1, max=60, continuous_update=False)
        self.ny_slider = ipw.IntSlider(description="ny", min=1, max=30, continuous_update=False)
        self.nz_slider = ipw.IntSlider(description="nz", min=1, max=10,value=4, continuous_update=False)
        self.create_bttn = ipw.Button(description="Add slab",disabled=True)
        self.create_bttn.on_click(self.create_slab)
        super().__init__(children=[self.drop_surface,ipw.HBox([self.nx_slider,self.ny_slider,self.nz_slider, self.create_bttn])])
        
    def create_slab(self, _=None):
        nx = self.nx_slider.value
        ny = self.ny_slider.value
        nz = self.nz_slider.value
        which_surf = self.drop_surface.value
        #inp_descr.value = orig_structure.description + "_" + which_surf + ".slab(%d,%d,%d)"%(nx,ny,nz)
        self.slab = slabs.prepare_slab(self._molecule, dx=0.0, dy=0.0, dz=0.0, phi=0.0, nx=nx, ny=ny, nz=nz, which_surf=which_surf)
        self.structure =   self._molecule + self.slab 


        
    @observe('molecule')
    def on_struct_change(self, change=None):
        """Selected molecule from structure."""
        #analyzer = StructureAnalyzer(who_called_me='BuildSlab',only_sys_type=True)
        #analyzer.structure = change['new']
        #details = analyzer.analyze()
        #print('observe moleucle in Build ')
        #if self.sys_type == 'Molecule':
        if self.details and self.details['system_type'] == 'Molecule':
            self.create_bttn.disabled = False
            nx, ny = slabs.guess_slab_size(self.molecule)
            self.nx_slider.value = nx
            self.ny_slider.value = ny
            self._molecule = change['new'].copy()
        else:
            self.create_bttn.disabled = True


    
    @observe('manager')
    def _change_manager(self, value):
        """Set structure manager trait."""
        manager = value['new']
        #print('manager change')
        if manager is None:
            return
        dlink((manager, 'structure'), (self, 'molecule'))
        dlink((self, 'structure'), (manager, 'structure'))
        