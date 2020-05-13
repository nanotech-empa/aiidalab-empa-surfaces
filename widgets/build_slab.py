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

from traitlets import HasTraits, Instance, Dict, observe



class BuildSlab(ipw.VBox):
    structure = Instance(Atoms, allow_none=True)
    on_substrate_structure = Instance(Atoms, allow_none=True)
    details = Dict()
    def __init__(self):

        self.drop_surface = ipw.Dropdown(description="Surface",
                                    options=["Au(111)","Ag(111)","Cu(111)","hBN","PdGa_A_Pd1","PdGa_A_Pd3"],
                                    value="Au(111)"
        #                           ,style=style, layout=layout
                                    )
        self.drop_surface.observe(self.on_nxyz_change, names='value')
        self.nx_slider = ipw.IntSlider(description="nx", min=1, max=60, continuous_update=False)
        self.nx_slider.observe(self.on_nxyz_change, names='value')
        self.ny_slider = ipw.IntSlider(description="ny", min=1, max=30, continuous_update=False)
        self.ny_slider.observe(self.on_nxyz_change, names='value')
        self.nz_slider = ipw.IntSlider(description="nz", min=1, max=10,value=4, continuous_update=False)
        self.nz_slider.observe(self.on_nxyz_change, names='value')

        display(ipw.VBox())        

        super().__init__(children=[self.drop_surface,ipw.HBox([self.nx_slider,self.ny_slider,self.nz_slider])])
        
    def  on_nxyz_change(self,c=None):
        print('inside on_xyz')
        if self.structure:
            nx = self.nx_slider.value
            ny = self.ny_slider.value
            nz = self.nz_slider.value
            which_surf = self.drop_surface.value
            #inp_descr.value = orig_structure.description + "_" + which_surf + ".slab(%d,%d,%d)"%(nx,ny,nz)
            mol = self.structure.copy()
            slab = slabs.prepare_slab(mol,dx=0.0,dy=0.0,dz=0.0,phi=0.0, nx=nx, ny=ny, nz=nz,which_surf=which_surf)
            atoms = mol + slab
            self.on_substrate_structure = atoms
    @observe('structure') ##selected molecule from structure selectore
    def on_struct_change(self,c=None):
        if self.structure:
            nx, ny = slabs.guess_slab_size(self.structure)
            self.nx_slider.value = nx
            self.ny_slider.value = ny
            #orig_structure = s
            self.on_nxyz_change()

