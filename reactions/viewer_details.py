from __future__ import print_function
from __future__ import absolute_import

import nglview
import ipywidgets as ipw
from IPython.display import display, clear_output

from ase import Atoms

import numpy as np

MOL_ASPECT  = 3.0
REST_ASPECT = 10.0

class ViewerDetails(ipw.VBox):
    
    def __init__(self, **kwargs):
        
        self.atoms = None
        self.details = None
        
        self.mol_inds = None
        self.rest_inds = None
        
        self._translate_i_glob_loc = None # global index -> (i_component, index)
        self._translate_i_loc_glob = None # (i_component, index) -> global index
        
        self.molecules_ase = None
        self.rest_ase = None
        
        self.selection = []
        
        self.viewer = nglview.NGLWidget()
        ##avoid center-on-click
        self.viewer.stage.set_parameters(mouse_preset='pymol')
        #self.viewer.stage.set_parameters(mouse_preset='coot')
       
        self.info_out = ipw.Output()
        
        children = [
            self.viewer,
            self.info_out
        ]

        super(ViewerDetails, self).__init__(children=children, **kwargs)
        
    def _on_atom_click(self, c):
        if 'atom1' not in self.viewer.picked.keys():
            return # did not click on atom
        elem = self.viewer.picked['atom1']['element']
        x = self.viewer.picked['atom1']['x']
        y = self.viewer.picked['atom1']['y']
        z = self.viewer.picked['atom1']['z']
        index = self.viewer.picked['atom1']['index']
        
        if index < len(self.molecules_ase):
            candidate = self.molecules_ase[index]
            global_i = self._translate_i_loc_glob[(0, index)]
            
            if not np.allclose(candidate.position, np.array([x, y, z]), atol=1e-2):
                candidate = self.rest_ase[index]
                global_i = self._translate_i_loc_glob[(1, index)]
        else:
            candidate = self.rest_ase[index]
            global_i = self._translate_i_loc_glob[(1, index)]
        
        if global_i in self.selection:
            self.selection.remove(global_i)
        else:
            self.selection.append(global_i)
            self.selection.sort()
        self.reset()
        self.highlight_atoms(self.selection, color='green', size=0.3, opacity=0.2)
        
        with self.info_out:
            clear_output()
            print("Atom: %s [%.3f %.3f %.3f], i=%d (starts from 1)" % (elem, x, y, z, global_i+1))
            print("Selection: [" + " ".join([str(x+1) for x in self.selection]) + "]")
        
    def _gen_translation_indexes(self):
        self._translate_i_glob_loc = {}
        self._translate_i_loc_glob = {}
        c_mol_i = 0
        c_rest_i = 0
        
        for i_m in self.mol_inds:
            self._translate_i_glob_loc[i_m] = (0, c_mol_i)
            self._translate_i_loc_glob[(0, c_mol_i)] = i_m
            c_mol_i += 1
            
        for i_r in self.rest_inds:
            self._translate_i_glob_loc[i_r] = (1, c_rest_i)
            self._translate_i_loc_glob[(1, c_rest_i)] = i_r
            c_rest_i += 1
    
    def _translate_glob_loc(self, indexes):
        mol_i = []
        rest_i = []
        for i_v in indexes:
            i_c, i_a = self._translate_i_glob_loc[i_v]
            if i_c == 0:
                mol_i.append(i_a)
            else:
                rest_i.append(i_a)
        return mol_i, rest_i
    
    def reset_selection(self):
        self.selection = []
        with self.info_out:
            clear_output()
            print("Selection: [" + " ".join([str(x+1) for x in self.selection]) + "]")
        
    def setup(self, atoms, details=None):
        
        self.atoms = atoms
        self.details = details
        
        # delete all old components
        while hasattr(self.viewer, "component_0"):
            self.viewer.component_0.clear_representations()
            cid = self.viewer.component_0.id
            self.viewer.remove_component(cid)
        
        if details is None:
            self.mol_inds = [] #list(np.arange(0, len(atoms)))
            if atoms is None:
                return
            else:
                self.rest_inds = list(np.arange(0, len(atoms))) # [] #default all big spheres
        else:
            if details['system_type']=='Bulk':
                self.mol_inds = [] 
                self.rest_inds = list(np.arange(0, len(atoms))) 
            elif details['system_type']=='Wire':
                self.mol_inds = list(np.arange(0, len(atoms))) 
                self.rest_inds = [] 
            else:
                self.mol_inds = [item for sublist in self.details['all_molecules'] for item in sublist]
                self.rest_inds = self.details['slabatoms']+self.details['bottom_H']+self.details['adatoms'] +self.details['unclassified']
        self._gen_translation_indexes() 
        
        #print('in view mol ',self.mol_inds)
        #print('in view rest ',self.rest_inds)
        if len(self.mol_inds) > 0:
            self.molecules_ase = self.atoms[self.mol_inds]
        else:
            self.molecules_ase=Atoms()
        if len(self.rest_inds) > 0:
            self.rest_ase = self.atoms[self.rest_inds]
        else:
            self.rest_ase=Atoms()

        # component 0: Molecule
        self.viewer.add_component(nglview.ASEStructure(self.molecules_ase), default_representation=False)
        self.viewer.add_ball_and_stick(aspectRatio=MOL_ASPECT, opacity=1.0,component=0)
        
        # component 1: Everything else
        self.viewer.add_component(nglview.ASEStructure(self.rest_ase), default_representation=False)
        self.viewer.add_ball_and_stick(aspectRatio=REST_ASPECT, opacity=1.0, component=1)
        
        self.viewer.add_unitcell()
        self.viewer.center()

        #viewer.component_0.add_ball_and_stick(aspectRatio=10.0, opacity=1.0)
        #for an in set(atoms.numbers):
        #    vdwr=vdw_radii[an]
        #    sel=[s[0] for s in np.argwhere(atoms.numbers==an)]
        #    viewer.add_ball_and_stick(selection=sel,aspectRatio=6.2*vdwr, opacity=1.0,component=0)

        # Orient camera to look from positive z
        cell_z = self.atoms.cell[2, 2]
        com = self.atoms.get_center_of_mass()
        def_orientation = self.viewer._camera_orientation
        top_z_orientation = [1.0, 0.0, 0.0, 0,
                             0.0, 1.0, 0.0, 0,
                             0.0, 0.0, -np.max([cell_z, 30.0]) , 0,
                             -com[0], -com[1], -com[2], 1]
        self.viewer._set_camera_orientation(top_z_orientation)
        
        self.viewer.observe(self._on_atom_click, names='picked')
    
    
    def reset(self):
        """
        Resets the representations of currently set up viewer instance
        """
            
        self.viewer.component_0.clear_representations()
        self.viewer.add_ball_and_stick(aspectRatio=MOL_ASPECT, opacity=1.0,component=0)
        
        self.viewer.component_1.clear_representations()
        self.viewer.add_ball_and_stick(aspectRatio=REST_ASPECT, opacity=1.0,component=1)
        
        self.viewer.add_unitcell()
        
    
    def highlight_atoms(self, global_i_list, color='red', size=0.2, opacity=0.6):
        
        if not hasattr(self.viewer, "component_0"):
            return
        
        mol_v_list, rest_v_list = self._translate_glob_loc(global_i_list)

        self.viewer.component_0.add_ball_and_stick(selection=mol_v_list, color=color, aspectRatio=MOL_ASPECT+size, opacity=opacity)
        self.viewer.component_1.add_ball_and_stick(selection=rest_v_list, color=color, aspectRatio=REST_ASPECT+size, opacity=opacity)

        
    def show_fixed(self, fixed_atoms_str):
        
        if not hasattr(self.viewer, "component_0"):
            return
        
        self.reset()
        
        if fixed_atoms_str == "":
            return
        
        bounds = np.array(fixed_atoms_str.split('..'), dtype=int)
        f_list = list(range(bounds[0]-1, bounds[1])) # the cp2k list is edge-inclusive!
        
        self.highlight_atoms(f_list, color='green', size=0.1, opacity=1.0)
        
        
    def visualize_extra(self, vis_list):
        """
        Two visualization options supported:
          int - defines the atom index to highlight
          xyz - defines a point in space
        """
        if not hasattr(self.viewer, "component_0"):
            return
        
        self.reset()
        
        if hasattr(self.viewer, "component_2"):
            self.viewer.component_2.clear_representations()
            self.viewer.component_2.remove_unitcell()
            cid = self.viewer.component_2.id
            self.viewer.remove_component(cid)
        
        if len(vis_list) > 0:

            vis_atoms  = [x for x in vis_list if isinstance(x, int)]
            vis_points = [x for x in vis_list if not isinstance(x, int)]
            
            if len(vis_atoms) != 0:
                
                self.highlight_atoms(vis_atoms, color='red', size=0.2, opacity=0.6)
                
            if len(vis_points) != 0:
                fake_atoms = Atoms('Xe'*len(vis_points), positions=vis_points)
                self.viewer.add_component(nglview.ASEStructure(fake_atoms), default_representation=False)
                self.viewer.component_2.add_ball_and_stick(color='blue', aspectRatio=3.1, opacity=0.7)
        
        
        

    