import numpy as np
import nglview
from aiidalab_widgets_base.structures import StructureDataViewer
from apps.surfaces.widgets.ANALYZE_structure import StructureAnalyzer
from aiidalab_widgets_base.utils import string_range_to_list, list_to_string_range
from traitlets import  Dict, Unicode, observe

## custom set for visualization
def default_vis_func(structure):
    if structure is None:
        return {},{}
    an=StructureAnalyzer()
    #an.set_trait('structure',structure)
    an.structure = structure
    details=an.details
    if not details:
        return {},{}
    if details['system_type']=='SlabXY':
        all_mol = [item for sublist in details['all_molecules'] for item in sublist]
        the_rest = list(set(range(details['numatoms']))-set(all_mol))
        vis_dict={ ## rep 0 must be the substrate otherwise in case of no moleucle it crashes
            0 : {
                'ids' : list_to_string_range(the_rest,shift=0),
                'aspectRatio' : 10 , 
                'highlight_aspectRatio' : 10.1 , 
                'highlight_color' : 'green',
                'highlight_opacity' :0.6,
                'name' : 'substrate',
                'type' : 'ball+stick'
            },  
            1 : {
                'ids' : list_to_string_range(all_mol,shift=0),
                'aspectRatio' : 2.5 , 
                'highlight_aspectRatio' : 2.6 , 
                'highlight_color' : 'red',
                'highlight_opacity' : 0.6,
                'name' : 'molecule',
                'type' : 'ball+stick'
            },            
        }
    elif details['system_type']=='Molecule':
        the_rest = list(range(details['numatoms']))
        vis_dict={
            0 : {
                'ids' : list_to_string_range(the_rest,shift=0),
                'aspectRatio' : 2.5 , 
                'highlight_aspectRatio' : 2.6 , 
                'highlight_color' : 'red',
                'highlight_opacity' : 0.6,
                'name' : 'molecule',
                'type' : 'ball+stick'
            }
        }
    else :
        the_rest = list(range(details['numatoms']))
        vis_dict={
            0 : {
                'ids' : list_to_string_range(the_rest,shift=0),
                'aspectRatio' : 3 , 
                'highlight_aspectRatio' : 3.1 , 
                'highlight_color' : 'red',
                'highlight_opacity' : 0.6,
                'name' : 'molecule',
                'type' : 'ball+stick'
            }
        }
    return vis_dict ,details
class EmpaStructureViewer(StructureDataViewer):
    DEFAULT_SELECTION_OPACITY = 0.2
    DEFAULT_SELECTION_RADIUS = 6
    DEFAULT_SELECTION_COLOR = 'green'
    vis_dict = Dict()
    details = Dict()
    sys_type = Unicode()
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
        self.selection_dict = {} ## needed to display info about selected atoms e.g. distance, angle..
        self.selection_info = ''  ## needed to display info about selected atoms e.g. distance, angle..
        self.vis_func = vis_func
        super().__init__(**kwargs)
        
       

    def _gen_translation_indexes(self):
        """Transfromation of indexes in case of multiple representations
        dictionaries for  back and forth transformations."""
        
        self._translate_i_glob_loc = {}
        self._translate_i_loc_glob = {}
        for component in range(len(self.vis_dict.keys())):
            comp_i = 0
            ids = list(string_range_to_list(self.vis_dict[component]['ids'],shift=0)[0])
            for i_g in ids: 
                self._translate_i_glob_loc[i_g] = (component, comp_i)
                self._translate_i_loc_glob[(component, comp_i)] = i_g
                comp_i += 1
      
    def _translate_glob_loc(self, indexes):
        """From global index to indexes of different components."""
        all_comp=[list() for i in range(len(self.vis_dict.keys()))]
        for i_g in indexes:            
            i_c, i_a = self._translate_i_glob_loc[i_g]
            all_comp[i_c].append(i_a)

        return all_comp

    def _on_atom_click(self, change=None):  # pylint:disable=unused-argument
        """Update selection when clicked on atom."""
        if 'atom1' not in self._viewer.picked.keys():
            return  # did not click on atom
        index = self._viewer.picked['atom1']['index']
        
        selection = self.selection.copy()

        if self.vis_dict:
            component = self._viewer.picked['component']
            index = self._translate_i_loc_glob[(component,index)]        
        
        if selection:
            if index not in selection:
                self.selection_dict[index] = self._viewer.picked['atom1']
                selection.append(index)
            else:                
                selection.remove(index)
                self.selection_dict.pop(index, None)                
        else:
            self.selection_dict[index] = self._viewer.picked['atom1']
            selection = [index]        
            
        self.selection = selection        

        return

        ## first update selection_dict then update selection
        selection_tmp = self.selection.difference({index})

        self.selection = selection_tmp

    def highlight_atoms(self,
                        vis_list,
                        color=DEFAULT_SELECTION_COLOR,
                        size=DEFAULT_SELECTION_RADIUS,
                        opacity=DEFAULT_SELECTION_OPACITY):
        """Highlighting atoms according to the provided list."""
        
        if not hasattr(self._viewer, "component_0"):
            return

        if self.vis_dict is None :
                self._viewer._remove_representations_by_name(repr_name='selected_atoms')  # pylint:disable=protected-access
                self._viewer.add_ball_and_stick(  # pylint:disable=no-member
                name="selected_atoms",
                selection=list() if vis_list is None else vis_list,
                color=color,
                aspectRatio=size,
                opacity=opacity)
        else:
            
            ncomponents=len(self.vis_dict.keys())
            for component in range(ncomponents):
                name = 'highlight_'+self.vis_dict[component]['name']
                self._viewer._remove_representations_by_name(repr_name=name,component=component)
                color = self.vis_dict[component]['highlight_color']
                aspectRatio = self.vis_dict[component]['highlight_aspectRatio']
                opacity = self.vis_dict[component]['highlight_opacity']
                if vis_list is None:                   
                    self._viewer.add_ball_and_stick(name=name,
                                                    selection=list(),
                                                    color=color,
                                                    aspectRatio=aspectRatio,
                                                    opacity=opacity,
                                                    component=component
                                                   )                    
                else:
                    all_comp = self._translate_glob_loc(vis_list)
                    selection= all_comp[component]
                    self._viewer.add_ball_and_stick(name=name,
                                                    selection=selection,
                                                    color=color,
                                                    aspectRatio=aspectRatio,
                                                    opacity=opacity,
                                                    component=component
                                                   )      

    def custom_vis(self, c=None):
        """use cutom visualization if available and if compatible"""
        if self.vis_func is None:
            return
                
        else:   
            vis_dict,self.details = self.vis_func(self.structure) # self.vis_func(self.structure)

            ## keys must be integers 0,1,2,...
            if vis_dict:
                self.sys_type = self.details['system_type']
                types = set([vis_dict[i]['type'] for i in range(len(vis_dict)) ])
                    ## only {'ball+stick','licorice'} implemented
                    ## atom pick very difficult with 'licorice' and 'hyperball'
                if not types.issubset({'ball+stick','licorice','hyperball'}): 
                    print('type unknown')
                    self.vis_func = None
                    return
                #except:
                #    self.vis_func = None
                #    return

                # delete all old components
                while hasattr(self._viewer, "component_0"):
                    self._viewer.component_0.clear_representations()
                    cid = self._viewer.component_0.id
                    self._viewer.remove_component(cid)            

                self.vis_dict = vis_dict
                for component in range(len(self.vis_dict)):

                    rep_indexes=list(string_range_to_list(self.vis_dict[component]['ids'],shift=0)[0])
                    if rep_indexes:
                        mol = self.structure[rep_indexes]

                        self._viewer.add_component(nglview.ASEStructure(mol), default_representation=False)

                        if self.vis_dict[component]['type'] == 'ball+stick':
                            aspectRatio = self.vis_dict[component]['aspectRatio']
                            self._viewer.add_ball_and_stick(aspectRatio=aspectRatio, opacity=1.0,component=component)
                        elif self.vis_dict[component]['type'] == 'licorice':
                            self._viewer.add_licorice(opacity=1.0,component=component)
                        elif self.vis_dict[component]['type'] == 'hyperball':
                            self._viewer.add_hyperball(opacity=1.0,component=component)                    
            self._gen_translation_indexes()
            self._viewer.add_unitcell()
            self._viewer.center()
    
    def orient_z_up(self, _=None):
        try:
            cell_z = self.structure.cell[2, 2]
            com = self.structure.get_center_of_mass()
            def_orientation = self._viewer._camera_orientation
            top_z_orientation = [1.0, 0.0, 0.0, 0,
                                 0.0, 1.0, 0.0, 0,
                                 0.0, 0.0, -np.max([cell_z, 30.0]) , 0,
                                 -com[0], -com[1], -com[2], 1]
            self._viewer._set_camera_orientation(top_z_orientation)  
        except:
            return

    @observe('structure')
    def _update_displayed_structure(self, change):
        super()._update_displayed_structure(change=change)
        if(self.vis_func) : self.custom_vis()
        self.orient_z_up()
    
#    @observe('displayed_structure')
#    def _update_structure_viewer(self, change):
#        super()._update_structure_viewer(change)
#        if(self.vis_func) : self.custom_vis()
#