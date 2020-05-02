from apps.surfaces.widgets.analyze_structure import mol_ids_range
from apps.surfaces.widgets import analyze_structure


from datetime import datetime

from ase import Atom, Atoms

from apps.surfaces.widgets.metadata import MetadataWidget

from aiida_cp2k.workchains.base import Cp2kBaseWorkChain


import ipywidgets as ipw
from IPython.display import display, clear_output

from collections import OrderedDict

from traitlets import Instance, Int, List, Set, Dict, Unicode, Union, link, default, observe, validate


style = {'description_width': '120px'}
layout = {'width': '70%'}
layout2 = {'width': '35%'}
#FUNCTION_TYPE = type(lambda c: c)
 
WIDGETS_ENABLED = {
    'None'     : [],
    'SlabXY'   : ['fixed_atoms','calc_type','vdw_switch','convergence','cell'],
    'Bulk'     : ['vdw_switch','convergence','cell'],
    'Molecule' : ['vdw_switch','convergence','cell']
}
CENTER_COORD = {
    'None'                 : 'False',
    'Bulk'                 : 'False',
    'SlabXY'               : 'False',
    'Molecule'             : 'True'
}


class InputDetails(ipw.VBox):
    details = Dict()
    to_fix = List()

    def __init__(self,):
        """
        Arguments:
            sections(list): list of tuples each containing the displayed name of an input section and the
                section object. Each object should containt 'structure' trait pointing to the imported
                structure. The trait will be linked to 'structure' trait of this class.
        """

        # Displaying input sections.
        self.output = ipw.Output()
        self.displayed_sections = []
            
        super().__init__(children=[self.output])

    @observe('details')
    def _observe_details(self, _=None):
        with self.output:
            clear_output()
            
#            if self.sections is None:
            if  self.details:
                sys_type =  self.details['system_type']    
            else:
                sys_type = 'None'

            self.displayed_sections = []
            for sec in SECTIONS_TO_DISPLAY[sys_type]:
                section = sec()
                section.manager = self
                self.displayed_sections.append(section)
                
            display(ipw.VBox(self.displayed_sections))
    
    def return_final_dictionary(self):
        final_dictionary = {}
        for section in self.displayed_sections:
            final_dictionary.update(section.return_dict())
        return final_dictionary


class PlainInputDetails(ipw.VBox):
    #structure = Instance(Atoms, allow_none=True)
    details = Dict()
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self): 
        ## PLAIN TEXT INPUT
        self.plain_input=ipw.Textarea(value='', disabled=False, layout={'width': '60%'})
        self.plain_input_accordion = ipw.Accordion(selected_index=None)
        self.plain_input_accordion.children=[self.plain_input]
        self.plain_input_accordion.set_title(0,'plain input')

        ## VALIDATE AND CREATE INPUT
        create_input=ipw.Button(description='create input', layout={'width': '10%'})
       
    
        ## CREATE PLAIN INPUT
        def on_create_input_btn_click(c):
            #GET VALUES FROM ALL SECTIONS
            plain_input.value = 'input created'

        create_input.on_click(on_create_input_btn_click)
    ## END CREATE PLAIN INPUT        
        
        super().__init__(children=[self.plain_input])
        
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))

class GwDetails(ipw.VBox):
    details = Dict()
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self):    
        #### GW

        self.gw_type_btn = ipw.RadioButtons(description='GW type',
            options=['GW', 'GW-LS','GW-IC'], value='GW',disabled=False
        )
        self.max_force = ipw.BoundedFloatText(descritpion='MAX FORCE',value=1e-4, min=1e-4, 
                                   max=1e-3, step=1e-4,style=style, layout=layout2)
       
        def button_click():
            visualize_widgets=[self.gw_type_btn] 
            self.max_force.observe()
            
        super().__init__(children=visualize_widgets)  
    
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))

class VdwSelectorWidget(ipw.ToggleButton):
    details = Dict()
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self):
        super().__init__(value=True, description='Dispersion Corrections', tooltip='VDW_POTENTIAL', style={'description_width': '120px'})
    
    def return_dict(self):
        return {'vdw_switch': self.value}

    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
    

class UksSectionWidget(ipw.VBox):
    details = Dict()
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self):
                #### UKS
        self.multiplicity = ipw.IntText(value=0,placeholder='leave 0 for RKS',
                                           description='MULTIPLICITY',
                                           style=style, layout=layout)
        self.spin_u = ipw.Text(placeholder='1..10 15',
                                            description='IDs atoms spin UP',
                                            style=style, layout={'width': '60%'})

        self.spin_d = ipw.Text(placeholder='1..10 15',
                                            description='IDs atoms spin DOWN',
                                            style=style, layout={'width': '60%'})
        self.charge = ipw.IntText(value=0,
                                 description='net charge',
                                 style=style, layout=layout)

        self.uks = ipw.Accordion(selected_index=None)
        self.uks.children = [ipw.VBox([self.multiplicity, self.spin_u, self.spin_d, self.charge])]
        self.uks.set_title(0,'RKS/UKS')
        super().__init__(children = [self.uks])
        
    def return_dict(self):
        return {
            'multiplicity' : self.multiplicity.value,
            'spin_u'       : self.spin_u.value,
            'spin_d'       : self.spin_d.value,
            'charge'       : self.charge.value,
        }
    
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
    
class MixedDftWidget(ipw.ToggleButtons):
    details = Dict()
    to_fix = List()
    manager = Instance(InputDetails, allow_none=True)

    def __init__(self,):
        
        super().__init__(options = ['Mixed DFTB', 'Mixed DFT', 'Full DFT'],
                         description = 'Calculation Type', 
                         value = 'Full DFT',
                         tooltip = 'Active: DFT, Inactive: DFTB', 
                         style = {'description_width': '120px'})
        
        self.observe(self.update_list_fixed, 'value')
        
    def return_dict(self):
        return {'calc_type': [self.value]}  
    
    #self.observe()
    def update_list_fixed(self,c=None):
        if self.details:
            if 'Slab' in self.details['system_type']:
                if self.value == 'Full DFT':
                    self.to_fix=[i for i in self.details['bottom_H'] + 
                    self.details['slab_layers'][0] +
                    self.details['slab_layers'][1]]
                else:
                    self.to_fix=self.details['bottom_H'] + self.details['slabatoms']        
    

    
#    @observe('details')
#    def _observe_details(self, _=None):
#        self.update_list_fixed()
#        print('mdft ob det',self.value,mol_ids_range(self.to_fix),datetime.now().strftime("%H:%M:%S"))
    
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
            link((self.manager, 'to_fix'), (self, 'to_fix'))
            self.update_list_fixed()
        #print('m obmanager',self.value,mol_ids_range(self.to_fix),datetime.now().strftime("%H:%M:%S"))
                
class FixedAtomsWidget(ipw.Text):
    details = Dict()
    to_fix = List()
    manager = Instance(InputDetails, allow_none=True)
    
    def __init__(self,):
        #self.value = mol_ids_range(self.to_fix)
        super().__init__(placeholder='1..10',
                         value = mol_ids_range(self.to_fix),
                                description='Fixed Atoms',
                                style=style, layout={'width': '60%'})
        
    def return_dict(self):
        return {'fixed_atoms': self.value}
    
    @observe('to_fix')
    def _observe_to_fix(self, _=None):
        self.value = mol_ids_range(self.to_fix)
            
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
            link((self.manager, 'to_fix'), (self, 'to_fix'))
            self.value = mol_ids_range(self.to_fix)

            
            
class CellSectionWidget(ipw.VBox):
    details = Dict()
    to_fix = List()
    manager = Instance(InputDetails, allow_none=True)
    
    def __init__(self):
    
        self.periodic = ipw.Dropdown(description='PBC',options=['XYZ','NONE', 'X','XY',
                                              'XZ','Y','YZ',
                                              'Z'],
                                       value='XYZ',
                                       style=style, layout=layout2) 


        self.poisson_solver = ipw.Dropdown(description='Poisson solver',
                                           options=['MT','PERIODIC', 'ANALYTIC','IMPLICIT',
                                           'MULTIPOLE','WAVELET'],
                                           value='PERIODIC',
                                           style=style, layout=layout2)
        self.cell_sym  = ipw.Dropdown(description='symmetry',
                                      options=['CUBIC','HEXAGONL', 'MONOCLINIC','NONE',
                                               'ORTHORHOMBIC','RHOMBOHEDRAL','TETRAGONAL_AB',
                                               'TETRAGONAL_AC','TETRAGONAL_BC','TRICLINIC'],
                                      value='ORTHORHOMBIC',
                                      style=style, layout=layout)

        self.cell = ipw.Text(description='cell size',
                            style=style, layout={'width': '60%'})
        
        self.center_coordinates = ipw.RadioButtons(description='center coordinates',
                                                   options=['False', 'True'],
                                                   value='False',
                                                   disabled=False)


        self.cell_spec = ipw.Accordion(selected_index=None)
        self.cell_spec.children = [ipw.VBox([self.periodic, self.poisson_solver, self.cell_sym,
                                            self.cell, self.center_coordinates ])]
        self.cell_spec.set_title(0,'CELL/PBC details')  
        
        super().__init__(children = [self.cell_spec])
        
    def return_dict(self):
        return {
            'periodic'           : self.periodic.value,
            'poisson_solver'     : self.poisson_solver.value,
            'cell_sym'           : self.cell_sym.value,
            'cell'               : self.cell.value,
            'center_coordinates' : self.center_coordinates.value
        }
    
    @observe('details')
    def _observe_details(self, _=None):
        if self.details :
            self.center_coordinates.value = CENTER_COORD[self.details['system_type']]
        
    
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))        
            
SECTIONS_TO_DISPLAY = {
        'None'     : [],
        'Bulk'     : [VdwSelectorWidget],
        'SlabXY'   : [VdwSelectorWidget, UksSectionWidget, 
                      MixedDftWidget, FixedAtomsWidget,CellSectionWidget],
        'Molecule' : [MixedDftWidget]
    }
