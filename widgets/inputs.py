from apps.surfaces.widgets.analyze_structure import mol_ids_range
from apps.surfaces.widgets import analyze_structure
from apps.surfaces.widgets.cp2k_input_validity import validate_input

from datetime import datetime

from ase import Atom, Atoms

from aiida.orm import Code

from apps.surfaces.widgets.metadata import MetadataWidget

from aiida_cp2k.workchains.base import Cp2kBaseWorkChain

from apps.surfaces.widgets.cp2k2dict import CP2K2DICT
from aiida_cp2k.utils import Cp2kInput

from apps.surfaces.widgets.get_cp2k_input import Get_CP2K_Input

import ipywidgets as ipw
from IPython.display import display, clear_output

from collections import OrderedDict

from traitlets import Instance, Int, List, Set, Dict, Unicode, Union, link, default, observe, validate


STYLE = {'description_width': '120px'}
LAYOUT = {'width': '70%'}
LAYOUT2 = {'width': '35%'}
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
    selected_code = Union([Unicode(), Instance(Code)], allow_none=True)
    details = Dict()
    to_fix = List()
    calc_type = Unicode()

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
                
            self.plain_input=ipw.Textarea(value='', disabled=False, layout={'width': '60%'})
            self.plain_input_accordion = ipw.Accordion(selected_index=None)
            self.plain_input_accordion.children=[self.plain_input]
            self.plain_input_accordion.set_title(0,'plain input')
              
            self.displayed_sections = []
            for sec in SECTIONS_TO_DISPLAY[sys_type]:
                section = sec()
                section.manager = self
                self.displayed_sections.append(section)    
            display(ipw.VBox(self.displayed_sections + [self.plain_input_accordion]))

    def create_plain_input(self):
        inp_dict = Get_CP2K_Input(input_dict = self.final_dictionary).inp
        inp_plain = Cp2kInput(inp_dict)
        self.plain_input.value = inp_plain.render()
        #return CP2K2DICT(input_lines = self.plain_input.value)         
        
    def return_final_dictionary(self):
        self.final_dictionary = {}
        
        ## PUT LIST OF ELEMENTS IN DICTIONARY
        self.final_dictionary['elements']=self.details['all_elements']
        
        ## RETRIEVE ALL WIDGET VALUES
        for section in self.displayed_sections:
            to_add = section.return_dict()
            if to_add : self.final_dictionary.update(to_add)  
        
        ## DECIDE WHICH KIND OF WORKCHAIN
        
        ## SLAB
        if self.details['system_type'] == 'SlabXY':
            self.final_dictionary.update({'workchain' : 'SlabGeoOptWorkChain'})
            ## IN CASE MIXED DFT FOR SLAB IDENTIFY MOLECULE
            if self.final_dictionary['calc_type'] != 'Full DFT':
                self.final_dictionary['first_slab_atom'] = min(self.details['bottom_H'] +
                                                               self.details['slabatoms']) + 1
                self.final_dictionary['last_slab_atom']  = max(self.details['bottom_H'] +
                                                               self.details['slabatoms']) + 1
        ## MOLECULE   
        elif self.details['system_type'] == 'Molecule' :
            self.final_dictionary.update({'workchain' : 'MoleculeOptWorkChain'})
            
        ## BULK
        elif self.details['system_type'] == 'Bulk' :
            if self.final_dictionary['opt_cell']:
                self.final_dictionary.update({'workchain' : 'CellOptWorkChain'})
            else:
                self.final_dictionary.update({'workchain' : 'BulkOptWorkChain'})
                
        ## CHECK input validity
        can_submit,error_msg=validate_input(self.details,self.final_dictionary)
                
        ## CREATE PLAIN INPUT  
        if can_submit :
            self.create_plain_input()        
        
        ## RETURN DICT of widgets details
        return  can_submit,error_msg, self.final_dictionary


class ConvergenceDetailsWidget(ipw.Accordion):
    details = Dict()
    calc_type = Unicode()
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self):    
        #### GW
        self.max_force = ipw.FloatText(descritpion='MAX FORCE',value=1e-4,
                                       style=STYLE, layout=LAYOUT2)
        self.mgrid_cutoff = ipw.IntText(descritpion='MGRID CUTOFF',value=600,
                                        style=STYLE, layout=LAYOUT2)

        
        self.set_title(0,'Convergence parameters')    
        super().__init__(selected_index=None)  

    def return_dict(self):
            if self.calc_type == 'Mixed DFTB':
                return {'max_force' : self.max_force.value }
            else:
                return {'max_force' : self.max_force.value , 'mgrid_cutoff' : self.mgrid_cutoff.value}

            
    def widgets_to_show(self):
        if self.calc_type == 'Mixed DFTB':
            self.children = [ipw.VBox([self.max_force])]
        else:
            self.children = [ipw.VBox([self.max_force,self.mgrid_cutoff])]        

    @observe('calc_type')
    def _observe_calc_type(self, _=None):
            self.widgets_to_show()
            
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
            link((self.manager, 'calc_type'), (self, 'calc_type'))
            self.widgets_to_show()
            
            
class VdwSelectorWidget(ipw.ToggleButton):
    details = Dict()
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self):
        super().__init__(value=True, description='Dispersion Corrections', 
                         tooltip='VDW_POTENTIAL', style={'description_width': '120px'})
    
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
                                           style=STYLE, layout=LAYOUT)
        self.spin_u = ipw.Text(placeholder='1..10 15',
                                            description='IDs atoms spin UP',
                                            style=STYLE, layout={'width': '60%'})

        self.spin_d = ipw.Text(placeholder='1..10 15',
                                            description='IDs atoms spin DOWN',
                                            style=STYLE, layout={'width': '60%'})
        self.charge = ipw.IntText(value=0,
                                 description='net charge',
                                 style=STYLE, layout=LAYOUT)

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
    calc_type = Unicode()
    manager = Instance(InputDetails, allow_none=True)

    def __init__(self,):
        
        super().__init__(options = ['Mixed DFTB', 'Mixed DFT', 'Full DFT'],
                         description = 'Calculation Type', 
                         value = 'Full DFT',
                         tooltip = 'Active: DFT, Inactive: DFTB', 
                         style = {'description_width': '120px'})
        
        self.observe(self.update_list_fixed, 'value')
        
    def return_dict(self):
        return {'calc_type': self.value}  
    
    #self.observe()
    def update_list_fixed(self,c=None):
        self.calc_type = self.value
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
            link((self.manager, 'calc_type'), (self, 'calc_type'))
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
                                style=STYLE, layout={'width': '60%'})
        
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
    manager = Instance(InputDetails, allow_none=True)
    
    def __init__(self):
        
    
        self.periodic = ipw.Dropdown(description='PBC',options=['XYZ','NONE', 'X','XY',
                                              'XZ','Y','YZ',
                                              'Z'],
                                       value='XYZ',
                                       style=STYLE, layout=LAYOUT2) 


        self.poisson_solver = ipw.Dropdown(description='Poisson solver',
                                           options=['MT','PERIODIC', 'ANALYTIC','IMPLICIT',
                                           'MULTIPOLE','WAVELET'],
                                           value='PERIODIC',
                                           style=STYLE, layout=LAYOUT2)
        self.cell_sym  = ipw.Dropdown(description='symmetry',
                                      options=['CUBIC','HEXAGONL', 'MONOCLINIC','NONE',
                                               'ORTHORHOMBIC','RHOMBOHEDRAL','TETRAGONAL_AB',
                                               'TETRAGONAL_AC','TETRAGONAL_BC','TRICLINIC'],
                                      value='ORTHORHOMBIC',
                                      style=STYLE, layout=LAYOUT)

        self.cell = ipw.Text(description='cell size',
                            style=STYLE, layout={'width': '60%'})
        
        self.center_coordinates = ipw.RadioButtons(description='center coordinates',
                                                   options=['False', 'True'],
                                                   value='True',
                                                   disabled=False)
        self.opt_cell = ipw.ToggleButton(value=False, description='Optimize cell',
                                         style={'description_width': '120px'})

        self.cell_cases = {
            'Bulk'                 : [
                ('cell_sym', self.cell_sym),
                ('cell', self.cell),
                ('opt_cell', self.opt_cell)
            ],
            'SlabXY'               : [
                ('periodic', self.periodic),
                ('poisson_solver', self.poisson_solver),
                ('cell', self.cell)
            ],
            'Molecule'             :  [
                ('periodic', self.periodic),
                ('poisson_solver', self.poisson_solver),
                ('cell', self.cell),
                ('center_coordinates', self.center_coordinates)
            ]
        }        

        self.cell_spec = ipw.Accordion(selected_index=None) 
        
        super().__init__(children = [self.cell_spec])
        
    def return_dict(self):
            to_return = {}
            for i in self.cell_cases[self.details['system_type']]:
                to_return.update({i[0] : i[1].value})
            return to_return
        
        
    
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
            self.cell.value = self.details['cell']
            self.cell_spec.children = [ipw.VBox([i[1] for i in self.cell_cases[self.details['system_type']]])]
            self.cell_spec.set_title(0,'CELL/PBC details')
            
                
            

class MetadataWidget(ipw.VBox):
    """Setup metadata for an AiiDA process."""
    
    details = Dict()
    selected_code = Union([Unicode(), Instance(Code)], allow_none=True)
    manager = Instance(InputDetails, allow_none=True)    

    def __init__(self):
        """ Metadata widget to generate metadata"""

        self.walltime_d = ipw.IntText(value=0,
                                      description='d:',
                                      style={'description_width': 'initial'},
                                      layout={'width': 'initial'})

        self.walltime_h = ipw.IntText(value=24,
                                      description='h:',
                                      style={'description_width': 'initial'},
                                      layout={'width': 'initial'})

        self.walltime_m = ipw.IntText(value=0,
                                      description='m:',
                                      style={'description_width': 'initial'},
                                      layout={'width': 'initial'})
        
        self.num_machines = ipw.IntText(value=1, description='# Nodes', style=STYLE, layout=LAYOUT2)

        self.num_mpiprocs_per_machine = ipw.IntText(value=12, description='# Tasks', style=STYLE, layout=LAYOUT2)

        self.num_cores_per_mpiproc = ipw.IntText(value=1, description='# Threads', style=STYLE, layout=LAYOUT2)

        children = [
            self.num_machines, self.num_mpiprocs_per_machine, self.num_cores_per_mpiproc,
            ipw.HBox([ipw.HTML("walltime:"), self.walltime_d, self.walltime_h, self.walltime_m])
        ]

        super().__init__(children=children)
        ### ---------------------------------------------------------

    def return_dict(self):
        mpi_tasks = self.num_machines.value * self.num_mpiprocs_per_machine.value
        walltime = int(self.walltime_d.value * 3600 * 24 + self.walltime_h.value * 3600 + 
                       self.walltime_m.value * 60)
        return {
                'mpi_tasks' : mpi_tasks,
                'walltime'  : walltime , 
                'metadata'  : {
                    'options' : {
                        'resources' : {
                            'num_machines' : self.num_machines.value,
                            'num_mpiprocs_per_machine' : self.num_mpiprocs_per_machine.value,
                            'num_cores_per_mpiproc' : self.num_cores_per_mpiproc.value
                                      },
                        'max_wallclock_seconds' : walltime,
                        'withmpi': True
                    }
                }
        }

    
    
    @observe('selected_code')
    def _observe_selected_code(self, _=None):
        if self.selected_code:
            self.num_mpiprocs_per_machine.value = self.selected_code.computer.get_default_mpiprocs_per_machine()
            
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
            link((self.manager, 'selected_code'), (self, 'selected_code'))

            
SECTIONS_TO_DISPLAY = {
        'None'     : [],
        'Bulk'     : [VdwSelectorWidget, MetadataWidget],
        'SlabXY'   : [VdwSelectorWidget, UksSectionWidget, 
                      MixedDftWidget, FixedAtomsWidget,
                      ConvergenceDetailsWidget,CellSectionWidget, 
                      MetadataWidget],
        'Molecule' : [MixedDftWidget, MetadataWidget]
    }
