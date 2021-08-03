from apps.surfaces.widgets.ANALYZE_structure import mol_ids_range
from apps.surfaces.widgets.cp2k_input_validity import validate_input
from aiidalab_widgets_base.utils import string_range_to_list, list_to_string_range

from datetime import datetime

from ase import Atom, Atoms

from aiida.orm import Code

#from aiida_cp2k.workchains.base import Cp2kBaseWorkChain

from apps.surfaces.widgets.cp2k2dict import CP2K2DICT
from apps.surfaces.widgets.number_of_nodes import compute_cost,compute_nodes
from aiida_cp2k.utils import Cp2kInput

from apps.surfaces.widgets.get_cp2k_input import Get_CP2K_Input

import ipywidgets as ipw
from IPython.display import display, clear_output

from collections import OrderedDict

from traitlets import Instance, Bool, Int, List, Set, Dict, Unicode, Union, link, default, observe, validate


STYLE = {'description_width': '120px'}
LAYOUT = {'width': '70%'}
LAYOUT2 = {'width': '35%'}

 
#WIDGETS_ENABLED = {
#    'None'     : [],
#    'SlabXY'   : ['fixed_atoms','calc_type','vdw_switch','convergence','cell'],
#    'Bulk'     : ['vdw_switch','convergence','cell'],
#    'Molecule' : ['vdw_switch','convergence','cell']
#}
#CENTER_COORD = {
#    'None'                 : 'False',
#    'Bulk'                 : 'False',
#    'SlabXY'               : 'False',
#    'Molecule'             : 'True'
#}


class InputDetails(ipw.VBox):
    selected_code = Union([Unicode(), Instance(Code)], allow_none=True)
    details = Dict()
    final_dictionary = Dict()
    to_fix = List()
    do_cell_opt = Bool()
    uks = Bool()
    calc_type = Unicode()
#    gw_trait = Unicode()
    net_charge = Int()

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
#        self.gw = ipw.ToggleButton(value=False, description='Activate GW',
#                                                    tooltip='Activate GW', style={'description_width': '80px'})
#        self.gw.observe(self.on_gw_toggle,'value')
            
        super().__init__(children=[self.output])

        
        
    @observe('details')
    def _observe_details(self, _=None):
        self.to_fix = []
        self.net_charge = 0
        self.do_cell_opt = False
        self.uks = False
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
            add_children = []
#            if  'Molecule' in sys_type:
#                supported_all_elements = all(elem in ['C','H','B','N','S','Zn','O',]  for elem in list(set(self.details['all_elements'])))
#                if supported_all_elements :
#                    add_children=[self.gw]
            for sec in SECTIONS_TO_DISPLAY[sys_type]:
                section = sec()
                section.manager = self
                self.displayed_sections.append(section)    
            display(ipw.VBox(add_children + self.displayed_sections + [self.plain_input_accordion]))
            
#    def on_gw_toggle(self,c=None):
#        tmp_details = self.details.copy()
#        if self.gw.value:
#            tmp_details['system_type'] = 'Molecule_GW'
#            self.details = tmp_details
#            self.gw_trait='GW'
#        else:
#            tmp_details['system_type'] = 'Molecule'
#            self.details = tmp_details
#            self.gw_trait='None'

    def create_plain_input(self):
        inp_dict = Get_CP2K_Input(input_dict = self.final_dictionary).inp
        #self.plain_input.value = str(inp_dict) #test
        #print(inp_dict)
        inp_plain = Cp2kInput(inp_dict)
        self.plain_input.value = inp_plain.render()         
        
    def return_final_dictionary(self):
        tmp_dict = {}
        
        ## PUT LIST OF ELEMENTS IN DICTIONARY
        tmp_dict['elements']=self.details['all_elements']
        
        ## RETRIEVE ALL WIDGET VALUES
        for section in self.displayed_sections:
            to_add = section.return_dict()
            if to_add : tmp_dict.update(to_add)  
        
        ## DECIDE WHICH KIND OF WORKCHAIN
        
        ## SLAB
        if self.details['system_type'] == 'SlabXY':
            tmp_dict.update({'workchain' : 'SlabGeoOptWorkChain'})
            ## IN CASE MIXED DFT FOR SLAB IDENTIFY MOLECULE
            if tmp_dict['calc_type'] != 'Full DFT':
                tmp_dict['first_slab_atom'] = min(self.details['bottom_H'] +
                                                               self.details['slabatoms']) + 1
                tmp_dict['last_slab_atom']  = max(self.details['bottom_H'] +
                                                               self.details['slabatoms']) + 1
        ## MOLECULE   
        elif self.details['system_type'] == 'Molecule' :
            tmp_dict.update({'workchain' : 'MoleculeOptWorkChain'})
#        elif self.details['system_type'] == 'Molecule_GW' :    
#            tmp_dict.update({'workchain' : 'GWWorkChain'})
            
        ## BULK
        elif self.details['system_type'] == 'Bulk' :
            if tmp_dict['opt_cell']:
                tmp_dict.update({'workchain' : 'CellOptWorkChain'})
            else:
                tmp_dict.update({'workchain' : 'BulkOptWorkChain'})
                
        ## CHECK input validity
        can_submit,error_msg=validate_input(self.details,tmp_dict)
                
        ## CREATE PLAIN INPUT  
        if can_submit :
            self.final_dictionary = tmp_dict
            self.create_plain_input()        
        
        #print(self.final_dictionary)
        ## RETURN DICT of widgets details
        return  can_submit,error_msg, self.final_dictionary


class DescriptionWidget(ipw.Text):    
    
## DESCRIPTION OF CALCULATION
    def __init__(self):
        
        super().__init__(description='Process description: ', value='',
                               placeholder='Type the name here.',
                               style={'description_width': '120px'},
                               layout={'width': '70%'})    
        
    def return_dict(self):
        return {'description' : self.value }     
    
class StructureInfoWidget(ipw.Accordion): 
    details = Dict()  
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self):
        
        self.info=ipw.Output()
        
        self.set_title(0,'Structure details')
        super().__init__(selected_index=None)
        
    @observe('details')
    def _observe_details(self, _=None):
        if self.details is None:
            return
        else:
            self.children=[ipw.VBox([self.info])]
            with self.info:
                clear_output()
                print(self.details['summary'])        
        
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:            
            link((self.manager, 'details'), (self, 'details')) 
                
    def return_dict(self):
        return {}        
    
class ConvergenceDetailsWidget(ipw.Accordion):
    details = Dict()
    calc_type = Unicode()
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self):    
        
        self.max_force = ipw.FloatText(description='MAX FORCE',value=1e-4,
                                       style={'description_width': 'initial'}, layout={'width': '170'})
        self.mgrid_cutoff = ipw.IntText(description='MGRID CUTOFF',value=600,
                                        style={'description_width': 'initial'}, layout={'width': '170'})

        
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
    

class UksSectionWidget(ipw.Accordion):
    details = Dict()
    uks = Bool()
    net_charge = Int()
    calc_type = Unicode()
    manager = Instance(InputDetails, allow_none=True)
    def __init__(self):
                #### UKS
            
        self.uks_toggle          = ipw.ToggleButton(value=False, description='UKS',
                                                    tooltip='Activate UKS', style={'description_width': '80px'}) 
        
        
        def on_uks(c=None):
            self.uks = self.uks_toggle.value
            
        self.uks_toggle.observe(on_uks, 'value')        
        
        self.multiplicity = ipw.IntText(value=0,
                                           description='MULTIPLICITY',
                                           style={'description_width': 'initial'}, layout={'width': '140px'})
        self.spin_u = ipw.Text(placeholder='1..10 15',
                                            description='IDs atoms spin U',
                                            style={'description_width': 'initial'})

        self.spin_d = ipw.Text(placeholder='1..10 15',
                                            description='IDs atoms spin D',
                                            style={'description_width': 'initial'})
        
        
        self.charge = ipw.IntText(value=0,
                                 description='net charge',
                                 style={'description_width': 'initial'}, layout={'width': '120px'})
        
        ## guess multiplicity
        def multiplicity_guess(c=None):
            self.net_charge = self.charge.value
            system_charge=self.details['total_charge'] - self.net_charge
            setu=set(string_range_to_list(self.spin_u.value)[0])
            setd=set(string_range_to_list(self.spin_d.value)[0])
            ## check if same atom entered in two different spins
            if bool(setu & setd):
                self.multiplicity.value = 1
                self.spin_u.value = ''
                self.spin_d.value = '' 
                
            nu = len(string_range_to_list(self.spin_u.value)[0])
            nd = len(string_range_to_list(self.spin_d.value)[0])
            if not system_charge % 2:
                self.multiplicity.value = min(abs(nu - nd) * 2 + 1,3)
            else:
                self.multiplicity.value = 2 
            
        self.spin_u.observe(multiplicity_guess, 'value')
        self.spin_d.observe(multiplicity_guess, 'value')
        self.charge.observe(multiplicity_guess, 'value')
        
        super().__init__(selected_index=None)
        
    def return_dict(self):
        if self.calc_type == 'Mixed DFTB':
            return {
                'multiplicity' : 0,
                'spin_u'       : '',
                'spin_d'       : '',
                'charge'       : 0
            }
        elif self.uks:
            return {
                'multiplicity' : self.multiplicity.value,
                'spin_u'       : self.spin_u.value,
                'spin_d'       : self.spin_d.value,
                'charge'       : self.charge.value,
            }
        else:
            return {
                'multiplicity' : 0,
                'spin_u'       : '',
                'spin_d'       : '',
                'charge'       : self.charge.value,
            }
            

    def widgets_to_show(self):
        self.set_title(0,'RKS/UKS')
        if self.calc_type == 'Mixed DFTB':
            self.uks_toggle.value=False
            self.children = []
        elif self.uks:
            self.children = [ipw.VBox(
                [ipw.HBox([self.uks_toggle, self.multiplicity, self.spin_u, self.spin_d]), self.charge])]
        else:
            self.children = [ipw.VBox([self.uks_toggle, self.charge])]
            
        
    @observe('uks')    
    def _observe_uks(self,_=None):
        self.widgets_to_show()
        
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
            link((self.manager, 'uks'), (self, 'uks'))
            link((self.manager, 'net_charge'), (self, 'net_charge'))
            self.widgets_to_show()
    
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

#class GwWidget(ipw.HBox):
#    details = Dict()
#    gw_trait = Unicode()
#    manager = Instance(InputDetails, allow_none=True)
#    
#    def __init__(self,):
#        self.gw_type = ipw.ToggleButtons(value = 'GW',
#                                description='GW/GW+IC',options=['GW', 'GW-IC'],
#                                style={'description_width': 'initial'})
#        self.hq = ipw.ToggleButton(value=False, description='High precision',style={'description_width': 'initial'})
#        self.gw_trait='GW'
#        self.ic_plane_z = ipw.FloatText(description='IC z plane',value=8.22,
#                                        tooltip='e.g. 1.78A below molecule, 1.42A above Au(111)',
#                                       style={'description_width': 'initial'}, layout={'width': 'initial'}) #170
#        
#        super().__init__(children=[])
#        
#        self.gw_type.observe(self.on_gw, 'value')         
#        
#    def return_dict(self):
#        size = self.details['sys_size'] * 2 +15
#        gwcell = " ".join(map(str, [int(i) for i in size.tolist()]))
#        diag_method = 'GW'
#        if self.hq.value:
#            diag_method='GW_HQ'
#        if self.gw_type.value == 'GW':
#            size = self.details['sys_size'] * 2 +15
#            gwcell = " ".join(map(str, [int(i) for i in size.tolist()]))   
#
#            return {
#                'ic_plane_z' : None,
#                'gw_type': 'GW',
#                'gw_hq' : self.hq.value,
#                'cell'   : gwcell,
#                'periodic':'NONE',
#                'poisson_solver' : 'MT',
#                'center_coordinates' : True,
#                'diag_method'        : diag_method
#            }
#        else:
#            size = self.details['sys_size'] * 2 + 30
#            gwcell = " ".join(map(str, [int(i) for i in size.tolist()]))            
#            return {
#                'ic_plane_z' : self.ic_plane_z.value,
#                'gw_type': 'GW-IC',
#                'gw_hq' : self.hq.value,
#                'cell'   : gwcell,
#                'periodic':'NONE',
#                'poisson_solver' : 'MT',
#                'center_coordinates' : True,
#                'diag_method'        : diag_method
#            }                
#        
#
#    ## need to simplify this: trick to switch forth and back from GW and DFT    
#    def on_gw(self,c=None):
#        if self.gw_type.value == 'GW':
#            self.children=[self.gw_type,self.hq]
#            self.gw_trait='GW'
#        else:
#            self.children=[self.gw_type,self.ic_plane_z,self.hq]
#            self.gw_trait='GW-IC'
#        
#            
#    @observe('manager')
#    def _observe_manager(self, _=None):
#        if self.manager is None:
#            return
#        else:
#            link((self.manager, 'details'), (self, 'details'))  
#            link((self.manager, 'gw_trait'), (self, 'gw_trait'))
#            if self.details['system_type'] == 'Molecule_GW':
#                self.gw_type.value = 'GW'
#                self.children = [self.gw_type,self.hq]
#            elif self.details['system_type'] == 'Molecule_GW-IC':
#                self.gw_type.value = 'GW-IC'   
#                self.children = [self.gw_type,self.ic_plane_z,self.hq]
#            else:
#                self.gw_type.value = 'DFT'
#                self.children = [self.gw_type,self.hq]
#        
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

            
            
class CellSectionWidget(ipw.Accordion):
    details = Dict()
    do_cell_opt = Bool()
    net_charge = Int()
    manager = Instance(InputDetails, allow_none=True)
    
    def __init__(self):
        
    
        self.periodic = ipw.Dropdown(description='PBC',options=['XYZ','NONE', 'X','XY',
                                              'XZ','Y','YZ',
                                              'Z'],
                                       value='XYZ',
                                       style=STYLE, layout=LAYOUT2) 


        self.poisson_solver = ipw.Dropdown(description='Poisson solver',
                                           options=['PERIODIC'],
                                           value='PERIODIC',
                                           style=STYLE, layout=LAYOUT2)
        def observe_periodic(c=None):
            if self.periodic.value == 'NONE':
                self.poisson_solver.options = ['MT','ANALYTIC','IMPLICIT',
                                           'MULTIPOLE','WAVELET']
                self.poisson_solver.value = 'MT'
            elif self.periodic.value == 'XYZ':
                self.poisson_solver.options = ['PERIODIC']
                self.poisson_solver.value = 'PERIODIC'
                if self.net_charge and self.details['system_type']=='Molecule':
                    self.periodic.value = 'NONE'
                
        self.periodic.observe(observe_periodic)
        

        
        self.cell_sym  = ipw.Dropdown(description='symmetry',
                                      options=['CUBIC','HEXAGONL', 'MONOCLINIC','NONE',
                                               'ORTHORHOMBIC','RHOMBOHEDRAL','TETRAGONAL_AB',
                                               'TETRAGONAL_AC','TETRAGONAL_BC','TRICLINIC'],
                                      value='ORTHORHOMBIC',
                                      style=STYLE, layout=LAYOUT)

        self.cell = ipw.Text(description='cell size',
                            style=STYLE, layout={'width': '60%'})
        
        
        def observe_poisson(c=None):
            if self.poisson_solver.value == 'MT':
                cell = self.details['sys_size'] * 2 +15
                self.cell.value = " ".join(map(str, [int(i) for i in cell.tolist()]))
            elif self.poisson_solver.value == 'PERIODIC':
                self.cell.value =  self.details['cell']
                
            
        self.poisson_solver.observe(observe_poisson)        
        
        self.center_coordinates = ipw.RadioButtons(description='center coordinates',
                                                   options=['False', 'True'],
                                                   value='True',
                                                   disabled=False)
        self.opt_cell = ipw.ToggleButton(value=False, description='Optimize cell',
                                         style={'description_width': '120px'})
        
        def on_cell_opt(c=None):
            self.do_cell_opt = self.opt_cell.value
        self.opt_cell.observe(on_cell_opt, 'value')
        
        self.cell_free = ipw.ToggleButtons(options=['FREE','KEEP_ANGLES', 'KEEP_SYMMETRY'],
                                       description='Cell freedom',
                                       value='KEEP_SYMMETRY',
                                       style=STYLE, layout=LAYOUT)
        
#'cell_free'
        self.cell_cases = {
            'Cell_true'            : [
                ('cell', self.cell),
                ('cell_sym', self.cell_sym),
                ('cell_free', self.cell_free),
                ('opt_cell', self.opt_cell)
            ],
            'Bulk'                 : [
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

        
        
        super().__init__(selected_index=None)
        
    def return_dict(self):
            to_return = {}
            if self.opt_cell.value:
                cases = self.cell_cases['Cell_true']
            else:
                cases = self.cell_cases[self.details['system_type']]
                
            for i in cases:
                to_return.update({i[0] : i[1].value})
            return to_return
     
    def widgets_to_show(self):
        if self.opt_cell.value:
            self.set_title(0,'CELL/PBC details')
            self.children = [ipw.VBox([i[1] for i in self.cell_cases['Cell_true']])]
        else:
            self.set_title(0,'CELL/PBC details')
            self.children = [ipw.VBox([i[1] for i in self.cell_cases[self.details['system_type']]])]
            
        
    @observe('do_cell_opt')    
    def _observe_do_cell_opt(self,_=None):
        self.widgets_to_show()

    @observe('net_charge')    
    def _observe_net_charge(self,_=None):
        if self.net_charge and self.details['system_type']=='Molecule':
            self.periodic.value = 'NONE'
    
        
    
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
            link((self.manager, 'do_cell_opt'), (self, 'do_cell_opt'))
            link((self.manager, 'net_charge'), (self, 'net_charge'))
            self.cell.value = self.details['cell']
            self.widgets_to_show()

            
                
            

class MetadataWidget(ipw.VBox):
    """Setup metadata for an AiiDA process."""
    
    details = Dict()
#    gw_trait = Unicode()
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
        self.num_machines_scf = ipw.IntText(value=1, description='# Nodes_scf', style=STYLE, layout=LAYOUT2)

        self.num_mpiprocs_per_machine = ipw.IntText(value=12, description='# Tasks', style=STYLE, layout=LAYOUT2)
        

                                  
        self.num_mpiprocs_per_machine.observe(self.guess_nodes)
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
#        if self.details['system_type'] == 'Molecule_GW':
#            return {
#                    'num_machines_scf' : self.num_machines_scf.value,
#                    'num_mpiprocs_per_machine_scf' : self.selected_code.computer.get_default_mpiprocs_per_machine(),
#                    'mpi_tasks' : mpi_tasks,
#                    'walltime'  : walltime , 
#                    'metadata'  : {
#                        'options' : {
#                            'resources' : {
#                                'num_machines' : self.num_machines.value,
#                                'num_mpiprocs_per_machine' : self.num_mpiprocs_per_machine.value,
#                                'num_cores_per_mpiproc' : self.num_cores_per_mpiproc.value
#                                          },
#                            'max_wallclock_seconds' : walltime,
#                            'withmpi': True
#                        }
#                    }
#            }            
#        else:
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

    def guess_nodes(self,c=None):
#        if self.gw_trait == 'GW':
#            self.num_machines.value = (max(int(self.details['numatoms']/25),1))**3
#            self.num_machines_scf.value = max(int(self.details['numatoms']/30),1)
#        elif self.gw_trait == 'GW-IC':
#            self.num_machines.value = (max(int(self.details['numatoms']/20),1))**3
#            self.num_machines_scf.value = max(int(self.details['numatoms']/15),1)
#        elif self.details and 'all_elements' in self.details.keys():
        if self.details and 'all_elements' in self.details.keys():
            cost=compute_cost(self.details['all_elements'])
            self.num_machines.value = compute_nodes(cost,self.num_mpiprocs_per_machine.value)    
    
    @observe('selected_code')
    def _observe_selected_code(self, _=None):
        if self.selected_code:
            self.num_mpiprocs_per_machine.value = self.selected_code.computer.get_default_mpiprocs_per_machine()
            
#    @observe('gw_trait')
#    def _observe_gw_trit(self, _=None):   
#        self.guess_nodes()
            
    @observe('manager')
    def _observe_manager(self, _=None):
        if self.manager is None:
            return
        else:
            link((self.manager, 'details'), (self, 'details'))
#            link((self.manager, 'gw_trait'), (self, 'gw_trait'))
            link((self.manager, 'selected_code'), (self, 'selected_code'))
#            if self.details['system_type'] == 'Molecule_GW':
#                    self.children = [
#                        self.num_machines, self.num_mpiprocs_per_machine, self.num_cores_per_mpiproc,
#                        self.num_machines_scf,
#                        ipw.HBox([ipw.HTML("walltime:"), self.walltime_d, self.walltime_h, self.walltime_m])
#                    ]
#            else:
            self.children = [
                self.num_machines, self.num_mpiprocs_per_machine, self.num_cores_per_mpiproc,
                ipw.HBox([ipw.HTML("walltime:"), self.walltime_d, self.walltime_h, self.walltime_m])
            ]  
            self.guess_nodes()

            
SECTIONS_TO_DISPLAY = {
    'None'     : [],
    'Wire'     : [],
    'Bulk'     : [DescriptionWidget,
                  VdwSelectorWidget, 
                  UksSectionWidget,
                  StructureInfoWidget,
                  FixedAtomsWidget,
                  ConvergenceDetailsWidget,
                  CellSectionWidget, 
                  MetadataWidget],
    'SlabXY'   : [DescriptionWidget,
                  VdwSelectorWidget, 
                  UksSectionWidget, 
                  MixedDftWidget,
                  StructureInfoWidget,
                  FixedAtomsWidget,
                  ConvergenceDetailsWidget,
                  CellSectionWidget, 
                  MetadataWidget],
    'Molecule' : [DescriptionWidget,
                  VdwSelectorWidget,
                  UksSectionWidget,
                  StructureInfoWidget,
                  FixedAtomsWidget,
                  ConvergenceDetailsWidget,
                  CellSectionWidget, 
                  MetadataWidget],
#    'Molecule_GW' : [DescriptionWidget,
#                     GwWidget,
#                     UksSectionWidget,
#                     StructureInfoWidget,
#                     MetadataWidget],
    }
