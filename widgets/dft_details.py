from __future__ import print_function
from __future__ import absolute_import

from apps.surfaces.widgets.analyze_structure import mol_ids_range

import ipywidgets as ipw
from IPython.display import display, clear_output

from collections import OrderedDict

###The widgets defined here assign value to the following input keywords
###stored in job_details:
#  'max_force'    
#  'calc_type'    : 'Mixed DFT'
#  'mgrid_cutoff' 
#  'vdw_switch'   
#  'center_switch'
#  'charge'
#  'multiplicity' 
#  'uks_switch'   : 'UKS' or 'RKS'
#      'spin_guess' e.e. C1 -1 1 1 C2 -1 1 1

#  'cell_free'  :'FREE','KEEP_ANGLES', 'KEEP_SYMMETRY'   
#  'cell_sym'   : 'CUBIC' 'ORTHOROMBIC'...   
#  'functional' : 'PBE' 'B3LYP' B3LYP not implemented  
#  'gw_type'    : 'RI_RPA_GPW','GW-IC','GW-LS'  
#      'group_size' 'max_memory' 'size_freq_integ_group' 'ev_sc_iter' 'corr_occ' 'corr_virt'
#      'eps_filter' 'eps_grid' 'rpa_num_quad_points' 'eps_schwarz' 'eps_pgf_orb_s'
#       'group_size_3c' 'gorup_size_p' 'memory_cut'

#  'cell'       : '30 30 30' not yet implemented    



style = {'description_width': '120px'}
layout = {'width': '70%'}
layout2 = {'width': '35%'}
FUNCTION_TYPE = type(lambda c: c)
 
WIDGETS_ENABLED = {
    'None'                 : [],
    'SlabGeoOptWorkChain'  : ['fixed_atoms','calc_type','vdw_switch','convergence','cell'],
    'CellOptWorkChain'     : ['vdw_switch','convergence','cell'],
    'BulkOptWorkChain'     : ['vdw_switch','convergence','cell'],
    'MoleculeOptWorkChain' : ['vdw_switch','uks','convergence','cell'],    
    'GWWorkChain'          : ['gw','uks','convergence','cell']
}
CENTER_COORD = {
    'None'                 : 'False',
    'SlabGeoOptWorkChain'  : 'False',
    'CellOptWorkChain'     : 'False',
    'BulkOptWorkChain'     : 'False',
    'MoleculeOptWorkChain' : 'True',    
    'GWWorkChain'          : 'True'
}
class DFTDetails(ipw.VBox):        
    def __init__(self,
                 workchain           = 'SlabGeoOptWorkChain',
                 structure_details   = {},
                 **kwargs):
        
        self.structure_details=structure_details
        self.widgets_enabled = WIDGETS_ENABLED[workchain]
#        self.the_dict = the_dict
#        self.the_dict['workchain']=workchain


        #### ALONE vdW FIXED_ATOMS AND CALC_TYPE
        self.vdw_switch = ipw.ToggleButton(value=True,
                      description='Dispersion Corrections',
                      tooltip='VDW_POTENTIAL',
                      style=style, layout=layout)
        self.calc_type = ipw.ToggleButtons(options=['Mixed DFTB', 'Mixed DFT', 'Full DFT'],
                               description='Calculation Type',
                               value='Full DFT',
                               tooltip='Active: DFT, Inactive: DFTB',
                               style=style, layout=layout)        
        self.fixed_atoms = ipw.Text(placeholder='1..10',
                                    description='Fixed Atoms',
                                    style=style, layout={'width': '60%'})
        
        self.btn_fixed_atoms = ipw.Button(description='show',
                    layout={'width': '10%'})
        self.btn_fixed_pressed = False 
        self.fixed_display=ipw.HBox([self.fixed_atoms ,self.btn_fixed_atoms])
        
        
        #### CONVERGENCE PARAM.
        self.max_force = ipw.BoundedFloatText(descritpion='MAX FORCE',value=1e-4, min=1e-4, 
                                   max=1e-3, step=1e-4,style=style, layout=layout2)
        self.mgrid_cutoff = ipw.IntSlider(descritpion='MGRID CUTOFF',value=600, step=100,
                              min=200, max=1200,
                              style=style, layout=layout2)

        self.convergence_dict={
            'max_force'    : self.max_force,
            'mgrid_cutoff' : self.mgrid_cutoff
            }

        self.convergence = ipw.Accordion(selected_index=None)
        self.convergence.children = [ipw.VBox([self.convergence_dict[k] for k in self.convergence_dict.keys() ])]
        self.convergence.set_title(0,'Convergence parameters')

        
        #### CELL and PBC
        self.periodic     = ipw.Dropdown(description='PBC',options=['XYZ','NONE', 'X','XY',
                                              'XZ','Y','YZ',
                                              'Z'],
                                       value='XYZ',
                                       style=style, layout=layout2) 


        self.poisson_solver      = ipw.Dropdown(description='Poisson solver',options=['MT','PERIODIC', 'ANALYTIC','IMPLICIT',
                                              'MULTIPOLE','WAVELET'],
                                       value='PERIODIC',
                                       style=style, layout=layout2)
        self.cell_sym     = ipw.Dropdown(description='symmetry',options=['CUBIC','HEXAGONL', 'MONOCLINIC','NONE',
                                              'ORTHORHOMBIC','RHOMBOHEDRAL','TETRAGONAL_AB',
                                              'TETRAGONAL_AC','TETRAGONAL_BC','TRICLINIC'],
                                       value='ORTHORHOMBIC',
                                       style=style, layout=layout)

        self.cell = ipw.Text(description='cell size',
                            style=style, layout={'width': '60%'})
        
        self.center_coordinates = ipw.RadioButtons(description='center coordinates',
            options=['False', 'True'], value=CENTER_COORD[workchain],disabled=False
        )

        self.cell_spec_dict={
            'periodic'           : self.periodic,
            'poisson_solver'     : self.poisson_solver,
            'cell_sym'           : self.cell_sym,
            'cell'               : self.cell,
            'center_coordinates' : self.center_coordinates
        }


        self.cell_spec = ipw.Accordion(selected_index=None)
        self.cell_spec.children = [ipw.VBox([self.cell_spec_dict[k] for k in self.cell_spec_dict.keys() ])]
        self.cell_spec.set_title(0,'CELL/PBC details')        
        
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
        self.charge       = ipw.IntText(value=0,
                                   description='net charge',
                                   style=style, layout=layout)

        self.uks_dict={
            'multiplicity' : self.multiplicity,
            'spin_u'       : self.spin_u,
            'spin_d'       : self.spin_d,
            'charge'       : self.charge
                 }


        self.uks = ipw.Accordion(selected_index=None)
        self.uks.children = [ipw.VBox([self.uks_dict[k] for k in self.uks_dict.keys() ])]
        self.uks.set_title(0,'RKS/UKS')        
        
        #### GW

        self.gw_type_btn = ipw.RadioButtons(description='GW type',
            options=['GW', 'GW-LS','GW-IC'], value='GW',disabled=False
        )
        ###GROUP_SIZE 12
        self.group_size = ipw.IntText(value=12,
                           description='Group Size',
                           style=style, layout=layout)
        ###GROUP_SIZE_3C 32
        self.group_size_3c = ipw.IntText(value=32,
                           description='Group Size 3C',
                           style=style, layout=layout)
        ###GROUP_SIZE_P 4
        self.group_size_p = ipw.IntText(value=4,
                           description='Group Size P',
                           style=style, layout=layout)
        ###MAX_MEMORY 0
        #self.max_memory = ipw.IntText(value=0,
        #                   description='Max Memory',
        #                   style=style, layout=layout)
        #self.gw_w_dict['GW']['max_memory']=self.max_memory
        #self.gw_w_dict['GW-IC']['max_memory']=self.max_memory  
        ###MEMORY_CUT 12
        self.memory_cut = ipw.IntText(value=12,
                           description='Memory cut',
                           style=style, layout=layout)        
        ###SIZE_FREQ_INTEG_GROUP  1200
        self.size_freq_integ_group = ipw.IntText(value=-1,
                           description='Size freq integ group',
                           style=style, layout=layout)
        ###EV_SC_ITER 10
        self.ev_sc_iter = ipw.IntText(value=10,
                           description='# EV SC iter',
                           style=style, layout=layout)
        ###CORR_OCC 15            
        self.corr_occ = ipw.IntText(value=10,
                           description='# KS occ',
                           style=style, layout=layout)
        ###CORR_VIRT 15            
        self.corr_virt = ipw.IntText(value=10,
                           description='# KS virt',
                           style=style, layout=layout)
        ###EPS_DEFAULT  1.0E-15         
        self.eps_default = ipw.FloatLogSlider(value=-15,min=-30, base=10,
                                        max=-12, step=1, description='EPS_DEFAULT',
                                        style=style, layout=layout
                                        )      
        ###EPS_FILTER  1.0E-12    
        self.eps_filter = ipw.FloatLogSlider(value=-12,min=-20, base =10,
                                        max=-12, step=1, description='EPS_FILTER',
                                        style=style, layout=layout
                                        )
        ###EPS_GRID 1.0E-12    
        self.eps_grid = ipw.FloatLogSlider(value=-12,min=-30,  base =10,
                                      max=-12, step=1, description='EPS_GRID',
                                      style=style, layout=layout
                                     )
        ###EPS_PGF_ORB_S 1.0E-30    
        self.eps_pgf_orb_s = ipw.FloatLogSlider(value=-12,min=-30, base =10,
                                      max=-12, step=1, description='EPS_PGF_ORB_S',
                                      style=style, layout=layout
                                     )        
        ###RPA_NUM_QUAD_POINTS 200  NOT IN IC          
        self.rpa_num_quad_points = ipw.IntText(value=200,
                                               description='# rpa quad pt. use 12 (max 20) for  LS',
                                               style=style, layout=layout
                                              )
        ###EPS_SCHWARZ   1.0E-13  NOT IN IC  
        self.eps_schwarz = ipw.FloatLogSlider(value=-13,min=-20, base =10,
                                         max=-13, step=1, description='EPS_SCHWARZ',
                                         style=style, layout=layout
                                        )
    
        ###EPS_FILTER_IM_TIME 1.0E-12 IC  
        self.eps_filter_im_time = ipw.FloatLogSlider(value=-12,min=-20, base =10,
                                         max=-12, step=1, description='EPS_FILTER_IM_TIME',
                                         style=style, layout=layout
                                        )

        ###        
        self.ads_height = ipw.FloatText(value=3.0, step=0.1,
                                   description='Ads. height (wrt geom. center)',
                                   style=style, layout=layout)
        
        self.gw_dict={
            'GW'    : {'group_size'            : self.group_size,
                       'size_freq_integ_group' : self.size_freq_integ_group,
                       'ev_sc_iter'            : self.ev_sc_iter,
                       'corr_occ'              : self.corr_occ,
                       'corr_virt'             : self.corr_virt,
                       'eps_default'           : self.eps_default,
                       'eps_filter'            : self.eps_filter,
                       'eps_grid'              : self.eps_grid,
                       'rpa_num_quad_points'   : self.rpa_num_quad_points,
                       'eps_schwarz'           : self.eps_schwarz},
            'GW-LS' : {'group_size_3c'         : self.group_size_3c,
                       'group_size_p'          : self.group_size_p,
                       'memory_cut'            : self.memory_cut,
                       'ev_sc_iter'            : self.ev_sc_iter,
                       'corr_occ'              : self.corr_occ,
                       'corr_virt'             : self.corr_virt,
                       'eps_default'           : self.eps_default,
                       'eps_filter'            : self.eps_filter,
                       'eps_grid'              : self.eps_grid,
                       'eps_pgf_orb_s'         : self.eps_pgf_orb_s,
                       'rpa_num_quad_points'   : self.rpa_num_quad_points,
                       'eps_filter_im_time'    : self.eps_filter_im_time},
            'GW-IC' : {'group_size'            : self.group_size,
                       'corr_occ'              : self.corr_occ,
                       'corr_virt'             : self.corr_virt,
                       'eps_default'           : self.eps_default,
                       'eps_filter'            : self.eps_filter,
                       'eps_grid'              : self.eps_grid,
                       'eps_filter_im_time'    : self.eps_filter_im_time,
                       'ads_height'            : self.ads_height}

            }


        self.gw = ipw.Accordion(selected_index=None)
        self.gw_display=ipw.VBox([self.gw_type_btn,self.gw])
        self.gw_type=self.gw_type_btn.value
        def on_gw_select(c):
            self.gw_type=self.gw_type_btn.value
            self.gw.children = [ipw.VBox([self.gw_dict[self.gw_type][k1] for k1 in self.gw_dict[self.gw_type].keys()])]
        self.gw_type_btn.observe(on_gw_select)
        self.gw.children = [ipw.VBox([self.gw_dict[self.gw_type][k1] for k1 in self.gw_dict[self.gw_type].keys()])]
        self.gw.set_title(0,'GW parameters')
               
        #### Define the methods you want the widgets to observe
            ##fixed atoms
        def on_fixed_atoms_btn_press(b):
            self.btn_fixed_pressed = not self.btn_fixed_pressed
            self.btn_fixed_atoms.description = "hide" if self.btn_fixed_pressed else "show"
            
        self.btn_fixed_atoms.on_click(on_fixed_atoms_btn_press)
        
            ### DFT toggle button
        def on_dft_toggle(v): 
 #           with self.dft_out:
 #               clear_output()
                if 'Slab' in self.structure_details['system_type']:
                    if self.calc_type.value == 'Full DFT':
                        to_fix=[i for i in self.structure_details['bottom_H'] + 
                        self.structure_details['slab_layers'][0] +
                        self.structure_details['slab_layers'][1]]
                    else:
                        to_fix=self.structure_details['bottom_H'] + self.structure_details['slabatoms']
                    self.fixed_atoms.value=mol_ids_range(to_fix)
                    
        self.calc_type.observe(on_dft_toggle, 'value')        
        

        #### HOW TO DISPLAY/READ WIDGETS        
        self.all_widgets={
            'fixed_atoms' : {'display' : self.fixed_display, 'dict' : 'no'                  , 'retrieve' : self.fixed_atoms},
            'convergence' : {'display' : self.convergence,   'dict' : self.convergence_dict , 'retrieve' : self.convergence_dict},
            'calc_type'   : {'display' : self.calc_type,     'dict' : 'no'                  , 'retrieve' : self.calc_type},
            'vdw_switch'  : {'display' : self.vdw_switch,    'dict' : 'no'                  , 'retrieve' : self.vdw_switch},
            'gw'          : {'display' : self.gw_display,    'dict' : self.gw_dict          , 'retrieve' : self.get_gw_values},
            'cell'        : {'display' : self.cell_spec,     'dict' : self.cell_spec_dict   , 'retrieve' : self.cell_spec_dict},
            'uks'         : {'display' : self.uks,           'dict' : self.uks_dict         , 'retrieve' : self.uks_dict}
            }

        
        ####list widgets to be visualized and link to observe functions
   
        visualize_widgets=[self.all_widgets[k]['display'] for k in self.widgets_enabled]     

        #### IF A VALUE CHANGES DISABLE SUBMIT 
  #      def observe_all_widgets(c):
  #          print('something has changed')
  #      set_of_widgets=[]    
  #      for w in self.all_widgets.keys():
  #          if self.all_widgets[w]['dict'] == 'no':
  #              set_of_widgets.append(self.all_widgets[w]['retrieve'])
  #          else:
  #              for k in self.all_widgets[w]['dict'].keys():
  #                  if isinstance(self.all_widgets[w]['dict'][k],dict):
  #                      for k1 in self.all_widgets[w]['dict'][k].keys():
  #                          set_of_widgets.append(self.all_widgets[w]['dict'][k][k1])
  #                  else:
  #                      set_of_widgets.append(self.all_widgets[w]['dict'][k])
  #      for i in set_of_widgets:
  #          print(i.value,len(set_of_widgets))
  #          i.observe(observe_all_widgets)
  #                      
                    
                
        
        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX

        
##################SUPER        
        
        super(DFTDetails, self).__init__(children=visualize_widgets, **kwargs)
        
        #with self.dft_out:            
        #    display(self.mgrid_cutoff)

        


    #### specific FUNCTIONS to retrieve widget values                   
    def get_gw_values(self,the_dict):
        self.gw_type=self.gw_type_btn.value
        the_dict['gw_type']=self.gw_type
        for k in  self.gw_dict[self.gw_type].keys():
            the_dict[k]=self.gw_dict[self.gw_type][k].value    
            
    #### GENERAL FUNCTION to retrieve widget values
    def get_standard_values(self, d , the_dict):
        for k in d.keys():
            the_dict[k]=d[k].value    
    def get_widget_values(self,the_dict):
        if 'Slab' in self.structure_details['system_type'] and self.calc_type.value != 'Full DFT':
            the_dict['first_slab_atom'] = min(self.structure_details['bottom_H'] + self.structure_details['slabatoms']) + 1
            the_dict['last_slab_atom']  = max(self.structure_details['bottom_H'] + self.structure_details['slabatoms']) + 1
        for w in self.widgets_enabled:
            ## get widget value directly from widget
            if self.all_widgets[w]['dict']=='no':  
                the_dict[w]=self.all_widgets[w]['retrieve'].value
                
            ## get widget value directly from get_standard_values and widget_dict
            elif type(self.all_widgets[w]['retrieve']) == type({}):
                self.get_standard_values(self.all_widgets[w]['retrieve'],the_dict)
                
            ## get widget value directly from ad-hoc function
            else:
                self.all_widgets[w]['retrieve'](the_dict) 
                       
        
    def reset(self,structure_details): 
        self.structure_details=structure_details
        self.cell.value=self.structure_details['cell']
        if 'Slab' in self.structure_details['system_type']:
            to_fix=[i for i in self.structure_details['bottom_H'] + 
                    self.structure_details['slab_layers'][0] +
                    self.structure_details['slab_layers'][1]]
            self.fixed_atoms.value=mol_ids_range(to_fix)
