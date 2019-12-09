from __future__ import print_function
from __future__ import absolute_import

import ipywidgets as ipw
from IPython.display import display, clear_output

from collections import OrderedDict

###The widgets defined here assign value to the following input keywords
###stored in job_details:
#  'calc_name'    : 'TEST CALC'
#  'fixed_atoms'  : '1..100'
#  'num_machines' 
#  'max_force'    
#  'calc_type'    : 'Mixed DFT'
#  'mgrid_cutoff' 
#  'vdw_switch'   
#  'center_switch'
#  'system_charge'
#  'multiplicity' 
#  'uks_switch'   : 'UKS' or 'RKS'
#      'spin_guess' e.e. C1 -1 1 1 C2 -1 1 1

#  'cell_free'  :'FREE','KEEP_ANGLES', 'KEEP_SYMMETRY'   
#  'cell_sym'   : 'CUBIC' 'ORTHOROMBIC'...   
#  'functional' : 'PBE' 'B3LYP' B3LYP not implemented  
#  'gw_type'    : 'RI_RPA_GPW','GW-IC','GW-LS'  
#      'group_size' 'size_freq_integ_group' 'ev_sc_iter' 'corr_occ' 'corr_virt'
#      'eps_filter' 'eps_grid' 'rpa_num_quad_points' 'eps_schwarz'

#  'cell'       : '30 30 30' not yet implemented    



style = {'description_width': '120px'}
layout = {'width': '70%'}

FUNCTION_TYPE = type(lambda c: c)
WIDGETS_DISABLE_DEFAULT = {
    'calc_name'     : False,  
    'fixed_atoms'   : False,
    'num_machines'  : False,        
    'max_force'     : False,
    'calc_type'     : False,
    'mgrid_cutoff'  : False,
    'vdw_switch'    : False,
    'periodic'      : True,
    'center_switch' : True,
    'system_charge' : True,
    'multiplicity'  : True,
    'uks_switch'    : True,
    'cell_free'     : True,
    'cell_sym'      : True,
    'functional'    : True,
    'gw_type'       : True,
    'cell'          : True,
}

class DFTDetails(ipw.VBox):        
    def __init__(self,
                 job_details           = {},
                 widgets_disabled      = {},
                 **kwargs):
        
        self.widgets_disabled = WIDGETS_DISABLE_DEFAULT
        for wd in widgets_disabled:
            self.widgets_disabled[wd] = widgets_disabled[wd]
        
        ### ---------------------------------------------------------
        ### Define all child widgets contained in this composite widget
        self.job_details=job_details

        self.calc_name = ipw.Text(description='Calculation Name: ',
                                  placeholder='A great name.',
                                  style=style, layout=layout)

        self.fixed_atoms = ipw.Text(placeholder='1..10',
                                    description='Fixed Atoms',
                                    style=style, layout={'width': '60%'})
        
        self.btn_fixed_atoms = ipw.Button(description='show',
                    layout={'width': '10%'})
        self.btn_fixed_pressed = False

        self.num_machines = ipw.IntText(value=1,
                           description='# Nodes',
                           style=style, layout=layout)   
        
        self.max_force = ipw.BoundedFloatText(description='MAX_FORCE:', value=1e-4, min=1e-4, 
                                        max=1e-3, step=1e-4,style=style, layout=layout)

        self.calc_type = ipw.ToggleButtons(options=['Mixed DFTB', 'Mixed DFT', 'Full DFT'],
                               description='Calculation Type',
                               value='Full DFT',
                               tooltip='Active: DFT, Inactive: DFTB',
                               style=style, layout=layout)

        self.mgrid_cutoff = ipw.IntSlider(description='MGRID_CUTOFF:',
                              value=600, step=100,
                              min=200, max=1200,
                              style=style, layout=layout)
        
        self.vdw_switch = ipw.ToggleButton(value=True,
                              description='Dispersion Corrections',
                              tooltip='VDW_POTENTIAL',
                              style=style, layout=layout)
        
        self.center_switch = ipw.ToggleButton(value=False,
                              disabled=False,
                              description='Center Coordinates (only FULL DFT)',
                              tooltip='Center Coordinates',
                              style=style, layout=layout)
        
        self.periodic     = ipw.Dropdown(options=['XYZ','NONE', 'X','XY',
                                              'XZ','Y','YZ',
                                              'Z'],
                                       description='PBC',
                                       value='NONE',
                                       style=style, layout=layout) 
        
        self.psolver      = ipw.Dropdown(options=['MT','PERIODIC', 'ANALYTIC','IMPLICIT',
                                              'MULTIPOLE','WAVELET'],
                                       description='Poisson solver',
                                       value='MT',
                                       style=style, layout=layout)        
        def get_periodic():
            return self.periodic.value
        def get_psolver():
            return self.psolver.value
        
        self.dft_out=ipw.Output()
        self.message_output=ipw.Output()
###for CELL and UKS
        self.system_charge = ipw.IntText(value=0,
                                   description='net charge',
                                   style=style, layout=layout)

        self.multiplicity = ipw.IntText(value=0,
                                   description='MULTIPLICITY',
                                   style=style, layout=layout)
        self.uks_switch = ipw.ToggleButtons(options=['UKS','RKS'],
                                       description='UKS',
                                       value='RKS',
                                       style=style, layout=layout)
        self.spin_guess = ipw.VBox()
        self.spin_guess_string = None
        def get_spin_string():
            return self.spin_guess_string
        self.create_spin_guess_boxes()

        self.cell_free   = ipw.ToggleButtons(options=['FREE','KEEP_ANGLES', 'KEEP_SYMMETRY'],
                                       description='Cell freedom',
                                       value='KEEP_SYMMETRY',
                                       style=style, layout=layout)

        self.cell_sym     = ipw.Dropdown(options=['CUBIC','HEXAGONL', 'MONOCLINIC','NONE',
                                              'ORTHORHOMBIC','RHOMBOHEDRAL','TETRAGONAL_AB',
                                              'TETRAGONAL_AC','TETRAGONAL_BC','TRICLINIC'],
                                       description='Cell symmetry',
                                       value='ORTHORHOMBIC',
                                       style=style, layout=layout)
        
        self.functional = ipw.Dropdown(options=['PBE','B3LYP'],
                                   description='XC Functional',
                                   value='PBE',
                                   style=style, layout=layout)    
        
       
        
        self.cell = ipw.Text(placeholder=' 30 30 20',
                            description='cell',
                            style=style, layout={'width': '60%'})
        
        
##GW SECTION
        self.gw_type = ipw.Dropdown(options=['RI_RPA_GPW','GW-IC','GW-LS'],
                                   description='GW type',
                                   value='RI_RPA_GPW',
                                   style=style, layout=layout) 
        self.gw_w_dict={'gw_type' :self.gw_type}
########GROUP_SIZE 12
        self.group_size = ipw.IntText(value=12,
                           description='Group Size',
                           style=style, layout=layout)
        self.gw_w_dict['group_size']=self.group_size
########SIZE_FREQ_INTEG_GROUP  1200
        self.size_freq_integ_group = ipw.IntText(value=1200,
                           description='Size freq integ group',
                           style=style, layout=layout)
        self.gw_w_dict['size_freq_integ_group']=self.size_freq_integ_group
########EV_SC_ITER 10
        self.ev_sc_iter = ipw.IntText(value=10,
                           description='# EV SC iter',
                           style=style, layout=layout)
        self.gw_w_dict['ev_sc_iter']=self.ev_sc_iter
########CORR_OCC 15            
        self.corr_occ = ipw.IntText(value=10,
                           description='# KS occ',
                           style=style, layout=layout)
        self.gw_w_dict['corr_occ']=self.corr_occ
########CORR_VIRT 15            
        self.corr_virt = ipw.IntText(value=10,
                           description='# KS virt',
                           style=style, layout=layout)
        self.gw_w_dict['corr_virt']=self.corr_virt
########EPS_FILTER  1.0E-12    
        self.eps_filter = ipw.IntSlider(value=-12,min=-20, 
                                        max=-12, step=1, description='EPS_FILTER 10^-',
                                        style=style, layout=layout
                                        )
        self.gw_w_dict['eps_filter']=self.eps_filter
########EPS_GRID 1.0E-12    
        self.eps_grid = ipw.IntSlider(value=-12,min=-20, 
                                      max=-12, step=1, description='EPS_GRID 10^-',
                                      style=style, layout=layout
                                     )
        self.gw_w_dict['eps_grid']=self.eps_grid
########RPA_NUM_QUAD_POINTS 200            
        self.rpa_num_quad_points = ipw.IntText(value=200,
                                               description='# rpa quad pt.',
                                               style=style, layout=layout
                                              )
        self.gw_w_dict['rpa_num_quad_points']=self.rpa_num_quad_points
########EPS_SCHWARZ   1.0E-13    
        self.eps_schwarz = ipw.IntSlider(value=-13,min=-20, 
                                         max=-13, step=1, description='EPS_SCHWARZ 10^-',
                                         style=style, layout=layout
                                        )
        self.gw_w_dict['eps_schwarz']=self.eps_schwarz
##END GW
        
        #### -------------------------------------------------------------------------------------------
        #### -------------------------------------------------------------------------------------------
        #### -------------------------------------------------------------------------------------------
        #### Methods to observe
        
        # Define the methods you want the widgets to observe
        # by default, the widgets will be tied to the member function update_job_details
        
        def on_fixed_atoms_btn_press(b):
            self.btn_fixed_pressed = not self.btn_fixed_pressed
            self.btn_fixed_atoms.description = "hide" if self.btn_fixed_pressed else "show"
            
        self.btn_fixed_atoms.on_click(on_fixed_atoms_btn_press)
        
        ### DFT toggle button
        def on_dft_toggle(v): 
            self.update_job_details()
            with self.dft_out:
                clear_output()
                if self.calc_type.value in ['Mixed DFT', 'Full DFT']:
                    display(self.mgrid_cutoff)
                    
        self.calc_type.observe(on_dft_toggle, 'value')
                    
        def check_charge():
            with self.message_output:
                clear_output()
                total_charge=np.sum(atoms.get_atomic_numbers()) +  self.system_charge.value
                if total_charge % 2 > 0 :
                    print("odd charge: UKS NEEDED if CHARGE  is odd")
                    self.uks_switch.value = 'UKS'
                    self.multiplicity.value = 2
                else:
                    self.uks_switch.value = 'RKS'
                    self.multiplicity.value = 0
                self.update_job_details()
                
        self.system_charge.observe(lambda c: check_charge(), 'value')
        
        
        self.uks_switch.observe(lambda c: self.create_spin_guess_boxes(), 'value')
                
        #### -------------------------------------------------------------------------------------------
        #### -------------------------------------------------------------------------------------------
        #### -------------------------------------------------------------------------------------------
        #### 
                
        self.independent_widgets = OrderedDict([
            ('num_machines' ,   self.num_machines  ),
            ('calc_name'    ,   self.calc_name     ),
            ('fixed_atoms'  ,   self.fixed_atoms   ),
            ('max_force'    ,   self.max_force     ),
            ('calc_type'    ,   self.calc_type     ),             
            ('vdw_switch'   ,   self.vdw_switch    ),         
            ('center_switch',   self.center_switch ),
            ('periodic'     ,   self.periodic      ),
            ('gw_type'      ,   self.gw_type       ),   
            ('functional'   ,   self.functional    ),            
            ('mgrid_cutoff' ,   self.mgrid_cutoff  ),
            ('system_charge',   self.system_charge ), 
            ('multiplicity' ,   self.multiplicity  ),  
            ('uks_switch'   ,   self.uks_switch    ),
            ('cell_free'    ,   self.cell_free     ),       
            ('cell_sym'     ,   self.cell_sym      ),
            ('cell'         ,   self.cell          )
        ])

        ####some widgets do not have themselves to be visualized 
        ####here are listed the exceptions
        
        self.independent_widget_children = OrderedDict([            
            ('fixed_atoms'  , [ipw.HBox([self.fixed_atoms, self.btn_fixed_atoms])]),        
            ('mgrid_cutoff' , [self.dft_out]),
            ('periodic'     , [ipw.HBox([self.periodic, self.psolver])]),
            ('uks_switch'   , [self.uks_switch,self.spin_guess]),
            ('gw_type'      , list(self.gw_w_dict.values())),
        ])
        
        ####some widgets follow special rules to update the job_details e.g. uks_switch
        #### needs to store both 'UKS' and spin_guess
        self.independent_widget_jd = OrderedDict([                               
            ('uks_switch'   , {'spin_guess':get_spin_string}),
            ('gw_type'      , self.gw_w_dict ),
            ('periodic'     , {'periodic':self.periodic,'psolver':self.psolver})
        ])

        
        ####list widgets to be visualized and link to observe functions
        visualize_widgets=[]
        for wk in self.independent_widgets:
            enabled = not self.widgets_disabled[wk]
            if enabled:
                if wk in self.independent_widget_children:
                    visualize_widgets.extend(self.independent_widget_children[wk])
                    
                else:
                    visualize_widgets.append(self.independent_widgets[wk])
                
                # Add the default observe TODO: check if needed for every widget
                self.independent_widgets[wk].observe(lambda v: self.update_job_details(), 'value')
                
                
                
        
        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX

        
##################SUPER        
        
        super(DFTDetails, self).__init__(children=visualize_widgets, **kwargs)
        
        with self.dft_out:            
            display(self.mgrid_cutoff)
    #### update the job_details dictionary           
    def update_job_details(self):
        for w in self.independent_widgets.keys():
            if not self.widgets_disabled[w]:                                
                self.job_details[w]=self.independent_widgets[w].value
                if w in  self.independent_widget_jd:
                    for jd_key in self.independent_widget_jd[w].keys():
                        method=self.independent_widget_jd[w][jd_key]
                        if type(method)==FUNCTION_TYPE:
                            self.job_details[jd_key]=method()
                        else:
                            self.job_details[jd_key]=method.value
                        
               

    def reset(self,
              calc_name="",
              fixed_atoms="",
              num_machines=1,
              btn_fixed_pressed=False,
              btn_fixed_atoms="show",
              vdw_switch=True,
              calc_type="Full DFT",
              center_switch=False,
              uks_switch='RKS',
              cell=''
             ):         
        self.calc_name.value = calc_name
        self.fixed_atoms.value = fixed_atoms
        self.num_machines.value = int(num_machines)
        self.btn_fixed_pressed=btn_fixed_pressed
        self.btn_fixed_atoms.description = btn_fixed_atoms           
        self.calc_type.value = calc_type
        self.vdw_switch.value = vdw_switch
        self.center_switch.value = center_switch
        self.uks_switch.value = uks_switch
        self.cell.value = cell
        self.update_job_details()
    
    def generate_spin_guess(self, int_net, guess_kinds):
        spin_guess = [
            [
                str(guess_kinds[i]), str(int_net[i][0].value),
                str(int_net[i][1].value), str(int_net[i][2].value)
            ] for i in range(len(guess_kinds))
        ]
        self.spin_guess_string=" ".join([x for xs in spin_guess for x in xs])
        
        self.update_job_details()
        
    def create_spin_guess_boxes(self):
        if self.uks_switch.value == 'UKS':
            spins_up   = list(self.job_details['slab_analyzed']['spins_up']) 
            spins_down = list(self.job_details['slab_analyzed']['spins_down'])

            self.int_net=[]
            guess_kinds=spins_up + spins_down
            if len(guess_kinds)>0:
                for k in guess_kinds:

                    self.int_net.append([
                        ipw.IntText(value=0,description='NEL '+k,style = {'description_width': '60px'}, layout = {'width': '15%'}),
                        ipw.IntText(value=0,description='L '+k,style   = {'description_width': '60px'}, layout = {'width': '15%'}),
                        ipw.IntText(value=0,description='N '+k,style   = {'description_width': '60px'}, layout = {'width': '15%'})
                    ])

                    for i_n in self.int_net[-1]:
                        i_n.observe(lambda c, int_n=self.int_net, g_kinds=guess_kinds: self.generate_spin_guess(int_n, g_kinds), 'value')

            self.spin_guess.children = tuple([  ipw.HBox([wn,wL,wN]) for wn, wL, wN, in self.int_net ])
            self.generate_spin_guess(self.int_net, guess_kinds)
        else:
            self.spin_guess.children = tuple()
            
    