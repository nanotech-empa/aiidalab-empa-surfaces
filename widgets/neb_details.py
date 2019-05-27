from __future__ import print_function
from __future__ import absolute_import

import ipywidgets as ipw
from IPython.display import display, clear_output
from apps.surfaces.widgets.neb_utils import mk_coord_files, mk_wfn_cp_commands
from aiida.orm.code import Code
from aiida.orm import load_node
from aiida.orm import Code, Computer

###The widgets defined here assign value to the following input keywords
###stored in jod_details:
#'nreplicas'
#'replica_pks'
#'wfn_cp_commands' copied in aiida.inp to retrieve old .wfn files
#'struc_folder' used to create replica files
#'nproc_rep'
#'spring'
#'nstepsit'
#'endpoints'
#'rotate'
#'align'


style = {'description_width': '120px'}
layout = {'width': '70%'}
layout3 = {'width': '23%'}

class NEBDetails(ipw.VBox):
    def __init__(self,job_details={},  **kwargs):
        """ Dropdown for DFT details
        """
        ### ---------------------------------------------------------
        ### Define all child widgets contained in this composite widget
        
        self.job_details=job_details
        
     

        self.proc_rep = ipw.IntText(value=324,
                           description='# Processors per replica',
                           style=style, layout=layout)  
        
        self.num_rep = ipw.IntText(value=15,
                           description='# replicas',
                           style=style, layout=layout)
        
        self.spring_constant = ipw.FloatText(description='Spring constant',
                             value=0.05,
                             min=0.01, max=0.5, step=0.01,
                             style=style, layout=layout)
        
        self.nsteps_it = ipw.IntText(value=5,
                           description='# steps before CI',
                           style=style, layout=layout)        
        
        self.optimize_endpoints = ipw.ToggleButton(value=False,
                              description='Optimize endpoints',
                              tooltip='Optimize endpoints',
                              style=style, layout=layout3)
        
        self.rotate_frames = ipw.ToggleButton(value=False,
                              description='Rotate Frames',
                              tooltip='Rotate Frames',
                              style=style, layout=layout3)
        
        self.align_frames = ipw.ToggleButton(value=False,
                              description='Align Frames',
                              tooltip='Align Frames',
                              style=style, layout=layout3)
        
        self.text_replica_pks = ipw.Text(placeholder='10000 10005 11113 11140',
                            description='Replica pks',
                            style=style, layout={'width': '50%'})
        
        self.btn_retrieve_wfn = ipw.Button(description='Retrieve WFN',
                                    layout={'width': '20%'})        

        ### ---------------------------------------------------------
        ### Logic

        def on_retrieve_wfn_btn_press(b):
            with self.neb_out:
                clear_output()                                           
                if 'cp2k_code' not in self.job_details.keys():
                    print("please select a computer")
                    return
                replica_pks = [int(a) for a in self.job_details['replica_pks'].split()]
                nreplicas=self.job_details['nreplicas']
                print('Find replica wavefunctions...')
                selected_computer = self.job_details['cp2k_code']
                aiida_wfn_cp_list = mk_wfn_cp_commands(nreplicas=nreplicas,
                                                       selected_computer = selected_computer ,
                                                       replica_pks = replica_pks)                           
                self.job_details['wfn_cp_commands']=aiida_wfn_cp_list
                print('Writing coordinate files...')
                #float_progress = ipw.FloatProgress(value=0, min=0, max=1)
                #display(float_progress)
                the_mols=self.job_details['slab_analyzed']['all_molecules']
                calc_type=self.job_details['calc_type']

                fd = mk_coord_files(replica_pks=replica_pks, all_mols=the_mols,calc_type=calc_type)
                self.job_details['struc_folder']=fd
                print(fd)
        
        self.btn_retrieve_wfn.on_click(on_retrieve_wfn_btn_press)
        
        update_jd_widgets = [
            self.proc_rep, self.num_rep, self.spring_constant,
            self.nsteps_it,  self.optimize_endpoints,
            self.rotate_frames, self.align_frames, self.text_replica_pks 
        ]
        for w in update_jd_widgets:
            w.observe(lambda v: self.update_job_details(), 'value')
        
        self.neb_out = ipw.Output()
        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX

        children = [
            self.proc_rep, 
            self.num_rep, 
            self.spring_constant,
            self.nsteps_it, 
            ipw.HBox([self.optimize_endpoints,self.rotate_frames, self.align_frames]), 
            ipw.HBox([self.text_replica_pks,self.btn_retrieve_wfn]),
            self.neb_out
        ]
            
        super(NEBDetails, self).__init__(children=children, **kwargs)
        
    ####TO DO decide how to deal with UPDATE VS WFN retrieve           
    def update_job_details(self):
             
        self.job_details['nproc_rep']=self.proc_rep.value
        self.job_details['nreplicas']=self.num_rep.value
        self.job_details['spring']=self.spring_constant.value
        self.job_details['nstepsit']=self.nsteps_it.value
        self.job_details['endpoints']=self.optimize_endpoints.value
        self.job_details['rotate']=self.rotate_frames.value
        self.job_details['align']=self.align_frames.value
        self.job_details['replica_pks']=self.text_replica_pks.value

    def reset(self,proc_rep=324, num_rep=15, spring_constant=0.05,
            nsteps_it=5,  optimize_endpoints=False,
            rotate_frames=False, align_frames=False, text_replica_pks=''):  
        
        self.proc_rep.value = proc_rep
        self.num_rep.value = num_rep
        self.spring_constant.value = spring_constant
        self.nsteps_it.value = nsteps_it 
        self.optimize_endpoints.value = optimize_endpoints
        self.rotate_frames.value = rotate_frames 
        self.align_frames.value = align_frames
        self.text_replica_pks.value = text_replica_pks
        #self.job_details={}
        self.update_job_details()
        


    