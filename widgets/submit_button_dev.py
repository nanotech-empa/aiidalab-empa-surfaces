from __future__ import print_function
from __future__ import absolute_import

import ipywidgets as ipw
from IPython.display import display, clear_output
from aiida.orm import load_node


from aiida.orm.code import Code
from aiida.orm import load_node
from aiida.orm import Code, Computer
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm.data.base import Int, Float, Str, Bool, List
from aiida.orm.data.parameter import ParameterData
from aiida.work.run import submit
from aiida.orm.data.structure import StructureData

style = {'description_width': '120px'}
layout = {'width': '70%'}


class SubmitButton(ipw.VBox):
    def __init__(self, the_workchain=None, job_details=None, presub_calls=[], **kwargs):
        """ Submit Button

        :param 
        """
        
        self.the_workchain=the_workchain
        self.submit_details={}
        self.job_details=job_details
        
        # list of methods to call after submit button is pressed
        self.presub_calls = presub_calls
        
        self.btn_submit = ipw.Button(description="Submit",disabled=False)
        self.walltime = ipw.IntText(value=86000,
                           description='walltime',
                           style={'description_width': '120px'}, layout={'width': '30%'})

        self.submit_out = ipw.Output()

        self.job_details['walltime']=self.walltime.value
        def set_walltime():
            self.job_details['walltime']=self.walltime.value
            
        def on_btn_submit_press(b):
            
            for presub_call in self.presub_calls:
                presub_call()
            
            #self.parse_job_details()
            keys_defined=self.job_details.keys()
            with self.submit_out:
                clear_output()
                
                ### CHECK VALIDITY OF INPUTS
                if 'structure' not in keys_defined:
                    print("Please select a structure.")
                    return
                if 'cp2k_code' not in keys_defined:
                    print("Please select a computer.")
                    return
                if len(self.job_details['calc_name']) < 5:
                    print("Please enter a longer calculation description.")
                    return
                
                ### ODD CHARGE AND RKS
                odd_charge  = self.job_details['slab_analyzed']['total_charge']
                if 'charge' in self.job_details.keys():
                    odd_charge += self.job_details['charge'] 
                    rks = True
                    if 'uks_switch' in self.job_details.keys():
                        if  self.job_details['uks_switch'] == 'UKS':
                            rks=False                       
                if  odd_charge%2 >0 and rks  :
                    print("ODD CHARGE AND RKS")
                    return                
                if self.job_details['workchain'] == 'NEBWorkchain':
                    if len(self.job_details['replica_pks'].split()) < 2:
                        print('Please select at least two  replica_pks')
                        return
                self.structure = self.job_details['structure']
                self.code      = self.job_details['cp2k_code']
                print("SUBMITTING workchain ",self.the_workchain)
                print("")
                
                self.btn_submit.disabled=True
                self.parse_job_details()
                for field in self.submit_details.keys():
                    #if field != 'slab_analyzed':
                    print(field, self.submit_details.get_dict()[field])
                    
                if self.job_details['calc_name'] != 'DO NOT SUBMIT':
                    arg_dict = {'code': self.code, 'structure': self.structure, 'parameters': self.submit_details}
                    if self.struc_folder is not None:
                        arg_dict['struc_folder'] = self.struc_folder
                    outputs = submit(self.the_workchain, **arg_dict)
                    print(outputs)
                else:
                    print("TEST NO SUBMISSION")
                    theworkchain=self.the_workchain()
                    outputs = theworkchain.build_calc_inputs(structure=self.structure,
                                                             input_dict=self.submit_details.get_dict())
                    print(outputs['parameters'].get_dict())

                print("")
                print("DONE")
                
            if self.job_details['calc_name'] != 'DO NOT SUBMIT':                 
                the_workcalc = load_node(outputs.pid)
                the_workcalc.description = self.job_details['calc_name']
        
        self.btn_submit.on_click(on_btn_submit_press)
        self.walltime.observe(set_walltime, 'value')
        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX
        
        children = [ipw.HBox([self.btn_submit,self.walltime]),self.submit_out]

        super(SubmitButton, self).__init__(children=children, **kwargs)
        ### ---------------------------------------------------------
        
        
    def parse_job_details(self):        
        self.submit_details=ParameterData()
        
        self.struc_folder=None
        if 'struc_folder' in self.job_details.keys():  ## e.g. for the NEB
            self.struc_folder=self.job_details['struc_folder']
            self.job_details.pop('struc_folder')
        
        self.job_details['walltime']=self.walltime.value
        self.job_details.pop('slab_analyzed')
        self.job_details.pop('cp2k_code')
        self.job_details.pop('structure')        
        self.submit_details.set_dict(self.job_details)   
        
    

    