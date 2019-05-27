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

#from apps.surfaces.slab.slabwork import SlabGeoOptWorkChain
#from apps.reactions.replicawork import ReplicaWorkchain
#from apps.reactions.nebwork import NEBWorkchain
#from apps.bulks.celloptwork import CellOptWorkChain
#from apps.bulks.bulkoptwork import BulkOptWorkChain
#from apps.surfaces.widgets.neb_utils import mk_coord_files, mk_wfn_cp_commands

style = {'description_width': '120px'}
layout = {'width': '70%'}


class SubmitButton(ipw.VBox):
    def __init__(self, the_workchain=None, job_details=None, presub_calls=[], **kwargs):
        """ Submit Button

        :param 
        """
        self.aiida_type_conversion = {
            type('a')        : Str,
            type(u'a')       : Str,
            type(int(1))     : Int,
            type(float(1.0)) : Float,
            type(True)       : Bool,
        }
        
        self.the_workchain=the_workchain
        self.submit_details={}
        self.job_details=job_details
        
        # list of methods to call after submit button is pressed
        self.presub_calls = presub_calls
        
        self.btn_submit = ipw.Button(description="Submit",disabled=False)

        self.submit_out = ipw.Output()

        def on_btn_submit_press(b):
            
            for presub_call in self.presub_calls:
                presub_call()
            
            self.parse_job_details()
            
            with self.submit_out:
                clear_output()
                if not self.job_details['structure']:
                    print("Please select a structure.")
                    return
                if not self.job_details['cp2k_code']:
                    print("Please select a computer.")
                    return
                if len(self.job_details['calc_name']) < 5:
                    print("Please enter a longer calculation description.")
                    return
                if self.job_details['workchain'] == 'NEBWorkchain':
                    if len(self.job_details['replica_pks'].split()) < 2:
                        print('Please select at least two  replica_pks')
                        return
                print("SUBMITTING workchain ",self.the_workchain)
                print("")
                
                self.btn_submit.disabled=True
                
                for field in self.submit_details.keys():
                    #if field != 'slab_analyzed':
                    print(field, self.submit_details[field])
                if self.job_details['calc_name'] != 'DO NOT SUBMIT':  
                    outputs = submit(self.the_workchain, **self.submit_details)
                    print(outputs)
                else:
                    print("TEST NO SUBMISSION")
                    theworkchain=self.the_workchain()
                    outputs = theworkchain.build_calc_inputs(**self.submit_details)
                    print(outputs['parameters'].get_dict())

                print("")
                print("DONE")
                
            if self.job_details['calc_name'] != 'DO NOT SUBMIT':                 
                the_workcalc = load_node(outputs.pid)
                the_workcalc.description = self.job_details['calc_name']
        
        self.btn_submit.on_click(on_btn_submit_press)
        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX
        
        children = [self.btn_submit,self.submit_out]

        super(SubmitButton, self).__init__(children=children, **kwargs)
        ### ---------------------------------------------------------
        
        
    def parse_job_details(self):
        
        self.submit_details={}
        
#        if self.job_details['workchain'] == 'SlabGeoOptWorkChain':            
#            self.the_workchain=SlabGeoOptWorkChain
#            
#        elif self.job_details['workchain'] == 'NEBWorkchain':
#            self.the_workchain=NEBWorkchain
#            
#        elif self.job_details['workchain'] == 'CellOptWorkChain':
#            self.the_workchain=CellOptWorkChain  
#            
#        elif self.job_details['workchain'] == 'BulkOptWorkChain':
#            self.the_workchain=BulkOptWorkChain            
            
            
        ##function to convert to aiida types
        def to_aiida_type(py_obj):
            if type(py_obj) in self.aiida_type_conversion.keys():
                return self.aiida_type_conversion[type(py_obj)](py_obj)
                #return aiida.orm.data.base.to_aiida_type(py_obj)
            elif type(py_obj) == type([1,2,3]):                
                aiidalist=List()
                aiidalist.extend(py_obj)
                return aiidalist
            elif type(py_obj) == type({}):                
                aiidadict=ParameterData()
                aiidadict.set_dict(py_obj)
                return aiidadict
            else:
                return py_obj
            
        ##CONVERSION to AIIDA types    
        to_be_converted=self.job_details.keys()
        to_be_converted.remove('slab_analyzed')
        for the_detail in to_be_converted:
            self.submit_details[the_detail]=to_aiida_type(self.job_details[the_detail])
        ##END CONVERSION to AIIDA types
    

    