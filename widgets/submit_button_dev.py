from __future__ import print_function
from __future__ import absolute_import

import ipywidgets as ipw
from IPython.display import display, clear_output
from aiida.orm import load_node


from aiida.orm import Code
from aiida.orm import load_node
from aiida.orm import Code, Computer
from aiida.orm.querybuilder import QueryBuilder
from aiida.orm import Int, Float, Str, Bool, List
from aiida.orm import Dict
from aiida.engine import submit, run
from aiida.orm import StructureData

style = {'description_width': '120px'}
layout = {'width': '70%'}


class SubmitButton(ipw.VBox):
    def __init__(self, workchain, param_function):
        """ Submit Button

        :param 
        """
        
        self.workchain = workchain
        self.param_function=param_function
        self.btn_submit = ipw.Button(description="Submit",disabled=False)
        self.walltime = ipw.IntText(value=86000,
                           description='walltime',
                           style={'description_width': '120px'}, layout={'width': '30%'})
        self.submit_out = ipw.Output()
        
        self.calc_name = ipw.Text(description='Calculation Name: ',
                          placeholder='A great name.',
                          style=style, layout=layout)

        self.num_machines = ipw.IntText(value=1,
                           description='# Nodes',
                           style=style, layout=layout) 
        
        self.num_mpiprocs_per_machine = ipw.IntText(value=12,
                           description='# Tasks',
                           style=style, layout=layout)
        
        self.num_cores_per_mpiproc = ipw.IntText(value=1,
                           description='# Threads',
                           style=style, layout=layout)

        self.btn_submit.on_click(self.on_btn_submit_press)
        
        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX
        
        children = [ipw.VBox([
            self.calc_name,
            self.num_machines,
            self.num_mpiprocs_per_machine,
            self.num_cores_per_mpiproc,
            ipw.HBox([self.btn_submit,self.walltime]),self.submit_out])]

        super(SubmitButton, self).__init__(children=children)
        ### ---------------------------------------------------------

    def on_btn_submit_press(self, b):
        with self.submit_out:
            clear_output()
            input_dict = self.param_function()
            input_dict['cp2k']['metadata'] = {
                "options": {
                    "resources": {
                        "num_machines" : self.num_machines.value,
                        "num_mpiprocs_per_machine" : self.num_mpiprocs_per_machine.value,
                        "num_cores_per_mpiproc" : self.num_cores_per_mpiproc.value,
                    },
                    "max_wallclock_seconds": self.walltime.value,
                    'withmpi': True,
                }
            }
    
            if self.calc_name.value == 'DO NOT SUBMIT':
                print("TEST NO SUBMISSION")
                print(input_dict['parameters'].get_dict())
                return
            else:
                run(self.workchain, **input_dict)
            self.btn_submit.disabled=True
            print("SUBMITTED workchain ",self.workchain)
            print("")
            return