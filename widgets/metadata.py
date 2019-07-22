from __future__ import print_function
from __future__ import absolute_import

import ipywidgets as ipw
style = {'description_width': '120px'}
layout = {'width': '70%'}


class MetadataWidget(ipw.VBox):
    def __init__(self):
        """ Metadata widget to generate metadata"""

        self.walltime = ipw.IntText(value=86000,
                           description='walltime',
                           style={'description_width': '120px'}, layout={'width': '30%'})
        
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

        children = [
            self.calc_name,
            self.num_machines,
            self.num_mpiprocs_per_machine,
            self.num_cores_per_mpiproc]

        super(MetadataWidget, self).__init__(children=children)
        ### ---------------------------------------------------------

    @property
    def dict(self):
        return {
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