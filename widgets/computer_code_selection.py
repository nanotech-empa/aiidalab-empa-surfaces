from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from collections import OrderedDict

import ipywidgets as ipw
from IPython.display import display, clear_output


class ComputerCodeDropdown(ipw.VBox):
    def __init__(self, job_details={},input_plugin='cp2k',text0='Select computer', text='Select code:', **kwargs):
        """ Dropdown for Computers and Codes for one 
        input plugin on the selected computer.

        :param input_plugin: Input plugin of codes to show
        :type input_plugin: str
        :param text and text0: Text to display before dropdown
        :type text: str
        """

        self.input_plugin = input_plugin
        self.job_details=job_details
        self.codes = {}

        self.label0 = ipw.Label(value='Select computer')
        self.dropdown0 = ipw.Dropdown(options=[], disabled=True)
        self.dropdown0.observe(self.on_comp_change, 'value')
        self.label = ipw.Label(value=text)
        self.dropdown = ipw.Dropdown(options=[], disabled=True)
        self.output = ipw.Output()
        
        
        def update_dict(d):
            self.job_details['cp2k_code']=self.codes[self.dropdown.value]
                    
        self.dropdown.observe(update_dict,'value')
        
        children = [ipw.VBox([self.label0, self.dropdown0,self.label, self.dropdown, self.output])]

        super(ComputerCodeDropdown, self).__init__(children=children, **kwargs)

        from aiida import load_dbenv, is_dbenv_loaded
        from aiida.backends import settings
        if not is_dbenv_loaded():
            load_dbenv(profile=settings.AIIDADB_PROFILE)
        self.refresh()
        

    def _get_codes(self, sel_comp,input_plugin):

        from aiida.orm.querybuilder import QueryBuilder
        from aiida.orm import Code, Computer
        from aiida.backends.utils import get_automatic_user

        current_user = get_automatic_user()

        qb = QueryBuilder()
        qb.append(
            Computer, filters={'enabled': True, 'name': sel_comp.name}, project=['*'], tag='computer')
        qb.append(
            Code,
            filters={
                'attributes.input_plugin': {
                    '==': input_plugin
                },
                'extras.hidden': {
                    "~==": True
                }
            },
            project=['*'],
            has_computer='computer')
        results = qb.all()

        # only codes on computers configured for the current user
        results = [r for r in results if r[0].is_user_configured(current_user)]
        codes = {"{}@{}".format(r[1].label, r[0].name): r[1] for r in results}
        return codes       

    def _get_computers(self):
        from aiida.orm.querybuilder import QueryBuilder
        from aiida.orm import Code, Computer
        from aiida.backends.utils import get_automatic_user

        current_user = get_automatic_user()

        qb = QueryBuilder()
        qb.append(
            Computer, filters={'enabled': True}, project=['*'], tag='computer')
                
        current_user = get_automatic_user()
        computers = qb.all()
        computers = [r for r in computers if r[0].is_user_configured(current_user)]
        #computers = [comp[0] for comp in qb.all()]

        computer_selection = OrderedDict()
        computer_selection['Please select a computer'] = False
        for comp in computers:
            computer_selection[comp[0].name] = comp
        return computer_selection
              
    
    def refresh(self):
        with self.output:
            clear_output()
            optionsc=self._get_computers()
            if not optionsc:
                print("No computers found for this user ")
                self.dropdown0.disabled = True
                self.dropdown.disabled = True
            else:
                self.dropdown0.disabled = False
                self.dropdown0.options=optionsc
                self.on_comp_change('value')
                
    def on_comp_change(self,v): 
        sel_comp=self.dropdown0.value
        if sel_comp:
            sel_comp_uuid=sel_comp[0]
            self.codes = self._get_codes(sel_comp_uuid,self.input_plugin)
            options = list(self.codes.keys())
            if not options:
                print("No codes found for input plugin '{}'.".format(
                    self.input_plugin))
                self.dropdown.disabled = True
            else:
                self.dropdown.disabled = False
                self.dropdown.options = options
        else:
            self.dropdown.options = []
            self.dropdown.disabled = True


    @property
    def selected_code(self):
        try:
            return self.codes[self.dropdown.value]
        except KeyError:
            return None
        
    @property
    def selected_computer(self):
        try:
            return self.codes[self.dropdown.value].get_computer()
        except KeyError:
            return None