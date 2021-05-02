from __future__ import print_function
from __future__ import absolute_import

import ipywidgets as ipw
from IPython.display import display, clear_output

#from apps.surfaces.widgets import analyze_structure
from apps.surfaces.widgets.ANALYZE_structure import StructureAnalyzer

from aiidalab_widgets_base import viewer

from aiida.orm import CalcFunctionNode, CalcJobNode, Node, QueryBuilder, WorkChainNode, StructureData, load_node
import datetime

FIELDS_DISABLE_DEFAULT={
    'cell'   : True,
    'volume' : True,
    'extras' : True,
}

AU_TO_EV = 27.211386245988

class SearchCompletedWidget(ipw.VBox):
    
    def __init__(self,version=0.0, wlabel='',clabel='', fields_disable = {}):
        
        self.fields_disable = FIELDS_DISABLE_DEFAULT
        for fd in fields_disable:
            self.fields_disable[fd] = fields_disable[fd]
        # search UI
        self.wlabel=wlabel
        self.clabel=clabel
        self.version=version
        style = {"description_width":"150px"}
        layout = ipw.Layout(width="600px")
        self.inp_pks = ipw.Text(description='PKs', placeholder='e.g. 4062 4753 (space separated)', layout=layout, style=style)
        self.inp_formula = ipw.Text(description='Formulas:', placeholder='e.g. C44H16 C36H4', layout=layout, style=style)
        self.text_description = ipw.Text(description='Calculation Name: ',
                                    placeholder='e.g. keywords',
                                    layout=layout, style=style)

        # ---------
        # date selection
        dt_now = datetime.datetime.now()
        dt_from = dt_now - datetime.timedelta(days=20)
        self.date_start = ipw.Text(value=dt_from.strftime('%Y-%m-%d'),
                                   description='From: ',
                                   style={'description_width': '60px'}, layout={'width': '225px'})

        self.date_end = ipw.Text(value=dt_now.strftime('%Y-%m-%d'),
                                  description='To: ',
                                  style={'description_width': '60px'}, layout={'width': '225px'})

        self.date_text = ipw.HTML(value='<p>Select the date range:</p>', width="150px")
        # ---------

        search_crit = [self.inp_pks, self.inp_formula, self.text_description, ipw.HBox([self.date_text, self.date_start, self.date_end])]
        
        
        


        button = ipw.Button(description="Search")

        self.results = ipw.HTML()
        self.info_out = ipw.Output()
        
        def on_click(b):
            with self.info_out:
                clear_output()
                self.search() 
                
        button.on_click(on_click)
        
        self.show_comments_check = ipw.Checkbox(
            value=False,
            description='show comments',
            indent=False
        )
        
        buttons_hbox = ipw.HBox([button, self.show_comments_check])
        
        app = ipw.VBox(children=search_crit + [buttons_hbox, self.results, self.info_out])
        
       
        
        #self.search()
        super(SearchCompletedWidget, self).__init__([app])
        
        
        #display(app)        
        
        
        
    def search(self):

        self.results.value = "preprocessing..."
        self.preprocess_newbies()
        try:
            import apps.scanning_probe.common
            apps.scanning_probe.common.preprocess_spm_calcs(
                workchain_list = ['STMWorkChain', 'PdosWorkChain', 'AfmWorkChain', 'HRSTMWorkChain', 'OrbitalWorkChain'])
            self.fields_disable['extras'] = False
        except Exception as e:
            print("Warning: scanning_probe app not found, skipping spm preprocess.")
            self.fields_disable['extras'] = True

        self.value = "searching..."

        # html table header
        html  = '<style>#aiida_results td,th {padding: 2px}</style>' 
        html += '<table border=1 id="aiida_results" style="margin:0px"><tr>'
        html += '<th>PK</th>'
        html += '<th>Creation Time</th>'
        html += '<th >Formula</th>'
        html += '<th>Calculation name</th>'
        html += '<th>Energy(eV)</th>'
        if not self.fields_disable['cell'] :
            html += '<th>Cell</th>'
        if not self.fields_disable['volume'] :
            html += '<th>Volume</th>'
        html += '<th style="width: 100px">Structure</th>'
        if self.show_comments_check.value:
            html += '<th>Comments</th>'
        if not self.fields_disable['extras'] :
            html += '<th style="width: 10%">Extras</th>'
        html += '</tr>'

        # query AiiDA database
        filters = {}
        filters['label'] = self.wlabel
        filters['extras.preprocess_version'] = self.version
        filters['extras.preprocess_successful'] = True
        filters['extras.obsolete'] = False

        pk_list = self.inp_pks.value.strip().split()
        if pk_list:
            filters['id'] = {'in': pk_list}

        formula_list = self.inp_formula.value.strip().split()
        if self.inp_formula.value:
            filters['extras.formula'] = {'in': formula_list}

        if len(self.text_description.value) > 1:
            filters['description'] = {'like': '%{}%'.format(self.text_description.value)}

        try: # If the date range is valid, use it for the search
            start_date = datetime.datetime.strptime(self.date_start.value, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(self.date_end.value, '%Y-%m-%d') + datetime.timedelta(hours=24)
        except ValueError: # Otherwise revert to the standard (i.e. last 10 days)
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=20)

            date_start.value = start_date.strftime('%Y-%m-%d')
            date_end.value = end_date.strftime('%Y-%m-%d')

        filters['ctime'] = {'and':[{'<=': end_date},{'>': start_date}]}

        qb = QueryBuilder()        
        qb.append(WorkChainNode, filters=filters)
        qb.order_by({WorkChainNode:{'ctime':'desc'}})

        for i, node_tuple in enumerate(qb.iterall()):
            node = node_tuple[0]
            thumbnail = node.extras['thumbnail']
            description = node.extras['structure_description']
            opt_structure_uuid = node.extras['opt_structure_uuid']

            ## Find all extra calculations done on the optimized geometry
            extra_calc_links = ""
            opt_structure = load_node(opt_structure_uuid)
            st_extras = opt_structure.extras

            ### --------------------------------------------------
            ### add links to SPM calcs
            try:
                import apps.scanning_probe.common
                extra_calc_links += apps.scanning_probe.common.create_viewer_link_html(st_extras, "../../")
            except Exception as e:
                pass
            ### --------------------------------------------------

            extra_calc_area = "<div id='wrapper' style='overflow-y:auto; height:100px; line-height:1.5;'> %s </div>" % extra_calc_links
            
                
            # append table row
            html += '<tr>'
            html += '<td>%d</td>' % node.pk
            html += '<td>%s</td>' % node.ctime.strftime("%Y-%m-%d %H:%M")
            html += '<td>%s</td>' % node.extras['formula']
            html += '<td>%s</td>' % node.description
            html += '<td>%.4f</td>' % node.extras['energy_ev']
            if not self.fields_disable['cell'] :
                html += '<td>%s</td>' % node.extras['cell']
            if not self.fields_disable['volume'] :
                html += '<td>%f</td>' % node.extras['volume']
            # image with a link to structure export
            html += '<td><a target="_blank" href="../export_structure.ipynb?uuid=%s">' % opt_structure_uuid
            html += '<img width="100px" src="data:image/png;base64,%s" title="PK%d: %s">' % (thumbnail, opt_structure.pk, description)
            html += '</a></td>'
            
            if self.show_comments_check.value:
                comment_area = "<div id='wrapper' style='overflow-y:auto; height:100px; line-height:1.5;'>"
                comment_area += '<a target="_blank" href="../comments.ipynb?pk=%s">add/view</a><br>' % node.pk
                for comment in node.get_comments():
                    comment_area += "<hr style='padding:0px; margin:0px;' />" + comment.content.replace("\n", "<br>");
                comment_area += '</div>'
                html += '<td>%s</td>' % (comment_area)
            
            if not self.fields_disable['extras'] :
                html += '<td>%s</td>' % extra_calc_area
            html += '</td>'
            html += '</tr>'

        html += '</table>'
        html += 'Found %d matching entries.<br>'%qb.count()

        self.results.value = html  
        
    def preprocess_newbies(self):
        qb = QueryBuilder()
        qb.append(WorkChainNode, filters={
            'label': self.wlabel,
            'or':[
                   {'extras': {'!has_key': 'preprocess_version'}},
                   {'extras.preprocess_version': {'<': self.version}},
               ],
        })

        for m in qb.all(): # iterall() would interfere with set_extra()
            n = m[0]
            if not n.is_sealed:
                print("Skipping underway workchain PK %d"%n.pk)
                continue
            if 'obsolete' not in n.extras:
                n.set_extra('obsolete', False)
            try:
                self.preprocess_one(n)
                n.set_extra('preprocess_successful', True)
                n.set_extra('preprocess_error', '')
                n.set_extra('preprocess_version', self.version)
                print("Preprocessed PK %d"%n.pk)
            except Exception as e:
                n.set_extra('preprocess_successful', False)
                n.set_extra('preprocess_error', str(e))
                n.set_extra('preprocess_version', self.version)
                print("Failed to preprocess PK %d: %s"%(n.pk, e))


    def preprocess_one(self,workcalc):

        def get_calc_by_label(workcalc,label):
            qb = QueryBuilder()
            qb.append(WorkChainNode, filters={'uuid':workcalc.uuid})
            qb.append(CalcJobNode, with_incoming=WorkChainNode, filters={'label':label})
            qb.order_by({'CalcJobNode_1':[{'id':{'order':'desc'}}]})
            if qb.count() == 0:
                raise(Exception("Could not find %s calculation."%label))
            calc = qb.all()[0][0]
            return calc

        # check if handler stopped workchain after 5 steps
        if workcalc.exit_status == 401:
            raise(Exception("The workchain reached the maximum step number. Check and resubmit manually"))
        if workcalc.exit_status != 0:
            raise(Exception("The workchain excepted. Check and resubmit manually"))            
        
        # optimized structure
        calc = get_calc_by_label(workcalc, self.clabel) # TODO deal with restarts, check final state
        opt_structure = calc.outputs.output_structure
        
        # initial structure:
        maybe = workcalc.inputs.cp2k__file__input_xyz.get_incoming().all_nodes()[0].get_incoming().all_nodes()
        for i in range(len(maybe)):
            if isinstance(maybe[i],type(opt_structure)):
                structure=maybe[i]
        ase_struct = structure.get_ase()        
        
        #res = analyze_structure.analyze(ase_struct)
        
        an=StructureAnalyzer()
        an.structure = ase_struct
        res=an.details
        
        mol_formula=''
        for imol in res['all_molecules']:
            mol_formula+=ase_struct[imol].get_chemical_formula()+' '
        if len(res['slabatoms'])>0:
            slab_formula=ase_struct[res['slabatoms']].get_chemical_formula()
            if len(res['bottom_H']) >0:
                slab_formula+=' saturated: ' + ase_struct[res['bottom_H']].get_chemical_formula()
            if len(res['adatoms']) >0:
                slab_formula+=' adatoms: ' + ase_struct[res['adatoms']].get_chemical_formula()  
            workcalc.set_extra('formula', '{} at {}'.format(mol_formula,slab_formula))
        else:
            formula = ase_struct.get_chemical_formula()
            workcalc.set_extra('formula', '{}'.format(formula))

        workcalc.set_extra('structure_description', structure.description)    



        workcalc.set_extra('opt_structure_uuid', calc.outputs.output_structure.uuid)
        workcalc.set_extra('energy', calc.res.energy)
        workcalc.set_extra('energy_ev', calc.res.energy * AU_TO_EV)
        workcalc.set_extra('cell', calc.outputs.output_structure.get_ase().get_cell_lengths_and_angles())
        workcalc.set_extra('volume', calc.outputs.output_structure.get_ase().get_volume())        
        
        

        # thumbnail
        thumbnail = viewer(opt_structure).thumbnail
        workcalc.set_extra('thumbnail', thumbnail)