import ipywidgets as ipw
from IPython.display import display, clear_output, HTML
import nglview
import time
import ase.io
import ase.units as aseu
from ase.data.colors import jmol_colors
import urllib.parse
import numpy as np
import copy

import re
from collections import OrderedDict

import matplotlib.pyplot as plt
from pprint import pprint

from tempfile import NamedTemporaryFile
from base64 import b64encode


from aiida.orm.querybuilder import QueryBuilder
from aiida.orm import StructureData, Dict, WorkChainNode
from aiida.orm import load_node

from aiida.plugins import WorkflowFactory, CalculationFactory

from .collective_variables import COLVARS


ReplicaWorkchain = WorkflowFactory('replica')
Cp2kCalculation = CalculationFactory('cp2k')


style = {'description_width': '120px'}
layout = {'width': '70%'}

class SearchReplicaWidget(ipw.VBox):
    
    def __init__(self, **kwargs):
        
        self.preprocess_version = 0.05
        
        btn_style = {'description_width': '60px'}
        btn_layout = {'width': '20%'}
        
        
        self.check_new_btn = ipw.Button(description="Check new calculations", style=btn_style, layout=btn_layout)
        self.out_preproc = ipw.Output()
        
        self.check_new_btn.on_click(lambda x: self.preprocess_replicas())
        
        self.replica_calcs = None
        
        self.drop_replica = ipw.Dropdown(options = [], description = 'Calculation:',
                                         style=style, layout=layout)
        
        self.parse_preprocessed_replica_calcs()

        self.btn_show = ipw.Button(description="Show", style=btn_style, layout=btn_layout)
        self.output_header = ipw.Output()
        self.output_plot = ipw.Output()
        self.output_thumbnails = ipw.Output()
        self.output_checked_pks = ipw.Output()
        
        self.btn_show.on_click(self.on_show_btn_click)
        
        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX
        children = [
            self.check_new_btn,
            self.out_preproc,
            self.drop_replica,
            self.btn_show,
            self.output_header,
            self.output_plot,
            self.output_thumbnails,
            self.output_checked_pks
        ]
        super(SearchReplicaWidget, self).__init__(children=children, **kwargs)
        ### ---------------------------------------------------------
        
    def parse_preprocessed_replica_calcs(self):
        self.replica_calcs = self.parse_rep_wcs(self.get_replica_wcs(True))
        
        options = OrderedDict(sorted([
            (str(self.replica_calcs[key]['wcs'][0].ctime.date()) + " " + key, key) for key in self.replica_calcs.keys() 
        ], reverse=True))
        self.drop_replica.options = options
        
    
    def on_show_btn_click(self, btn):
        
        selected_replica_calc = self.replica_calcs[self.drop_replica.value]
        
        html_list, check_list = self.generate_thumbnail_html(selected_replica_calc)

        with self.output_header:
            clear_output()
            html = '<h2>{}</h2><br/> PK: '.format(self.drop_replica.value) + str([wc.pk for wc in selected_replica_calc['wcs']])
            display(ipw.HTML(html))
        
        with self.output_plot:
            clear_output()
            self.generate_energy_cv_plot(selected_replica_calc)
        
        n_col = 4
        layout = {'width':'%.1f%%' % (100.0/n_col)}
        with self.output_thumbnails:
            clear_output()
            cur_row = []
            for i, (html, cb) in enumerate(zip(html_list, check_list)):
                cur_row.append(ipw.VBox([ipw.HTML(html), cb], layout=layout))
                if (i+1) % n_col == 0:
                    display(ipw.HBox(cur_row))
                    cur_row = []
            if len(cur_row) != 0:
                display(ipw.HBox(cur_row))

        self.print_checked_pk_list(selected_replica_calc, check_list)
        
    def print_checked_pk_list(self, replica_calc, check_list):
        list_of_pks = [rep[2].pk for rep in replica_calc['replicas']]
        with self.output_checked_pks:
            clear_output()
            print("List of all replica PKs:")
            rep_pk_str = "["
            for pk, cb in zip(list_of_pks, check_list):
                if cb.value:
                    rep_pk_str += "%d " % pk
            print(rep_pk_str[:-1] + "]")
        
    
    def generate_energy_cv_plot(self, replica_calc):
        plot_energy = []
        plot_colvar = []
        wc_pk_str = str(replica_calc['wcs'][0].pk)
        
        for i, rep in enumerate(replica_calc['replicas']):
            cv_target, energy, struct_node = rep
            colvar_actual = struct_node.get_extra('replica_calcs')[wc_pk_str]['colvar_actual']
            plot_energy.append(energy)
            plot_colvar.append(colvar_actual)
            
        plot_energy = np.array(plot_energy)*27.2114
        plot_energy -= plot_energy[0]
            
        plt.figure(figsize=(10, 5))
        plt.ylabel('Energy/eV')
        plt.xlabel('Collective variable')
        plt.plot(plot_colvar, plot_energy, 'o-')
       
        for i, rep in enumerate(replica_calc['replicas']):
            # if the replica has no cv_target and has energy, it's an initial one, paint as red
            cv_target, energy, struct_node = rep
            if cv_target == None and energy is not None:
                plt.plot(plot_colvar[0], plot_energy[0], 'ro-')            
        
        plt.grid()
        plt.show()
            
        
    def generate_thumbnail_html(self, replica_calc):

        html_list = []        
        check_list = [] # check boxes to show in final pk list
        
        colvar_type = list(replica_calc['colvar_def'].keys())[0] # 'DISTANCE', ...
        
        wc_pk_str = str(replica_calc['wcs'][0].pk)
        
        for i, rep in enumerate(replica_calc['replicas']):
            
            cv_target, energy, struct_node = rep
            
            html = '<table>'
            
            struct_rep_info = struct_node.get_extra('replica_calcs')[wc_pk_str]
            d2prev = struct_rep_info['dist_previous']
            colvar_actual = float(struct_rep_info['colvar_actual'])
            thumbnail = struct_rep_info['thumbnail']
            
            cv_target_str = "-" if cv_target is None else "%.2f" % cv_target
            energy_str = "%.6f" % energy
            colvar_actual_str = "%.4f" % colvar_actual
            d2prev_str = "-" if d2prev == '-' else "%.4f" % float(d2prev)
            
            check_me = ipw.Checkbox(
                value=True,
                description='Check me',
                disabled=False,
                layout=layout
            )
            check_me.observe(lambda x, rc=replica_calc, cl=check_list: self.print_checked_pk_list(rc, cl), 'value')
            check_list.append(check_me)

            html += '<td><img width="400px" src="data:image/png;base64,{}" title="">'.format(thumbnail)

            # Output some information about the replica...
            html += '<p><b>Target: {}</b><br> <b>CV actual:</b> {}<br> <b>Energy:</b> {}<br> <b>d2prev:</b> {}</p>'\
                    .format(cv_target_str, colvar_actual_str, energy_str, d2prev_str)
            html += '<p>pk: {}</p>'.format(struct_node.pk)

            # ... and the download link.
            html += '<p><a target="_blank" href="../export_structure.ipynb?uuid={}">View & export</a></p><td>'\
                    .format(struct_node.uuid)

            html += '</table>'

            html_list.append(html)

        return html_list, check_list
       
    
    def get_replica_wcs(self, preprocessed = False):
        
        qb = QueryBuilder()
        
        if preprocessed:
            qb.append(WorkChainNode, tag='wc', filters={
                'attributes.process_label': 'ReplicaWorkChain',
                'and':[
                    {'extras': {'has_key': 'preproc_v'}},
                    {'extras.preproc_v': {'==': self.preprocess_version}},
                ]
            })
        else:
            qb.append(WorkChainNode, tag='wc', filters={
                'attributes.process_label': 'ReplicaWorkChain',
                'or':[
                    {'extras': {'!has_key': 'preproc_v'}},
                    {'extras.preproc_v': {'<': self.preprocess_version}},
                ]
            })
        
        qb.order_by({'wc': {'ctime': 'asc'}})

        return qb.all()

    
    def parse_rep_wcs(self, wc_list, existing_rep_sets=OrderedDict()):
        
        replica_sets = OrderedDict()
        
        rep_set_template = {
            'replicas'    : [], # (cv_target, energy, StructureData)
            'wcs'         : [],
            'colvar_def'  : None,
            'colvar_inc'  : None, # colvar increasing or decreasing ?
        }
        
        for wc_qb in wc_list:
            wc = wc_qb[0]
            
            wc_out_names = list(wc.outputs)
            
            if 'replica_0' not in wc_out_names:
                continue
            if 'energies' not in wc_out_names:
                continue
            
            name = wc.description
            cv_def = dict(wc.inputs['subsys_colvar'])
            cv_targets = [float(cvt) for cvt in wc.inputs['colvar_targets'].value.split()]
            # NB: there is no colvar target for replica_0 as that is the initial geometry
            cv_targets = [None] + cv_targets
            cv_inc = cv_targets[2] > cv_targets[1]
            energies = wc.outputs['energies'].get_array('energies')
            
            if len([n for n in wc_out_names if n.startswith('replica')]) != len(cv_targets):
                continue
            
            if name not in replica_sets:
                if name in existing_rep_sets:
                    # Already had a preprocessed part, add it
                    replica_sets[name] = copy.copy(existing_rep_sets[name])
                else:
                    # New replica set
                    replica_sets[name] = copy.deepcopy(rep_set_template)
                    replica_sets[name]['colvar_def'] = cv_def
                    replica_sets[name]['colvar_inc'] = cv_inc
                    
            # Does the current wc match with the replica set?
            if replica_sets[name]['colvar_def'] != cv_def or replica_sets[name]['colvar_inc'] != cv_inc:
                print("----")
                print("Warning! Replica calc CV definition doesn't match with previous ones.")
                print("Existing: %s %s" % (str([w.pk for w in replica_sets[name]['wcs']]),
                                           "cv_def: %s, cv_inc: %s" %(str(replica_sets[name]['colvar_def']),
                                                                      str(replica_sets[name]['colvar_inc']))))
                print("Skipping: %s %s " % (str(wc.pk), "cv_def: %s, cv_inc: %s" %(str(cv_def),
                                                                                   str(cv_inc))))
                print("----")
                continue
            
            # add it to the set
            replica_sets[name]['wcs'].append(wc)
            
            wc_replica_list = []
            for wc_o in sorted(list(wc.outputs)):
                if wc_o.startswith("replica_"):
                    struct = wc.outputs[wc_o]
                    # add description, if it doesn't exist already
                    if struct.description == "":
                        if struct.creator is None:
                            struct.description = "-"
                        else:
                            struct.description = struct.creator.description
                    wc_replica_list.append(wc.outputs[wc_o])
                    
            
            # Add the replicas of this wc to the set if it doesn't exist there already
            # (e.g. in case of continuation from another wc)
            for i_rep in range(len(wc_replica_list)):
                if wc_replica_list[i_rep].pk not in [s[2].pk for s in replica_sets[name]['replicas']]:
                    replica = (cv_targets[i_rep], energies[i_rep], wc_replica_list[i_rep])
                    replica_sets[name]['replicas'].append(replica)
            
            # Sort entries by cv target (e.g. one could be adding replicas in-between prev calculated ones)
            if cv_inc:
                replica_sets[name]['replicas'].sort(key=lambda x:(x[0] is not None, x[0], x[2].pk))
            else:
                replica_sets[name]['replicas'].sort(reverse=True, key=lambda x:(x[0] is not None, x[0], x[2].pk))
            
        return replica_sets
            

    def get_replica_distance(self, s1, s2):
        a1 = s1.get_positions()
        a2 = s2.get_positions()
        return np.linalg.norm(a1-a2)
    
    def render_thumbnail(self, atoms, vis_list=None):
        colors = None
        if vis_list:
            vis_list_atoms = [e for e in vis_list if isinstance(e, int)]
            colors = jmol_colors[atoms.numbers]
            for i_at in vis_list_atoms:
                colors[i_at] *= 0.6
                colors[i_at][0] = 1.0
        tmp = NamedTemporaryFile()
        ase.io.write(tmp.name, atoms, format='png', colors=colors) # does not accept StringIO
        raw = open(tmp.name, 'rb').read()
        tmp.close()
        return b64encode(raw).decode()
    
    def preprocess(self, replica_calc, overwrite_thumbnails=False):
        
        # Find all PKs of all work-calcs that contributed to this set
        print("wc pk: " + str([wc.pk for wc in replica_calc['wcs']]))
        
        progress = ipw.FloatProgress(description='Parsing images...', min=0, max=1, value=0.,
                                     style=style, layout=layout)
        display(progress)
            
        wc_preproc_failed = False
        
        n_rep = len(replica_calc['replicas'])
        
        # Set up the collective variable instance for evaluation
        colvar_type = list(replica_calc['colvar_def'].keys())[0]
        cv_instance = COLVARS[colvar_type].from_cp2k_subsys(replica_calc['colvar_def'])
        
        last_ase = None
        
        for i, rep in enumerate(replica_calc['replicas']):
            
            cv_target, energy, struct = rep
            
            progress.value = (i+1.)/n_rep
            prepoc_failed = False
            
            wc_pk_str = str(replica_calc['wcs'][0].pk)
            
            if 'replica_calcs' not in list(struct.extras):
                struct.set_extra('replica_calcs', {})
            
            rep_calcs = struct.extras['replica_calcs']
            
            if wc_pk_str not in rep_calcs:
                rep_calcs[wc_pk_str] = {}
            elif rep_calcs[wc_pk_str]['preproc_v'] == self.preprocess_version:
                last_ase = struct.get_ase()
                continue
            
            struct_ase = struct.get_ase()
            colvar_actual = cv_instance.eval_cv(struct_ase)
            
            if last_ase is not None:
                dist_previous = self.get_replica_distance(last_ase, struct_ase)
            else:
                dist_previous = '-'
            last_ase = struct_ase
            
            rep_calcs[wc_pk_str]['preproc_v']     = self.preprocess_version
            rep_calcs[wc_pk_str]['dist_previous'] = dist_previous
            rep_calcs[wc_pk_str]['colvar_actual'] = colvar_actual
            
            if 'thumbnail' not in rep_calcs[wc_pk_str] or overwrite_thumbnails:
                t = struct_ase
                vis_list = cv_instance.visualization_list(t)
                thumbnail = self.render_thumbnail(t, vis_list)
                rep_calcs[wc_pk_str]['thumbnail'] = thumbnail
            
            struct.set_extra('replica_calcs', rep_calcs)
            
            if prepoc_failed:
                wc_preproc_failed = True
                break
                
        for wc in replica_calc['wcs']:
            wc.set_extra('preproc_v', self.preprocess_version)
            wc.set_extra('preproc_failed', wc_preproc_failed)
            
    
    def preprocess_replicas(self):
        
        with self.out_preproc:
            print('Retrieving unparsed replica calculations...')
            reps_not_preproc = self.parse_rep_wcs(self.get_replica_wcs(False), existing_rep_sets=self.replica_calcs)
            print('Preprocessing {} replicas...'.format(len(reps_not_preproc.keys())))
        
        for i, k in enumerate(reps_not_preproc.keys()):
            with self.out_preproc:
                print('{}: {}/{}'.format(k, i+1, len(reps_not_preproc.keys())))
                self.preprocess(reps_not_preproc[k])

        with self.out_preproc:
            print('Done!')
            
        self.parse_preprocessed_replica_calcs()

        
        
        
        
        
    

