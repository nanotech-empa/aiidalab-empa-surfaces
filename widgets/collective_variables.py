
import numpy as np
import ipywidgets as ipw
from collections import OrderedDict
from IPython.display import display, clear_output, HTML

style = {'description_width': '120px'}
layout = {'width': '70%'}

### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------

class DistanceCV():
    
    def __init__(self, no_widget=False):
        
        self.spring_unit = 'eV/angstrom^2'
        self.target_unit = 'angstrom'
        
        self.text_colvar_atoms = None
        self.widget = None
        
        if not no_widget:
            self._create_widget()
        
        self.input_received = False
        
        # atom list (cp2k index convention: starts from 1)
        self.a_list = None
    
    @classmethod
    def from_cp2k_subsys(cls, cp2k_subsys):
        atoms_list = [int(x) for x in cp2k_subsys['DISTANCE']['ATOMS'].split()]
        cv = cls(no_widget=True)
        cv.a_list = atoms_list
        cv.input_received = True
        return cv        
        
    def _create_widget(self):
        self.text_colvar_atoms = ipw.Text(
            placeholder='1 2',
            description='Colvar Atoms',
            style=style, layout={'width': '60%'}
        )
        self.widget = ipw.VBox([self.text_colvar_atoms])
    
    def read_and_validate_inputs(self):
        try:
            a_list = [int(x) for x in self.text_colvar_atoms.value.split()]
        except:
            raise IOError("Error: wrong input for distance cv.")
        if len(a_list) != 2:
            raise IOError("Error: distance cv not two atoms.")
        self.a_list = a_list
        
        self.input_received = True
    
    
    def eval_cv(self, atoms):
        return np.linalg.norm(atoms[self.a_list[0]-1].position-atoms[self.a_list[1]-1].position)
            
    def print_cv(self, atoms):
        print('Distance between the atoms:')
        print(self.eval_cv(atoms))
    
    def visualization_list(self, atoms=None):
        if self.a_list is None:
            return []
        # index starts from 0
        return [i_a - 1 for i_a in self.a_list]

    def cp2k_subsys_inp(self):
        cp2k_subsys = { 'DISTANCE': {
            'ATOMS': self.text_colvar_atoms.value
        } }
        return cp2k_subsys

### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------

class AnglePlanePlaneCV():
    
    def __init__(self, no_widget = False):
        
        self.spring_unit = 'eV/deg^2'
        self.target_unit = 'deg'
        
        self.text_plane1_def = None
        self.toggle_plane2_type = None
        self.text_plane2_def = None
        self.widget = None
        
        if not no_widget:
            self._create_widget()
        
        self.input_received = False
        
        # definition lists for plane 1 and 2
        # includes either 3 atom indexes or vector xyz coordinates
        self.p1_def = None
        self.p2_def_type = None
        self.p2_def = None
        
    @classmethod
    def from_cp2k_subsys(cls, cp2k_subsys):
        
        cv = cls(no_widget=True)
        
        subsys_app = cp2k_subsys['ANGLE_PLANE_PLANE']
        
        cv.p1_def = np.array([int(x) for x in subsys_app['PLANE']['ATOMS'].split()])
        
        cv.p2_def_type = subsys_app['PLANE  ']['DEF_TYPE']
        
        if cv.p2_def_type == 'ATOMS':
            cv.p2_def = np.array([int(x) for x in subsys_app['PLANE  ']['ATOMS'].split()])
        else:
            cv.p2_def=np.array([float(x) for x in subsys_app['PLANE  ']['NORMAL_VECTOR'].split()])
        
        return cv
    
    def _create_widget(self):
        self.text_plane1_def = ipw.Text(placeholder='1 2 3',
                                   description='Plane 1 atoms',
                                   style=style, layout=layout)

        def on_plane2_type(c):
            self.text_plane2_def.description = self.toggle_plane2_type.value

        self.toggle_plane2_type = ipw.ToggleButtons(options=['ATOMS', 'VECTOR'],
                                               description='Plane 2 definition',
                                               style=style, layout=layout)
        self.toggle_plane2_type.observe(on_plane2_type, 'value')

        self.text_plane2_def = ipw.Text(placeholder='1 2 3',
                                   description='Atoms',
                                   style=style, layout=layout)
        
        
        self.widget = ipw.VBox([self.text_plane1_def,
                                self.toggle_plane2_type, self.text_plane2_def])
    
    def read_and_validate_inputs(self):
        try:
            self.p1_def = np.array([int(x) for x in self.text_plane1_def.value.split()])
        except:
            raise IOError("Error: wrong input for plane 1 definition.")
        if len(self.p1_def) != 3:
            raise IOError("Error: plane 1 needs 3 atoms.")
        
        self.p2_def_type = self.toggle_plane2_type.value
        
        if self.p2_def_type == 'ATOMS':
            try:
                self.p2_def = np.array([int(x) for x in self.text_plane2_def.value.split()])
            except:
                raise IOError("Error: wrong input for plane 2 definition.")
            if len(self.p2_def) != 3:
                raise IOError("Error: plane 2 needs 3 atoms.")
        else:
            try:
                self.p2_def=np.array([float(x) for x in self.text_plane2_def.value.split()])
            except:
                raise IOError("Error: wrong input for plane 2 definition.")
            if len(self.p2_def) != 3:
                raise IOError("Error: plane 2 normal needs 3 coordinates.")
                
        self.input_received = True
    
    
    def _cp2k_plane_normal(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        vec = np.cross(v1, v2)
        return vec/np.linalg.norm(vec)
    
    def _p1_normal(self, atoms):
        # NB until here, the indexes use cp2k convention (starts from 1)
        return self._cp2k_plane_normal(
            atoms[self.p1_def[0]-1].position,
            atoms[self.p1_def[1]-1].position,
            atoms[self.p1_def[2]-1].position
        )
    
    def _p2_normal(self, atoms):
        # NB until here, the indexes use cp2k convention (starts from 1)
        if self.p2_def_type == 'ATOMS':
            return self._cp2k_plane_normal(
                atoms[self.p2_def[0]-1].position,
                atoms[self.p2_def[1]-1].position,
                atoms[self.p2_def[2]-1].position
            )
        else:
            return self.p2_def/np.linalg.norm(self.p2_def)
            
    
    def eval_cv(self, atoms):
        """
        Evaluates the ANGLE PLANE PLANE collective variable according to cp2k conventions
        In case of defining 3 atoms a1, a2 and a3, the plane normal is defined as
        (a1 - a2) x (a3 - a2)
        """
        cosine = np.dot(self._p1_normal(atoms), self._p2_normal(atoms))
        angle = np.arccos(cosine)*180./np.pi
        return angle
        
    def print_cv(self, atoms):
        print('Angle between the planes:')
        print(self.eval_cv(atoms))
    
    def visualization_list(self, atoms):
        
        if self.p1_def is None:
            return []
        
        # atom indexes start from 0
        vis_list = list(self.p1_def - 1)
        
        # add middle point of p1 and a point along the normal
        p1_middle = np.mean(atoms[list(self.p1_def - 1)].positions, axis=0)
        vis_list.append(p1_middle)
        vis_list.append(p1_middle + 3.0 * self._p1_normal(atoms))
        
        if self.p2_def_type == 'ATOMS':
            vis_list.extend(list(self.p2_def - 1))
            p2_middle = np.mean(atoms[list(self.p2_def - 1)].positions, axis=0)
            vis_list.append(p2_middle)
            vis_list.append(p2_middle + 3.0 * self._p2_normal(atoms))
        else:
            vis_list.append(p1_middle + 3.0 * self._p2_normal(atoms))
        
        return vis_list

    def cp2k_subsys_inp(self):

        repl = {
            'ATOMS': 'ATOMS',
            'VECTOR': 'NORMAL_VECTOR'
        }
        
        cp2k_subsys = { 'ANGLE_PLANE_PLANE': {
            'PLANE': {
                'DEF_TYPE': 'ATOMS',
                'ATOMS': " ".join(self.p1_def)
            },
            'PLANE  ': {
                'DEF_TYPE': self.p2_def_type,
                repl[self.p2_def_type]: " ".join([str(x) for x in self.p2_def])
            },
        } }
        
        return cp2k_subsys

### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
    
class BondRotationCV():
    
    def __init__(self, no_widget = False):
        
        self.spring_unit = 'eV/deg^2'
        self.target_unit = 'deg'
        
        # Create the widget
        
        self.bond_point_texts = None

        self.bond_point_btns = None
        self.bond_point_textbs = None        
        
        self.widget = None
        
        if not no_widget:
            self._create_widget()
            
        self.input_received = False
        
        self.types_list = None
        self.data_txt_list = None
        self.data_list = None # atom indexes start from 0
        
    @classmethod
    def from_cp2k_subsys(cls, cp2k_subsys):
        cv = cls(no_widget = True)
        point_list = cp2k_subsys['BOND_ROTATION']['POINT']
        
        cv.types_list = []
        cv.data_list = []
        
        for p in point_list:
            cv.types_list.append(p['TYPE'])
            if p['TYPE'] == 'GEO_CENTER':
                cv.data_list.append([int(x)-1 for x in p['ATOMS'].split()])
            else:
                cv.data_list.append([float(x) for x in p['XYZ'].split()])
            
        cv.input_received = True
        return cv
        
    def _create_widget(self):
        
        self.bond_point_texts = [
            '1st point 1st line',
            '2nd point 1st line',
            '1st point 2nd line',
            '2nd point 2nd line',
        ]

        self.bond_point_btns = []
        self.bond_point_textbs = []

        self.textbox_defaults = {
            'GEO_CENTER': (            '1 2 3',  'atom index(es)'),
            'FIX_POINT':  ('18.20 22.15 20.10', 'position in Angstrom'),
        }

        def on_bond_point_type_toggle(c, tb):
            tb.placeholder = self.textbox_defaults[c.new][0]
            tb.description = self.textbox_defaults[c.new][1]

        for i_p, bond_point_text in enumerate(self.bond_point_texts):

            toggle_button = ipw.ToggleButtons(
                options=['GEO_CENTER', 'FIX_POINT'],
                description=bond_point_text,
                style=style, layout=layout)

            textbox = ipw.Text(
                placeholder=self.textbox_defaults[toggle_button.value][0],
                description=self.textbox_defaults[toggle_button.value][1],
                style=style, layout=layout)

            toggle_button.observe(lambda c, tb=textbox: on_bond_point_type_toggle(c, tb), names='value')
            
            self.bond_point_btns.append(toggle_button)
            self.bond_point_textbs.append(textbox)
        
        self.widget = ipw.VBox([x for ab in zip(self.bond_point_btns, self.bond_point_textbs) for x in ab])
        
    
    def read_and_validate_inputs(self):
        
        self.types_list = []
        self.data_txt_list = []
        self.data_list = []
        
        for i_p, btn in enumerate(self.bond_point_btns): 
            typ = btn.value
            self.types_list.append(typ)
            
            if typ == 'GEO_CENTER':
                try:
                    dl = np.array([int(x) - 1 for x in self.bond_point_textbs[i_p].value.split()])
                except:
                    raise IOError("Error: wrong input for '%s'" % self.bond_point_texts[i_p])
            else:
                try:
                    dl = np.array([float(x) for x in self.bond_point_textbs[i_p].value.split()])
                except:
                    raise IOError("Error: wrong input for '%s'" % self.bond_point_texts[i_p])
                if len(dl) != 3:
                    raise IOError("Error: '%s' needs x,y,z" % self.bond_point_texts[i_p])
            self.data_list.append(dl)
            self.data_txt_list.append(self.bond_point_textbs[i_p].value)
                
        self.input_received = True
            
    
    def _point_list(self, atoms):
        p_list = []
        for typ, d in zip(self.types_list, self.data_list):
            if typ == 'GEO_CENTER':
                p_list.append(np.mean(atoms[d].positions, axis=0))
            else:
                p_list.append(d)
        return p_list
        
    def eval_cv(self, atoms):
        p_list = np.array(self._point_list(atoms))
        v1 = p_list[1] - p_list[0]
        v2 = p_list[3] - p_list[2]
        
        cosine = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        angle = np.arccos(cosine)*180./np.pi
        return angle
        
    def print_cv(self, atoms):
        print('Angle between the two lines:')
        print(self.eval_cv(atoms))
    
    def visualization_list(self, atoms):
        return self._point_list(atoms)

    def cp2k_subsys_inp(self):

        repl = {
            'GEO_CENTER': 'ATOMS',
            'FIX_POINT': 'XYZ'
        }

        point_list = [{'TYPE':typ, repl[typ]:text} for typ, text in zip(self.types_list, self.data_txt_list)]

        cp2k_subsys = { 'BOND_ROTATION': {
            'POINT': point_list,
            'P1_BOND1':'1',
            'P2_BOND1':'2',
            'P1_BOND2':'3',
            'P2_BOND2':'4'
        }}
        return cp2k_subsys

### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------

COLVARS = {
    'DISTANCE': DistanceCV,
    'ANGLE_PLANE_PLANE': AnglePlanePlaneCV,
    'BOND_ROTATION': BondRotationCV,
}

class CollectiveVariableWidget(ipw.VBox):
    
    def __init__(self, viewer_widget=None, **kwargs):
        
        self.current_cv_instance = None
        
        def on_choose_colvar(c):
            self.current_cv_instance = self.drop_colvar_type.value()
            with self.out_colvar:
                clear_output()
                display(self.current_cv_instance.widget)

        self.drop_colvar_type = ipw.Select(
            options=COLVARS,
            description='Colvar Type',
            style=style, layout=layout
        )

        self.drop_colvar_type.observe(on_choose_colvar, 'value')
        
        self.out_colvar = ipw.Output(layout={'border': '1px solid #ccc'})


        self.text_colvar_targets = ipw.Text(placeholder='0.9 1.3 1.7 2.4',
                                       description='Colvar Targets',
                                       style=style, layout=layout)

        self.visualize_colvar_btn = ipw.Button(description='Visualize CV',
                                          style=style, layout=layout)

        self.float_spring = ipw.FloatText(description='Spring constant',
                                 value=30.0,
                                 style=style, layout=layout)
        
        self.error_out = ipw.Output()

        on_choose_colvar('')
        
        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX
        children = [
            self.drop_colvar_type,
            self.out_colvar,
            self.visualize_colvar_btn,
            self.text_colvar_targets,
            self.float_spring,
            self.error_out
        ]
        super(CollectiveVariableWidget, self).__init__(children=children, **kwargs)
        ### ---------------------------------------------------------
    
    def validation_check(self):
        try:
            self.current_cv_instance.read_and_validate_inputs()
        except Exception as e:
            with self.error_out:
                print(e.message)
            return False
        return True
    
    def set_job_details(self, job_details):
        if not self.validation_check():
            return False
        
        #TODO also check the inputs below
        
        job_details['colvar_targets'] = self.text_colvar_targets.value
        job_details['spring'] = self.float_spring.value
        job_details['spring_unit'] = self.current_cv_instance.spring_unit
        job_details['target_unit'] = self.current_cv_instance.target_unit
        job_details['subsys_colvar'] = self.current_cv_instance.cp2k_subsys_inp()
        
        return True
        

### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------
### --------------------------------------------------------------------------------------------------

from aiida.orm.querybuilder import QueryBuilder

from replicawork import ReplicaWorkchain
from aiida.engine.process import WorkCalculation
from aiida_cp2k.calculations import Cp2kCalculation
from aiida.orm import StructureData
from aiida.orm import Dict

from aiida.orm import load_node 

import ipywidgets as ipw
from IPython.display import display, clear_output, HTML
import nglview
import time
import ase.io
import ase.units as aseu
from ase.data.colors import jmol_colors
import urlparse
import numpy as np
import copy

import re
from collections import OrderedDict

import matplotlib.pyplot as plt
from pprint import pprint

from tempfile import NamedTemporaryFile
from base64 import b64encode



class SearchReplicaWidget(ipw.VBox):
    
    def __init__(self, **kwargs):
        
        self.preprocess_version = 0.20
        
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
        list_of_pks = [struct.pk for struct in replica_calc['structs']]
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
        
        for i, struct_node in enumerate(replica_calc['structs']):
            colvar_actual = struct_node.get_extra('replica_calcs')[wc_pk_str]['colvar_actual']
            if replica_calc['info'][i]['energy'] is not None:
                plot_energy.append(replica_calc['info'][i]['energy'])
                plot_colvar.append(colvar_actual)
            
        plot_energy = np.array(plot_energy)*27.2114
        plot_energy -= plot_energy[0]
            
        plt.figure(figsize=(10, 5))
        plt.ylabel('Energy/eV')
        plt.xlabel('Collective variable')
        plt.plot(plot_colvar, plot_energy, 'o-')
        
        if replica_calc['info'][0]['colvar_target'] == None and replica_calc['info'][0]['energy'] is not None:
            plt.plot(plot_colvar[0], plot_energy[0], 'ro-')
        
        plt.grid()
        plt.show()
            
        
    def generate_thumbnail_html(self, replica_calc):

        html_list = []        
        check_list = [] # check boxes to show in final pk list
        
        colvar_type = replica_calc['colvar_def'].keys()[0] # 'DISTANCE', ...
        
        wc_pk_str = str(replica_calc['wcs'][0].pk)
        
        for i, struct_node in enumerate(replica_calc['structs']):
            
            html = '<table>'
            
            struct_rep_info = struct_node.get_extra('replica_calcs')[wc_pk_str]
            d2prev = struct_rep_info['dist_previous']
            colvar_actual = struct_rep_info['colvar_actual']
            thumbnail = struct_rep_info['thumbnail']
            
            if replica_calc['info'][i]['colvar_target'] is not None:
                colvar_target = replica_calc['info'][i]['colvar_target']['TARGET'].split()[1]
            else:
                colvar_target = "-"
            energy = replica_calc['info'][i]['energy']

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
            html += '<p><b>Target: {}</b> ({})<br> <b>Energy:</b> {}<br> <b>d2prev:</b> {}</p>'\
                    .format(colvar_target, colvar_actual, energy, d2prev)
            html += '<p>pk: {}</p>'.format(struct_node.pk)

            # ... and the download link.
            html += '<p><a target="_blank" href="./export_structure.ipynb?uuid={}">Export Structure</a></p><td>'\
                    .format(struct_node.uuid)

            html += '</table>'

            html_list.append(html)

        return html_list, check_list
       
    
    def get_replica_wcs(self, preprocessed = False):
        
        qb = QueryBuilder()
        
        if preprocessed:
            qb.append(WorkCalculation, tag='wc', filters={
                'attributes._process_label': 'ReplicaWorkchain',
                'and':[
                    {'extras': {'has_key': 'preproc_v'}},
                    {'extras.preproc_v': {'==': self.preprocess_version}},
                ]
            })
        else:
            qb.append(WorkCalculation, tag='wc', filters={
                'attributes._process_label': 'ReplicaWorkchain',
                'or':[
                    {'extras': {'!has_key': 'preproc_v'}},
                    {'extras.preproc_v': {'<': self.preprocess_version}},
                ]
            })
        
        qb.order_by({'wc': {'ctime': 'asc'}})
        
        #return [[load_node(116530)]]

        return qb.all()
    
    def _get_cp2k_struct_info(self, struct, initial=False):
        
        info = {
            'energy'        : None,
            'description'   : None,
            'colvar_def'    : None,
            'colvar_target' : None,
        }
        
        def find_colvar(force_eval):
            if isinstance(force_eval, list):
                # in case of multiple force evals,
                # find the one, which has the cv defined
                for f in force_eval:
                    if f['SUBSYS'].has_key('COLVAR'):
                        return f['SUBSYS']['COLVAR']
            else:
                return force_eval['SUBSYS']['COLVAR']
        
        if 'output_structure' not in struct.get_inputs_dict():
            return info
        
        parent_cp2k = struct.get_inputs_dict()['output_structure']
        
        if 'output_parameters' not in parent_cp2k.get_outputs_dict():
            return info
        
        cp2k_out_params = parent_cp2k.get_outputs_dict()['output_parameters'].get_attrs()
        
        info['energy'] = cp2k_out_params['energy']
        
        cp2k_inp_params = parent_cp2k.get_inputs_dict()['parameters'].get_attrs()
        
        if not initial and 'COLLECTIVE' in cp2k_inp_params['MOTION']['CONSTRAINT']:
            info['description'] = parent_cp2k.description
            info['colvar_def'] = find_colvar(cp2k_inp_params['FORCE_EVAL'])
            info['colvar_target'] = cp2k_inp_params['MOTION']['CONSTRAINT']['COLLECTIVE']
        
        return info
    
    def parse_rep_wcs(self, wc_list, existing_rep_sets=OrderedDict()):
        
        replica_sets = OrderedDict()
        
        rep_set_template = {
            'structs'     : [],
            'info'        : [],
            'wcs'         : [],
            'colvar_def'  : None,
            'colvar_inc'  : None, # colvar increasing or decreasing ?
        }
        
        for wc in wc_list:
            
            wc_out = wc[0].get_outputs_dict()
            if 'CALL' not in wc_out:
                continue
            
            cp2k_pk_keys = [(int(x.split('_')[1]), x) for x in wc_out.keys() if "CALL_" in x]
            
            name = wc[0].description
            
            
            if name not in replica_sets:
                if name in existing_rep_sets:
                    # Already had a preprocessed part, add it
                    replica_sets[name] = copy.deepcopy(existing_rep_sets[name])
                else:
                    # New replica set
                    replica_sets[name] = copy.deepcopy(rep_set_template)
                
            replica_sets[name]['wcs'].append(wc[0])
            
            # add the initial structure as a "replica"
            inp_structure = wc[0].get_inputs_dict()['structure']
            if inp_structure.pk not in [s.pk for s in replica_sets[name]['structs']]:
                replica_sets[name]['structs'].append(inp_structure)
                replica_sets[name]['info'].append(self._get_cp2k_struct_info(inp_structure, initial=True))
                
            last_cv_target = None
            
            for pk, k in sorted(cp2k_pk_keys):
                cp2k_calc = wc_out[k]
                if 'output_structure' not in cp2k_calc.get_outputs_dict():
                    continue
                if 'output_parameters' not in cp2k_calc.get_outputs_dict():
                    continue
                if cp2k_calc.get_outputs_dict()['output_parameters'].get_attrs()['exceeded_walltime']:
                    continue
                    
                cp2k_out_struct = cp2k_calc.get_outputs_dict()['output_structure']
                if cp2k_out_struct.pk not in [s.pk for s in replica_sets[name]['structs']]:
                    replica_sets[name]['structs'].append(cp2k_out_struct)
                    replica_sets[name]['info'].append(self._get_cp2k_struct_info(cp2k_out_struct))
                    
                    if replica_sets[name]['colvar_def'] is None:
                        replica_sets[name]['colvar_def'] = replica_sets[name]['info'][-1]['colvar_def']
                    
                    cv_target = replica_sets[name]['info'][-1]['colvar_target']
                    if last_cv_target is not None:
                        replica_sets[name]['colvar_inc'] = cv_target > last_cv_target
                    last_cv_target = cv_target
                
        # filter and sort the entries by colvar_target
        for key in replica_sets:
            
            if replica_sets[key]['colvar_def'] is None:
                del replica_sets[key]
                continue
            
            structs = replica_sets[key]['structs']
            info = replica_sets[key]['info']
            cv_targets = [ i['colvar_target'] for i in info ]
            for i in range(len(cv_targets)):
                if cv_targets[i] is not None:
                    cv_targets[i] = float(cv_targets[i]['TARGET'].split()[1])
            
            zipped = zip(cv_targets, structs, info)
            if replica_sets[key]['colvar_inc']:
                zipped.sort(key=lambda x:(x[0] is not None, x[0], x[1].pk))
            else:
                zipped.sort(reverse=True, key=lambda x:(x[0] is None, x[0], -x[1].pk))
            
            cv_targets, structs, info = zip(*zipped)
            replica_sets[key]['structs'] = list(structs)
            replica_sets[key]['info'] = list(info)
            
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
        raw = open(tmp.name).read()
        tmp.close()
        return b64encode(raw)
    
    def preprocess(self, replica_calc, overwrite_thumbnails=False):
        
        # Find all PKs of all work-calcs that contributed to this set
        print("wc pk: " + str([wc.pk for wc in replica_calc['wcs']]))
        
        progress = ipw.FloatProgress(description='Parsing images...', min=0, max=1, value=0.,
                                     style=style, layout=layout)
        display(progress)
            
        wc_preproc_failed = False
        
        n_rep = len(replica_calc['structs'])
        
        # Set up the collective variable instance for evaluation
        colvar_type = replica_calc['colvar_def'].keys()[0]
        cv_instance = COLVARS[colvar_type].from_cp2k_subsys(replica_calc['colvar_def'])
        
        last_ase = None

        for i, (struct, info) in enumerate(zip(replica_calc['structs'], replica_calc['info'])):
            progress.value = (i+1.)/n_rep
            prepoc_failed = False

            wc_pk_str = str(replica_calc['wcs'][0].pk)
            
            if 'replica_calcs' not in struct.get_extras():
                struct.set_extra('replica_calcs', {})
                
            rep_calcs = struct.get_extra('replica_calcs')
            
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
            
            if info['description'] is not None and struct.description == "":
                struct.description = info['description']
            
            if prepoc_failed:
                wc_preproc_failed = True
                break
        
        for wc in replica_calc['wcs']:
            wc.set_extras({
                'preproc_v': self.preprocess_version,
                'preproc_failed': wc_preproc_failed,
            })
    
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

        
        
        
        
        
    

