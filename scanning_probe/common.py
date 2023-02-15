import os
from aiida.orm import WorkChainNode
from aiida.orm import load_node

from aiida.orm.querybuilder import QueryBuilder
from aiida.orm import SinglefileData, ArrayData
from aiida.orm import Code, Computer
from aiida.engine import CalcJob

import subprocess


import numpy as np

from io import StringIO
import tempfile
import shutil

# ## ----------------------------------------------------------------
# ## ----------------------------------------------------------------
# ## ----------------------------------------------------------------
# ## BS & PP

ATOMIC_KIND_INFO = {
    'H':  {'basis': 'TZV2P-MOLOPT-GTH',        'pseudo': 'GTH-PBE-q1' },
    'He': {'basis': 'DZVP-MOLOPT-SR-GTH-q2',   'pseudo': 'GTH-PBE-q2' },
    'Li': {'basis': 'DZVP-MOLOPT-SR-GTH-q3',   'pseudo': 'GTH-PBE-q3' },
    'Be': {'basis': 'DZVP-MOLOPT-SR-GTH-q4',   'pseudo': 'GTH-PBE-q4' },
    'B':  {'basis': 'DZVP-MOLOPT-SR-GTH-q3',   'pseudo': 'GTH-PBE-q3' },
    'C':  {'basis': 'TZV2P-MOLOPT-GTH',        'pseudo': 'GTH-PBE-q4' },
    'N':  {'basis': 'TZV2P-MOLOPT-GTH',        'pseudo': 'GTH-PBE-q5' },
    'O':  {'basis': 'TZV2P-MOLOPT-GTH',        'pseudo': 'GTH-PBE-q6' },
    'F':  {'basis': 'DZVP-MOLOPT-SR-GTH-q7',   'pseudo': 'GTH-PBE-q7' },
    'Ne': {'basis': 'DZVP-MOLOPT-SR-GTH-q8',   'pseudo': 'GTH-PBE-q8' },
    'Na': {'basis': 'DZVP-MOLOPT-SR-GTH-q9',   'pseudo': 'GTH-PBE-q9' },
    'Mg': {'basis': 'DZVP-MOLOPT-SR-GTH-q2',   'pseudo': 'GTH-PBE-q2' },
    'Al': {'basis': 'DZVP-MOLOPT-SR-GTH',      'pseudo': 'GTH-PBE-q3' },
    'Si': {'basis': 'DZVP-MOLOPT-SR-GTH-q4',   'pseudo': 'GTH-PBE-q4' },
    'P':  {'basis': 'DZVP-MOLOPT-SR-GTH-q5',   'pseudo': 'GTH-PBE-q5' },
    'S':  {'basis': 'TZV2P-MOLOPT-GTH',        'pseudo': 'GTH-PBE-q6' },
    'Cl': {'basis': 'TZV2P-MOLOPT-GTH',        'pseudo': 'GTH-PBE-q7' },
    'Ar': {'basis': 'DZVP-MOLOPT-SR-GTH-q8',   'pseudo': 'GTH-PBE-q8' },
    'K':  {'basis': 'DZVP-MOLOPT-SR-GTH-q9',   'pseudo': 'GTH-PBE-q9' },
    'Ca': {'basis': 'DZVP-MOLOPT-SR-GTH-q10',  'pseudo': 'GTH-PBE-q10' },
    'Sc': {'basis': 'DZVP-MOLOPT-SR-GTH-q11',  'pseudo': 'GTH-PBE-q11' },
    'Ti': {'basis': 'DZVP-MOLOPT-SR-GTH-q12',  'pseudo': 'GTH-PBE-q12' },
    'V':  {'basis': 'DZVP-MOLOPT-SR-GTH-q13',  'pseudo': 'GTH-PBE-q13' },
    'Cr': {'basis': 'DZVP-MOLOPT-SR-GTH-q14',  'pseudo': 'GTH-PBE-q14' },
    'Mn': {'basis': 'DZVP-MOLOPT-SR-GTH-q15',  'pseudo': 'GTH-PBE-q15' },
    'Fe': {'basis': 'TZV2P-MOLOPT-SR-GTH-q16', 'pseudo': 'GTH-PBE-q16' },
    'Co': {'basis': 'DZVP-MOLOPT-SR-GTH-q17',  'pseudo': 'GTH-PBE-q17' },
    'Ni': {'basis': 'DZVP-MOLOPT-SR-GTH-q18',  'pseudo': 'GTH-PBE-q18' },
    'Cu': {'basis': 'DZVP-MOLOPT-SR-GTH-q11',  'pseudo': 'GTH-PBE-q11' },
    'Zn': {'basis': 'DZVP-MOLOPT-SR-GTH-q12',  'pseudo': 'GTH-PBE-q12' },
    'Ga': {'basis': 'DZVP-MOLOPT-SR-GTH-q13',  'pseudo': 'GTH-PBE-q13' },
    'Ge': {'basis': 'DZVP-MOLOPT-SR-GTH-q4',   'pseudo': 'GTH-PBE-q4' },
    'As': {'basis': 'DZVP-MOLOPT-SR-GTH-q5',   'pseudo': 'GTH-PBE-q5' },
    'Se': {'basis': 'DZVP-MOLOPT-SR-GTH-q6',   'pseudo': 'GTH-PBE-q6' },
    'Br': {'basis': 'DZVP-MOLOPT-SR-GTH-q7',   'pseudo': 'GTH-PBE-q7' },
    'Kr': {'basis': 'DZVP-MOLOPT-SR-GTH-q8',   'pseudo': 'GTH-PBE-q8' },
    'Rb': {'basis': 'DZVP-MOLOPT-SR-GTH-q9',   'pseudo': 'GTH-PBE-q9' },
    'Sr': {'basis': 'DZVP-MOLOPT-SR-GTH-q10',  'pseudo': 'GTH-PBE-q10' },
    'Y':  {'basis': 'DZVP-MOLOPT-SR-GTH-q11',  'pseudo': 'GTH-PBE-q11' },
    'Zr': {'basis': 'DZVP-MOLOPT-SR-GTH-q12',  'pseudo': 'GTH-PBE-q12' },
    'Nb': {'basis': 'DZVP-MOLOPT-SR-GTH-q13',  'pseudo': 'GTH-PBE-q13' },
    'Mo': {'basis': 'DZVP-MOLOPT-SR-GTH-q14',  'pseudo': 'GTH-PBE-q14' },
    'Tc': {'basis': 'DZVP-MOLOPT-SR-GTH-q15',  'pseudo': 'GTH-PBE-q15' },
    'Ru': {'basis': 'DZVP-MOLOPT-SR-GTH-q16',  'pseudo': 'GTH-PBE-q16' },
    'Rh': {'basis': 'DZVP-MOLOPT-SR-GTH-q9',   'pseudo': 'GTH-PBE-q9' },
    'Pd': {'basis': 'DZVP-MOLOPT-SR-GTH-q18',  'pseudo': 'GTH-PBE-q18' },
    'Ag': {'basis': 'DZVP-MOLOPT-SR-GTH-q11',  'pseudo': 'GTH-PBE-q11' },
    'Cd': {'basis': 'DZVP-MOLOPT-SR-GTH-q12',  'pseudo': 'GTH-PBE-q12' },
    'In': {'basis': 'DZVP-MOLOPT-SR-GTH-q13',  'pseudo': 'GTH-PBE-q13' },
    'Sn': {'basis': 'DZVP-MOLOPT-SR-GTH-q4',   'pseudo': 'GTH-PBE-q4' },
    'Sb': {'basis': 'DZVP-MOLOPT-SR-GTH-q5',   'pseudo': 'GTH-PBE-q5' },
    'Te': {'basis': 'DZVP-MOLOPT-SR-GTH-q6',   'pseudo': 'GTH-PBE-q6' },
    'I':  {'basis': 'DZVP-MOLOPT-SR-GTH-q7',   'pseudo': 'GTH-PBE-q7' },
    'Xe': {'basis': 'DZVP-MOLOPT-SR-GTH-q8',   'pseudo': 'GTH-PBE-q8' },
    'Cs': {'basis': 'DZVP-MOLOPT-SR-GTH-q9',   'pseudo': 'GTH-PBE-q9' },
    'Ba': {'basis': 'DZVP-MOLOPT-SR-GTH-q10',  'pseudo': 'GTH-PBE-q10' },
    'La': {'basis': 'DZVP-MOLOPT-SR-GTH-q11',  'pseudo': 'GTH-PBE-q11' },
    'Ce': {'basis': 'DZVP-MOLOPT-SR-GTH-q12',  'pseudo': 'GTH-PBE-q12' },
    'Pr': {'basis': 'DZVP-MOLOPT-SR-GTH-q13',  'pseudo': 'GTH-PBE-q13' },
    'Nd': {'basis': 'DZVP-MOLOPT-SR-GTH-q14',  'pseudo': 'GTH-PBE-q14' },
    'Pm': {'basis': 'DZVP-MOLOPT-SR-GTH-q15',  'pseudo': 'GTH-PBE-q15' },
    'Sm': {'basis': 'DZVP-MOLOPT-SR-GTH-q16',  'pseudo': 'GTH-PBE-q16' },
    'Eu': {'basis': 'DZVP-MOLOPT-SR-GTH-q17',  'pseudo': 'GTH-PBE-q17' },
    'Gd': {'basis': 'DZVP-MOLOPT-SR-GTH-q18',  'pseudo': 'GTH-PBE-q18' },
    'Tb': {'basis': 'DZVP-MOLOPT-SR-GTH-q19',  'pseudo': 'GTH-PBE-q19' },
    'Dy': {'basis': 'DZVP-MOLOPT-SR-GTH-q20',  'pseudo': 'GTH-PBE-q20' },
    'Ho': {'basis': 'DZVP-MOLOPT-SR-GTH-q21',  'pseudo': 'GTH-PBE-q21' },
    'Er': {'basis': 'DZVP-MOLOPT-SR-GTH-q22',  'pseudo': 'GTH-PBE-q22' },
    'Tm': {'basis': 'DZVP-MOLOPT-SR-GTH-q23',  'pseudo': 'GTH-PBE-q23' },
    'Yb': {'basis': 'DZVP-MOLOPT-SR-GTH-q24',  'pseudo': 'GTH-PBE-q24' },
    'Lu': {'basis': 'DZVP-MOLOPT-SR-GTH-q25',  'pseudo': 'GTH-PBE-q25' },
    'Hf': {'basis': 'DZVP-MOLOPT-SR-GTH-q12',  'pseudo': 'GTH-PBE-q12' },
    'Ta': {'basis': 'DZVP-MOLOPT-SR-GTH-q13',  'pseudo': 'GTH-PBE-q13' },
    'W':  {'basis': 'DZVP-MOLOPT-SR-GTH-q14',  'pseudo': 'GTH-PBE-q14' },
    'Re': {'basis': 'DZVP-MOLOPT-SR-GTH-q15',  'pseudo': 'GTH-PBE-q15' },
    'Os': {'basis': 'DZVP-MOLOPT-SR-GTH-q16',  'pseudo': 'GTH-PBE-q16' },
    'Ir': {'basis': 'DZVP-MOLOPT-SR-GTH-q17',  'pseudo': 'GTH-PBE-q17' },
    'Pt': {'basis': 'DZVP-MOLOPT-SR-GTH-q18',  'pseudo': 'GTH-PBE-q18' },
    'Au': {'basis': 'DZVP-MOLOPT-SR-GTH-q11',  'pseudo': 'GTH-PBE-q11' },
    'Hg': {'basis': 'DZVP-MOLOPT-SR-GTH-q12',  'pseudo': 'GTH-PBE-q12' },
    'Tl': {'basis': 'DZVP-MOLOPT-SR-GTH-q13',  'pseudo': 'GTH-PBE-q13' },
    'Pb': {'basis': 'DZVP-MOLOPT-SR-GTH-q4',   'pseudo': 'GTH-PBE-q4' },
    'Bi': {'basis': 'DZVP-MOLOPT-SR-GTH-q5',   'pseudo': 'GTH-PBE-q5' },
    'Po': {'basis': 'DZVP-MOLOPT-SR-GTH-q6',   'pseudo': 'GTH-PBE-q6' },
    'At': {'basis': 'DZVP-MOLOPT-SR-GTH-q7',   'pseudo': 'GTH-PBE-q7' },
    'Rn': {'basis': 'DZVP-MOLOPT-SR-GTH-q8',   'pseudo': 'GTH-PBE-q8' },
}

# ## ----------------------------------------------------------------
# ## ----------------------------------------------------------------
# ## ----------------------------------------------------------------
# ## Preprocessing and viewer links

# This code and the way it's processed is to support
# multiple pre/postprocess versions of the same calculation
workchain_preproc_and_viewer_info = {
    'STMWorkChain': {
        # version : {info}
        0: { 
            'n_calls': 2,
            'viewer_path': "scanning_probe/stm/view_stm.ipynb",
            'retrieved_files': [(1, ["stm.npz"])], # [(step_index, list_of_retr_files), ...]
            'struct_label': 'structure',
        },
    },
    'PdosWorkChain': {
        0: {
            'n_calls': 3,
            'viewer_path': "scanning_probe/pdos/view_pdos.ipynb",
            'retrieved_files': [(0, ["aiida-list1-1.pdos"]), (2, ["overlap.npz"])],
            'struct_label': 'slabsys_structure',
        },
    },
    'AfmWorkChain': {
        0: {
            'n_calls': 3,
            'viewer_path': "scanning_probe/afm/view_afm.ipynb",
            'retrieved_files': [(1, ["df.npy"]), (2, ["df.npy"])],
            'struct_label': 'structure',
        },
    },
    'OrbitalWorkChain': {
        0: {
            'n_calls': 2,
            'viewer_path': "scanning_probe/orb/view_orb.ipynb",
            'retrieved_files': [(1, ["orb.npz"])],
            'struct_label': 'structure',
        },
    },
    'HRSTMWorkChain': {
        0: {
            'n_calls': 3,
            'viewer_path': "scanning_probe/hrstm/view_hrstm.ipynb",
            'retrieved_files': [(1, ["df.npy"]), (2, ['hrstm_meta.npy', 'hrstm.npz'])],
            'struct_label': 'structure',
        },
    },
}


PREPROCESS_VERSION = 1.08

def preprocess_one(workcalc):
    """
    Preprocess one SPM calc
    Supports preprocess of multiple versions
    """
    
    workcalc_name = workcalc.attributes['process_label']
    
    if 'version' in workcalc.extras:
        workcalc_version = workcalc.extras['version']
    else:
        workcalc_version = 0
        
    prepoc_info_dict = workchain_preproc_and_viewer_info[workcalc_name][workcalc_version]
    
    # Check if the calculation was successful
    # ---
    # check if number of calls matches
    if len(workcalc.called) < prepoc_info_dict['n_calls']:
        raise(Exception("Not all calculations started."))
    
    # check if the CP2K calculation finished okay
    cp2k_calc = workcalc.called[-1]
    if not cp2k_calc.is_finished_ok:
        raise(Exception("CP2K calculation didn't finish well."))
    
    # ---
    # check if all specified files are retrieved
    #success = True
    #for rlps in prepoc_info_dict['retrieved_files']:
    #    calc_step, retr_list = rlps
    #    calc = list(reversed(workcalc.called))[calc_step]
    #    retrieved_files = calc.outputs.retrieved.list_object_names()
    #    if not all(f in retrieved_files for f in retr_list):
    #        raise(Exception("Not all files were retrieved."))
    # ---
    
    structure = workcalc.inputs[prepoc_info_dict['struct_label']]
    
    # Add the link to the SPM calc to the structure extras in format STMWorkChain_1: <stm_wc_pk> 
    pk_numbers = [e for e in structure.extras if e.startswith(workcalc_name)]
    pk_numbers = [int(e.split('_')[1]) for e in pk_numbers if e.split('_')[1].isdigit()]
    pks = [e[1] for e in structure.extras.items() if e[0].startswith(workcalc_name)]
    if workcalc.pk in pks:
        return
    nr = 1
    if len(pk_numbers) != 0:
        for nr in range(1, 100):
            if nr in pk_numbers:
                continue
            break
    structure.set_extra('%s_%d_pk'% (workcalc_name, nr), workcalc.pk)


def preprocess_spm_calcs(workchain_list = ['STMWorkChain', 'PdosWorkChain', 'AfmWorkChain', 'OrbitalWorkChain']):
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={
        'attributes.process_label': {'in': workchain_list},
        'or':[
               {'extras': {'!has_key': 'preprocess_version'}},
               {'extras.preprocess_version': {'<': PREPROCESS_VERSION}},
           ],
    })
    qb.order_by({WorkChainNode:{'ctime':'asc'}})
    
    for m in qb.all():
        n = m[0]
        ## ---------------------------------------------------------------
        ## calculation not finished
        if not n.is_sealed:
            print("Skipping underway workchain PK %d"%n.pk)
            continue
        calc_states = [out.get_state() for out in n.outputs]
        if 'WITHSCHEDULER' in calc_states:
            print("Skipping underway workchain PK %d"%n.pk)
            continue
        ## ---------------------------------------------------------------
            
        if 'obsolete' not in n.extras:
            n.set_extra('obsolete', False)
        if n.get_extra('obsolete'):
            continue
        
        wc_name = n.attributes['process_label']
        
        try:
            if not all([calc.get_state() == 'FINISHED' for calc in n.outputs]):
                raise(Exception("Not all calculations are 'FINISHED'"))
            
            preprocess_one(n)
            print("Preprocessed PK %d (%s)"%(n.pk, wc_name))
            
            n.set_extra('preprocess_successful', True)
            n.set_extra('preprocess_version', PREPROCESS_VERSION)
            
            if 'preprocess_error' in n.extras:
                n.delete_extra('preprocess_error')
            
        except Exception as e:
            n.set_extra('preprocess_successful', False)
            n.set_extra('preprocess_error', str(e))
            n.set_extra('preprocess_version', PREPROCESS_VERSION)
            print("Failed to preprocess PK %d (%s): %s"%(n.pk, wc_name, e))

def create_viewer_link_html(structure_extras, apps_path):
    calc_links_str = ""
    for key in sorted(structure_extras.keys()):
        key_sp = key.split('_')        
        if len(key_sp) < 2:
            continue    
        wc_name, nr = key.split('_')[:2]
        if wc_name not in workchain_preproc_and_viewer_info:
            continue
            
        link_name = wc_name.replace('WorkChain', '')
        link_name = link_name.replace('Workchain', '')
        spm_pk = int(structure_extras[key])
        
        spm_node = load_node(spm_pk)
        ver = 0
        if 'version' in spm_node.extras:
            ver = spm_node.extras['version']
        
        viewer_path = workchain_preproc_and_viewer_info[wc_name][ver]['viewer_path']
        
        calc_links_str += "<a target='_blank' href='%s?pk=%s'>%s %s</a><br />" % (
            apps_path + viewer_path, spm_pk, link_name, nr)
        
    return calc_links_str


# ## ----------------------------------------------------------------
# ## ----------------------------------------------------------------
# ## ----------------------------------------------------------------
# ## Misc

def get_calc_by_label(workcalc, label):
    qb = QueryBuilder()
    qb.append(WorkChainNode, filters={'uuid':workcalc.uuid})
    qb.append(CalcJob, with_incoming=WorkChainNode, filters={'label':label})
    assert qb.count() == 1
    calc = qb.first()[0]
    assert(calc.is_finished_ok)
    return calc

def get_slab_calc_info(struct_node):
    html = ""
    try:
        cp2k_calc = struct_node.creator
        opt_workchain = cp2k_calc.caller
        thumbnail = opt_workchain.extras['thumbnail']
        description = opt_workchain.description
        struct_description = opt_workchain.extras['structure_description']
        struct_pk = struct_node.pk
        
        html += '<style>#aiida_results td,th {padding: 5px}</style>' 
        html += '<table border=1 id="geom_info" style="margin:0px;">'
        html += '<tr>'
        html += '<th> Structure description: </th>'
        html += '<td> %s </td>' % struct_description
        html += '<td rowspan="2"><img width="100px" src="data:image/png;base64,%s" title="PK:%d"></td>' % (thumbnail, struct_pk)
        html += '</tr>'
        html += '<tr>'
        html += '<th> Calculation description: </th>'
        html += '<td> %s </td>' % description
        html += '</tr>'
        
        html += '</table>'
        
    except:
        html = ""
    return html

def does_remote_file_exist(computer, path):
    ssh_config = computer.get_configuration()
    ssh_cmd = ["ssh"]
    if 'proxy_command' in ssh_config:
        ssh_cmd += ["-o", f"ProxyCommand={ssh_config['proxy_command']}"]
    hostname = ""
    if 'username' in ssh_config:
        hostname += f"{ssh_config['username']}@"
    hostname += f"{computer.hostname}"
    ssh_cmd += [hostname]
    ssh_cmd += [f"if [ -f {path} ]; then echo 1 ; else echo 0 ; fi"]
    f_exists = subprocess.check_output(ssh_cmd).decode()
    if f_exists[0] != '1':
        return False
    return True

def find_struct_wf(structure_node, computer):
    # check spm
    extras = structure_node.extras
    for ex_k in extras.keys():
        if ex_k.startswith(('stm', 'pdos', 'afm', 'orb', 'hrstm')):
            spm_workchain = load_node(extras[ex_k])
            
            # if calc was done using UKS, don't reuse WFN
            if not spm_workchain.inputs.dft_params['uks']:
                
                cp2k_scf_calc = get_calc_by_label(spm_workchain, 'scf_diag')
                if cp2k_scf_calc.computer.hostname == computer.hostname:
                    wfn_path = cp2k_scf_calc.outputs.remote_folder.get_remote_path() + "/aiida-RESTART.wfn"
                    # check if it exists
                    file_exists = does_remote_file_exist(computer, wfn_path)
                    if file_exists:
                        print("Found .wfn from %s"%ex_k)
                        return wfn_path
                    
    # check geo opt
    if structure_node.creator is not None:
        geo_opt_calc = structure_node.creator
        
        # if the geo opt was done using UKS, don't reuse WFN
        if 'UKS' not in dict(geo_opt_calc.inputs['parameters'])['FORCE_EVAL']['DFT']:
        
            geo_comp = geo_opt_calc.computer
            if geo_comp is not None and geo_comp.hostname == computer.hostname:
                wfn_path = geo_opt_calc.outputs.remote_folder.get_remote_path() + "/aiida-RESTART.wfn"
                # check if it exists
                file_exists = does_remote_file_exist(computer, wfn_path)
                if file_exists:
                    print("Found .wfn from geo_opt")
                    return wfn_path
    
    return ""

def comp_plugin_codes(computer_name, plugin_name):
    qb = QueryBuilder()
    qb.append(Computer, project='name', tag='computer')
    qb.append(Code, project='*', with_computer='computer', filters={
        'attributes.input_plugin': plugin_name,
        'or': [{'extras': {'!has_key': 'hidden'}}, {'extras.hidden': False}]
    })
    qb.order_by({Code: {'id': 'desc'}})
    codes = qb.all()
    sel_codes = []
    for code in codes:
        if code[0] == computer_name:
            sel_codes.append(code[1])
    return sel_codes

def get_bbox(ase_atoms):
    cx =np.amax(ase_atoms.positions[:,0]) - np.amin(ase_atoms.positions[:,0])
    cy =np.amax(ase_atoms.positions[:,1]) - np.amin(ase_atoms.positions[:,1])
    cz =np.amax(ase_atoms.positions[:,2]) - np.amin(ase_atoms.positions[:,2])
    return np.array([cx, cy, cz])

def make_geom_file(atoms, filename, spin_guess=None):
        # spin_guess = [[spin_up_indexes], [spin_down_indexes]]
        tmpdir = tempfile.mkdtemp()
        file_path = tmpdir + "/" + filename

        orig_file = StringIO()
        atoms.write(orig_file, format='xyz')
        orig_file.seek(0)
        all_lines = orig_file.readlines()
        comment = all_lines[1] # with newline character!
        orig_lines = all_lines[2:]
        
        modif_lines = []
        for i_line, line in enumerate(orig_lines):
            new_line = line
            lsp = line.split()
            if spin_guess is not None:
                if i_line in spin_guess[0]:
                    new_line = lsp[0]+"1 " + " ".join(lsp[1:])+"\n"
                if i_line in spin_guess[1]:
                    new_line = lsp[0]+"2 " + " ".join(lsp[1:])+"\n"
            modif_lines.append(new_line)
        
        final_str = "%d\n%s" % (len(atoms), comment) + "".join(modif_lines)

        with open(file_path, 'w') as f:
            f.write(final_str)
        aiida_f = SinglefileData(file=file_path)
        shutil.rmtree(tmpdir)
        return aiida_f

def check_if_calc_ok(self_, prev_calc):
    """Checks if a calculation finished well.

    Args:
        self_: The workchain instance, used for reporting.
        prev_calc (CalcNode): a calculation step

    Returns:
        Bool: True if workchain can continue, False otherwise
    """
    if not prev_calc.is_finished_ok:
        if prev_calc.is_excepted:
            self_.report("ERROR: previous step excepted.")
            return False
        if prev_calc.exit_status is not None and prev_calc.exit_status >= 500:
            self_.report("Warning: previous step: " + prev_calc.exit_message)
        else:
            self_.report("ERROR: previous step: " + prev_calc.exit_message)
            return False

    return True

def create_stm_parameterdata(
    extrap_plane,
    const_height_text,
    struct_symbols,
    parent_dir,
    elim_float_slider0,
    elim_float_slider1,
    de_floattext,
    const_current_text,
    fwhms_text,
    ptip_floattext
    ):
        
        max_height = max([float(h) for h in const_height_text])
        extrap_extent = max([max_height - extrap_plane, 5.0])
        
        # Evaluation region in z
        z_min = 'n-2.0_C' if 'C' in struct_symbols else 'p-4.0'
        z_max = 'p{:.1f}'.format(extrap_plane)
        
        
        
        energy_range_str = "%.2f %.2f %.3f" % (
            elim_float_slider0, elim_float_slider1, de_floattext
        )

        paramdata = {
            '--cp2k_input_file':    parent_dir+'aiida.inp',
            '--basis_set_file':     parent_dir+'BASIS_MOLOPT',
            '--xyz_file':           parent_dir+'aiida.coords.xyz',
            '--wfn_file':           parent_dir+'aiida-RESTART.wfn',
            '--hartree_file':       parent_dir+'aiida-HART-v_hartree-1_0.cube',
            '--output_file':        'stm.npz',
            '--eval_region':        ['G', 'G', 'G', 'G', z_min, z_max],
            '--dx':                 '0.15',
            '--eval_cutoff':        '16.0',
            '--extrap_extent':      str(extrap_extent),
            '--energy_range':       energy_range_str.split(),
            '--heights':            const_height_text,
            '--isovalues':          const_current_text,
            '--fwhms':              fwhms_text,
            '--p_tip_ratios':       ptip_floattext,
        }
        return paramdata

def create_orbitals_parameterdata(extrap_plane,
                               heights_text,
                               parent_dir,
                               n_homo_inttext,
                               n_lumo_inttext,
                               isovals_text,
                               fwhms_text,
                               ptip_floattext):
        max_height = max([float(h) for h in heights_text])
        extrap_extent = max([max_height - extrap_plane, 5.0])
        paramdata = {
            '--cp2k_input_file':    parent_dir+'aiida.inp',
            '--basis_set_file':     parent_dir+'BASIS_MOLOPT',
            '--xyz_file':           parent_dir+'aiida.coords.xyz',
            '--wfn_file':           parent_dir+'aiida-RESTART.wfn',
            '--hartree_file':       parent_dir+'aiida-HART-v_hartree-1_0.cube',
            '--orb_output_file':    'orb.npz',
            '--eval_region':        ['G', 'G', 'G', 'G', 'n-1.0_C', 'p%.1f'%extrap_plane],
            '--dx':                 '0.15',
            '--eval_cutoff':        '16.0',
            '--extrap_extent':      str(extrap_extent),
            '--n_homo':             str(n_homo_inttext+2),
            '--n_lumo':             str(n_lumo_inttext+2),
            '--orb_heights':        heights_text,
            '--orb_isovalues':      isovals_text,
            '--orb_fwhms':          fwhms_text,
            '--p_tip_ratios':       ptip_floattext,
        }
        return paramdata    

def create_pp_parameterdata(ase_geom,
                            dx,
                            scanminz_floattxt,
                            scanmaxz_floattxt,
                            amp_floattxt,
                            f0_cantilever_floattxt):
    cell = ase_geom.cell
    top_z = np.max(ase_geom.positions[:, 2])
    paramdata = {
        'probeType':    'O',
        'charge':       -0.028108681223969645,
        'sigma':        0.7,
        'tip':          's',
        'klat':         0.34901278868090491,
        'krad':         21.913190531846034,
        'r0Probe':      [0.0, 0.0, 2.97],
        'PBC':          'False',
        'gridA':        list(cell[0]),
        'gridB':        list(cell[1]),
        'gridC':        list(cell[2]),
        'scanMin':      [0.0, 0.0, np.round(top_z, 1)+scanminz_floattxt],
        'scanMax':      [cell[0,0], cell[1,1], np.round(top_z, 1)+scanmaxz_floattxt],
        'scanStep':     [dx, dx, dx],
        'Amplitude':    amp_floattxt,
        'f0Cantilever': f0_cantilever_floattxt
    }
    return paramdata

def create_2pp_parameterdata(ase_geom,
                             dx,
                             resp,
                             scanminz_floattxt,
                             scanmaxz_floattxt,
                             amp_floattxt,
                             f0_cantilever_floattxt):
    cell = ase_geom.cell
    top_z = np.max(ase_geom.positions[:, 2])
    paramdata = {
        'Catom':        6,
        'Oatom':        8,
        'ChargeCuUp':   resp[0],
        'ChargeCuDown': resp[1],
        'Ccharge':      resp[2],
        'Ocharge':      resp[3],
        'sigma':        0.7,
        'Cklat':        0.24600212465950813,
        'Oklat':        0.15085476515590224,
        'Ckrad':        20,
        'Okrad':        20,
        'rC0':          [0.0, 0.0, 1.82806112489999961213],
        'rO0':          [0.0, 0.0, 1.14881347770000097341],
        'PBC':          'False',
        'gridA':        list(cell[0]),
        'gridB':        list(cell[1]),
        'gridC':        list(cell[2]),
        'scanMin':      [0.0, 0.0, np.round(top_z, 1)+scanminz_floattxt],
        'scanMax':      [cell[0,0], cell[1,1], np.round(top_z, 1)+scanmaxz_floattxt],
        'scanStep':     [dx, dx, dx],
        'Amplitude':    amp_floattxt,
        'f0Cantilever': f0_cantilever_floattxt,
        'tip':          'None',
        'Omultipole':   's',
    }
    return paramdata

def create_hrstm_parameterdata(hrstm_code,
                               parent_dir,
                               ppm_dir,
                               ase_geom,
                               ppm_params_dict,
                               tiptype_ipw,
                               stip_ipw,
                                pytip_ipw,
                                pztip_ipw,
                                pxtip_ipw,
                                volmin_ipw,
                                volmax_ipw,
                                volstep_ipw,
                                volstep_ipwmin,
                                fwhm_ipw,
                                wfnstep_ipw,
                                extrap_ipw,
                                workfun_ipw,
                                orbstip_ipw,
                                fwhmtip_ipw,
                                rotate_ipw
                               ):
# External folders 
        cell = ArrayData()
        cell.set_array('cell', np.diag(ase_geom.cell))
                
        # PPM folder of position
        #ppmQK = ppm_dir+"Q%1.2fK%1.2f/" %(ppm_params_dict['Ocharge'], ppm_params_dict['Oklat'])
        # new convention:
        ppmQK = ppm_dir+"Qo%1.2fQc%1.2fK%1.2f/" % (ppm_params_dict['Ocharge'], ppm_params_dict['Ccharge'],
                                                   ppm_params_dict['Oklat'])
        
        # Tip type to determine PDOS and PPM position files
        if tiptype_ipw != "parametrized":
            pdos_list = tiptype_ipw
            path = os.path.dirname(hrstm_code.get_remote_exec_path())+"/hrstm_tips/"
            pdos_list = [path+"tip_coeffs.tar.gz"]
            tip_pos = [ppmQK+"PPpos", ppmQK+"PPdisp"]
        else: # Parametrized tip
            pdos_list = [str(stip_ipw), str(pytip_ipw), 
                         str(pztip_ipw), str(pxtip_ipw)]
            tip_pos = ppmQK+"PPdisp"

        # HRSTM parameters
        paramdata = {
            '--output':          'hrstm',
            '--voltages':        [str(val) for val in np.round(np.arange(
                volmin_ipw,
                volmax_ipw+volstep_ipw, 
                volstep_ipw), 
                len(str(volstep_ipwmin).split('.')[-1])).tolist()],
            # Sample information
            '--cp2k_input_file': parent_dir+'aiida.inp',
            '--basis_set_file':  parent_dir+'BASIS_MOLOPT',
            '--xyz_file':        parent_dir+'aiida.coords.xyz',
            '--wfn_file':        parent_dir+'aiida-RESTART.wfn',
            '--hartree_file':    parent_dir+'aiida-HART-v_hartree-1_0.cube',
            '--emin':            str(volmin_ipw-2.0*fwhm_ipw),
            '--emax':            str(volmax_ipw+2.0*fwhm_ipw),
            '--fwhm_sam':        str(fwhm_ipw),
            '--dx_wfn':          str(wfnstep_ipw),
            '--extrap_dist':     str(extrap_ipw),
            '--wn':              str(workfun_ipw),
            # Tip information
            '--pdos_list':       pdos_list,
            '--orbs_tip':        str(orbstip_ipw),
            '--tip_shift':       str(ppm_params_dict["rC0"][2]+ppm_params_dict["rO0"][2]),
            '--tip_pos_files':   tip_pos,
            '--fwhm_tip':        str(fwhmtip_ipw),
        }
        if rotate_ipw:
            paramdata['--rotate'] = ''   
        return paramdata 