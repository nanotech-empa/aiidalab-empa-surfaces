from aiida.orm.code import Code
from aiida.orm import load_node
from aiida.orm import Code, Computer
from aiida.orm.data.base import Int, Float, Bool, Str, List
from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.folder import FolderData
import numpy as np

import tempfile
import shutil
import os
import subprocess

import ase
import ase.io
from ase import Atom

def mk_coord_files(replica_pks=None, all_mols=None,calc_type=None):
    fd = FolderData()
    
    structures=[load_node(x) for x in replica_pks]
    tmpdir = tempfile.mkdtemp()
    if calc_type != 'Full DFT':
        # We need mol0.xyz for the initial mixed force_eval
        mol_fn = tmpdir + '/mol.xyz'
        atoms = structures[0].get_ase()
        mol_ids=[ia  for im in all_mols for ia in im]
        mol = atoms[mol_ids]
        mol.write(mol_fn)

    # And we also write all the replicas up to the final geometry.
    for i, s in enumerate(structures):
        #float_progress.value = 1.0*i/len(structures)
        atoms = s.get_ase()
        molslab_fn = tmpdir + '/replica{}.xyz'.format(i+1)
        atoms.write(molslab_fn)

    fd.replace_with_folder(folder=tmpdir)
    shutil.rmtree(tmpdir)
    return fd

def structure_available_wfn(struct_pk, current_hostname):
    """
    Checks availability of .wfn file corresponding to a structure and returns the remote path.
    Geo_opt, replica opt and NEB calculation output structures supported.
    """
    
    struct_node = load_node(struct_pk)
    
    parent_calc = None
    wfn_name = None
    
    for key, val in struct_node.get_inputs_dict().items():
        
        if "opt_replica_" in key:
            # parent is NEB
            imag_nr = int(key.split("_")[-1]) + 1
            parent_calc = val
            total_n_reps = parent_calc.get_inputs_dict()['CALL'].get_inputs_dict()['nreplicas']
            n_digits = len(str(total_n_reps))
            fmt = "%."+str(n_digits)+"d"
            wfn_name = "aiida-BAND"+str(fmt % imag_nr)+"-RESTART.wfn"
            break
        elif 'output_structure' in key:
            # parent is either REPLICA or GEO_OPT
            parent_calc = val
            wfn_name = "aiida-RESTART.wfn"
            break
    
    if parent_calc is None:
        print("Struct %d .wfn not avail: didn't find parent calc." % struct_pk)
        return None
    
    hostname = parent_calc.get_computer().hostname
    # Remote machine has to be the same as target machine, otherwise no copying possible...
    if hostname != current_hostname:
        print("Struct %d .wfn not avail: different hostname." % struct_pk)
        return None
    
    wfn_search_path = parent_calc._get_remote_workdir() + "/" + wfn_name
    ssh_cmd="ssh "+hostname+" if [ -f "+wfn_search_path+" ]; then echo 1 ; else echo 0 ; fi"
    #wfn_exists = ! ssh {hostname} "if [ -f {wfn_search_path} ]; then echo 1 ; else echo 0 ; fi"
    wfn_exists = subprocess.check_output(ssh_cmd.split())       
    if wfn_exists[0] == '1':
        return wfn_search_path
    
    print("Struct %d .wfn not avail: file deleted from remote." % struct_pk)
    return None

def mk_wfn_cp_commands(nreplicas=None,replica_pks=None,selected_computer=None):
    #print(replica_pks)
    available_wfn_paths = []
    list_wfn_available = []
    list_of_cp_commands = []
    for ir, node_pk in enumerate(replica_pks):
        
        the_selected_computer = selected_computer.get_computer()
        avail_wfn = structure_available_wfn(node_pk, the_selected_computer.hostname)
        
        if avail_wfn:
            list_wfn_available.append(ir) ## example:[0,4,8]
            available_wfn_paths.append(avail_wfn)
    
    if len(list_wfn_available) == 0:
        return []
    
    n_images_available = len(replica_pks)
    n_images_needed = nreplicas
    n_digits = len(str(n_images_needed))
    fmt = "%."+str(n_digits)+"d"
    
    # assign each initial replica to a block of created reps
    block_size = n_images_needed/float(n_images_available)
    
    for to_be_created in range(1,n_images_needed+1):
        name = "aiida-BAND"+str(fmt % to_be_created)+"-RESTART.wfn"
        
        lwa = np.array(list_wfn_available)
        
        #index_wfn = np.abs(np.round(lwa*block_size + block_size/2) - to_be_created).argmin()
        index_wfn = np.abs(lwa*block_size + block_size/2 - to_be_created).argmin()
        
        closest_available = lwa[index_wfn]
        
        print(name, closest_available)
        
        list_of_cp_commands.append("cp %s ./%s" % (available_wfn_paths[index_wfn], name))
        
    return list_of_cp_commands