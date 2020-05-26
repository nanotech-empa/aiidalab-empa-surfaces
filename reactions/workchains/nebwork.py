from aiida.orm import Dict
from aiida.orm import StructureData
from aiida.orm import Int, Str, Float, Bool, List
from aiida.orm import SinglefileData
from aiida.orm.nodes.data.folder import FolderData
from aiida.orm import Code
from aiida.common import NotExistent
# from aiida.orm.data.structure import StructureData

from aiida.engine import WorkChain, ToContext, while_
from aiida.engine import submit

from aiida.plugins import WorkflowFactory, CalculationFactory

import aiida_cp2k

Cp2kCalculation = CalculationFactory('cp2k')

import numpy as np

import os

#import find_mol

from apps.surfaces.widgets import analyze_structure

from apps.surfaces.widgets.get_cp2k_input import Get_CP2K_Input


class NEBWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(NEBWorkChain, cls).define(spec)        
        spec.input("cp2k_code"        , valid_type=Code)
        spec.input("structure"        , valid_type=StructureData)
        spec.input("max_force"        , valid_type=Float, default=lambda: Float(0.0005))
        spec.input("calc_type"        , valid_type=Str, default=lambda: Str('Full DFT'))
        spec.input("workchain"        , valid_type=Str, default=lambda: Str('NEBWorkChain'))
        spec.input("vdw_switch"       , valid_type=Bool, default=lambda: Bool(False))
        spec.input("mgrid_cutoff"     , valid_type=Int, default=lambda: Int(600))
        spec.input("fixed_atoms"      , valid_type=Str, default=lambda: Str(''))        
        spec.input("num_machines"     , valid_type=Int, default=lambda: Int(1))                
        spec.input("struc_folder"     , valid_type=FolderData)
        spec.input("wfn_cp_commands"  , valid_type=List)        
        spec.input("calc_name"        , valid_type=Str)
        spec.input("nproc_rep"        , valid_type=Int)
        spec.input("nreplicas"        , valid_type=Int)
        spec.input("replica_pks"      , valid_type=Str)
        spec.input("spring"           , valid_type=Float)
        spec.input("rotate"           , valid_type=Bool)
        spec.input("align"            , valid_type=Bool)
        spec.input("nstepsit"         , valid_type=Int)
        spec.input("endpoints"        , valid_type=Bool)
        
        
        spec.outline(
            cls.initialize,
            cls.calc_neb,
            #while_(cls.not_converged)(
            #    cls.calc_neb
            #),
            cls.store_outputs
        )
        spec.outputs.dynamic = True

    # ==========================================================================
    def not_converged(self):
        try:
            self.report('Convergence check DEBUG: {}'.format(self.ctx.neb))
            self.report('Convergence check: {}'
                        .format(self.ctx.neb.res.exceeded_walltime))
            return self.ctx.neb.res.exceeded_walltime
        except AttributeError:
            return True
        except NotExistent:
            return False

    # ==========================================================================
    def initialize(self):
        self.report('Init NEB')
        # Set the restart folder
        #try:
        #    self.ctx.remote_calc_folder = self.ctx.neb.remote_calc_folder
        #except AttributeError:
        #    self.ctx.remote_calc_folder = None

        # Here we need to create the xyz files of all the replicas
        #self.ctx.this_name = self.inputs.calc_name
        self.ctx.file_list = self.inputs.struc_folder.list_object_names()
        self.ctx.n_files = len(self.ctx.file_list) #-2

        # Report some things
        self.report('Passed #{} replica geometries + files'.format(self.ctx.n_files))
        self.report('Replicas: {}'.format(self.ctx.file_list))

    # ==========================================================================
    def calc_neb(self):
        self.report("Running CP2K CI-NEB calculation."
                    .format(self.inputs.calc_name))
        
        inputs = self.build_calc_inputs(cp2k_code          = self.inputs.cp2k_code,      
                                        structure          = self.inputs.structure ,     
                                        max_force          = self.inputs.max_force.value,      
                                        calc_type          = self.inputs.calc_type.value,      
                                        workchain          = self.inputs.workchain.value,      
                                        vdw_switch         = self.inputs.vdw_switch.value,     
                                        mgrid_cutoff       = self.inputs.mgrid_cutoff.value,   
                                        fixed_atoms        = self.inputs.fixed_atoms.value,    
                                        num_machines       = self.inputs.num_machines.value,   
                                        struc_folder       = self.inputs.struc_folder,   
                                        wfn_cp_commands    = self.inputs.wfn_cp_commands,
                                        nproc_rep          = self.inputs.nproc_rep.value,      
                                        nreplicas          = self.inputs.nreplicas.value, 
                                        replica_pks        = self.inputs.replica_pks.value,
                                        spring             = self.inputs.spring.value,         
                                        rotate             = self.inputs.rotate.value,         
                                        align              = self.inputs.align.value,          
                                        nstepsit           = self.inputs.nstepsit.value,       
                                        endpoints          = self.inputs.endpoints.value, 
                                        #file_list          = self.ctx.file_list,
                                        #remote_calc_folder = self.ctx.remote_calc_folder
                                       )
        
        # Use the neb parser
        inputs['metadata']['options']['parser_name'] = 'cp2k_neb_parser'
        
        self.report(" ")
        self.report("inputs: "+str(inputs))
        self.report(" ")
        future = self.submit(Cp2kCalculation, **inputs)
        self.report("future: "+str(future))
        self.report(" ")
        return ToContext(neb=future)
    
    # ==========================================================================
    def store_outputs(self):
        self.report("Storing the output")
        
        for i_rep in range(self.inputs.nreplicas.value):
            label = "opt_replica_%d" % i_rep
            self.out(label, self.ctx.neb.outputs[label])
        
        self.out("replica_energies", self.ctx.neb.outputs["replica_energies"])
        self.out("replica_distances", self.ctx.neb.outputs["replica_distances"])
        
        self.report("Finish!")
        
    
    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls, 
                          cp2k_code          = None,
                          structure          = None,
                          max_force          = None,
                          calc_type          = None,
                          workchain          = None,
                          vdw_switch         = None,
                          mgrid_cutoff       = None,
                          fixed_atoms        = None,
                          num_machines       = None,
                          struc_folder       = None,
                          wfn_cp_commands    = None,
                          nproc_rep          = None,
                          nreplicas          = None,
                          replica_pks        = None,
                          spring             = None,
                          rotate             = None,
                          align              = None,
                          nstepsit           = None,
                          endpoints          = None,
                          #file_list          = None,
                          **not_used
                         ): 

        inputs = {}
        inputs['metadata'] = {}
        inputs['metadata']['label'] = "neb"

        inputs['code'] = cp2k_code
        inputs['file'] = {}
        atoms = structure.get_ase()  # slow
        
        
        # The files passed by the notebook

        for f in struc_folder.list_object_names():            
            with struc_folder.open(f) as handle:
                f_no_dot = f.replace(".", "_")
                inputs['file'][f_no_dot] = SinglefileData(file=handle.name)


        remote_computer = cp2k_code.get_remote_computer()
        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
        
        slab_analyzed = analyze_structure.analyze(atoms)
        
        first_slab_atom = None
        if calc_type != 'Full DFT':
            
            # Au potential
            pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
            inputs['file']['au_pot'] = pot_f
            
            mol_indexes = list(itertools.chain(*slab_analyzed['all_molecules']))
            
            if len(mol_indexes) != np.max(mol_indexes) + 1:
                raise Exception("For mixed calculation, the molecule indexes " +
                                "need to be in the beginning of the file.")
            first_slab_atom = len(mol_indexes) + 2
            
            #mol_f = cls.mk_aiida_file(atoms[mol_indexes], "mol.xyz")
            #inputs['file']['mol_coords'] = mol_f
            
        if calc_type == 'Mixed DFTB':
            walltime = 18000
        else:
            walltime = 86000

        nreplica_files=len(replica_pks.split())
        
        cell_str = " ".join(["%.4f" % a for a in np.array(atoms.cell).flatten()])
        
        cp2k_dict = {
            'atoms': atoms,
            'cell': cell_str,
            'fixed_atoms': fixed_atoms,
            'max_force': max_force,
            'vdw_switch': vdw_switch,
            'mpi_tasks': machine_cores*num_machines,
            'align': align,
            'endpoints': endpoints,
            'nproc_rep': nproc_rep,
            'nreplicas': nreplicas,
            'nreplica_files': nreplica_files,
            'nstepsit': nstepsit,
            'rotate': rotate,
            'spring': spring,
            'calc_type': calc_type,
            'mgrid_cutoff': mgrid_cutoff,
            'first_slab_atom': first_slab_atom,
            'last_slab_atom': len(atoms),
            'walltime': walltime*0.97,
            'workchain': workchain,
            'elements': slab_analyzed['all_elements'],
        }
        
        inp = Get_CP2K_Input(cp2k_dict).inp

        inputs['parameters'] = Dict(dict=inp)

        # settings
        settings = Dict(dict={'additional_retrieve_list': ['*.xyz', '*.out', '*.ener']})
        inputs['settings'] = settings

        # resources
        inputs['metadata']['options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
        }
        if len(wfn_cp_commands) > 0:
            inputs['metadata']['options']["prepend_text"] = ""
            for wfn_cp_command in wfn_cp_commands:
                inputs['metadata']['options']["prepend_text"] += wfn_cp_command + "\n"
        return inputs

    # ==========================================================================

