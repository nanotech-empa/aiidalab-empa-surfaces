from aiida.orm import Dict
from aiida.orm import StructureData
from aiida.orm import Int, Str, Float, Bool, List
from aiida.orm import SinglefileData
from aiida.orm.nodes.data.folder import FolderData
from aiida.orm import Code
from aiida.common import NotExistent
# from aiida.orm.data.structure import StructureData

from aiida.engine import WorkChain, ToContext, Calc, while_
from aiida.engine import submit

import aiida_cp2k ##for debug print
from aiida_cp2k.calculations import Cp2kCalculation

import numpy as np

import find_mol

from apps.surfaces.widgets.get_cp2k_input import get_cp2k_input


class NEBWorkchain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(NEBWorkchain, cls).define(spec)        
        spec.input("cp2k_code"        , valid_type=Code)
        spec.input("structure"        , valid_type=StructureData)
        spec.input("max_force"        , valid_type=Float, default=Float(0.0005))
        spec.input("calc_type"        , valid_type=Str, default=Str('Full DFT'))
        spec.input("workchain"        , valid_type=Str, default=Str('NEBWorkchain'))
        spec.input("vdw_switch"       , valid_type=Bool, default=Bool(False))
        spec.input("mgrid_cutoff"     , valid_type=Int, default=Int(600))
        spec.input("fixed_atoms"      , valid_type=Str, default=Str(''))        
        spec.input("num_machines"     , valid_type=Int, default=Int(1))                
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
            cls.init,
            cls.calc_neb,
            #while_(cls.not_converged)(
            #    cls.calc_neb
            #),
            # cls.store_neb
        )
        spec.dynamic_output()

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
    def init(self):
        self.report('Init NEB')
        # Set the restart folder
        try:
            self.ctx.remote_calc_folder = self.ctx.neb.remote_calc_folder
        except AttributeError:
            self.ctx.remote_calc_folder = None

        # Here we need to create the xyz files of all the replicas
        #self.ctx.this_name = self.inputs.calc_name
        self.ctx.file_list = self.inputs.struc_folder.get_folder_list()
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
                                        max_force          = self.inputs.max_force,      
                                        calc_type          = self.inputs.calc_type,      
                                        workchain          = self.inputs.workchain,      
                                        vdw_switch         = self.inputs.vdw_switch,     
                                        mgrid_cutoff       = self.inputs.mgrid_cutoff,   
                                        fixed_atoms        = self.inputs.fixed_atoms,    
                                        num_machines       = self.inputs.num_machines,   
                                        struc_folder       = self.inputs.struc_folder,   
                                        wfn_cp_commands    = self.inputs.wfn_cp_commands,
                                        nproc_rep          = self.inputs.nproc_rep,      
                                        nreplicas          = self.inputs.nreplicas, 
                                        replica_pks        = self.inputs.replica_pks,
                                        spring             = self.inputs.spring,         
                                        rotate             = self.inputs.rotate,         
                                        align              = self.inputs.align,          
                                        nstepsit           = self.inputs.nstepsit,       
                                        endpoints          = self.inputs.endpoints, 
                                        file_list          = self.ctx.file_list,
                                        remote_calc_folder = self.ctx.remote_calc_folder
                                       )
        self.report(" ")
        self.report("inputs: "+str(inputs))
        self.report(" ")
        self.report("Using aiida-cp2k: "+str(aiida_cp2k.__file__))
        self.report(" ")
        future = submit(Cp2kCalculation.process(), **inputs)
        self.report("future: "+str(future))
        self.report(" ")
        return ToContext(neb=Calc(future))
    # ==========================================================================
    #def store_replica(self):structure =
    #    return self.out('replica_{}'.format(self.ctx.this_name),
    #                    self.ctx.neb.out.output_structure)
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
                          file_list          = None,
                          remote_calc_folder = None,
                          **not_used
                         ): 

        inputs = {}
        inputs['_label'] = "NEB"

        inputs['code'] = cp2k_code
        inputs['file'] = {}
        atoms = structure.get_ase()  # slow
        
        cell=[atoms.cell[0, 0],atoms.cell[0, 1], atoms.cell[0, 2],
              atoms.cell[1, 0],atoms.cell[1, 1], atoms.cell[1, 2],
              atoms.cell[2, 0],atoms.cell[2, 1], atoms.cell[2, 2]]
        
        # The files passed by the notebook

        for f in struc_folder.get_folder_list():
            path = struc_folder.get_abs_path()+'/path/'+f
            inputs['file'][f] = SinglefileData(file=path)


        remote_computer = cp2k_code.get_remote_computer()
        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
        first_slab_atom = None
        
        if calc_type != 'Full DFT':
            
            # Au potential
            pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
            inputs['file']['au_pot'] = pot_f
            mol_indexes = find_mol.extract_mol_indexes_from_slab(atoms)
            
            first_slab_atom = len(mol_indexes) + 1
            
        if calc_type == 'Mixed DFTB':
            walltime = 18000
        else:
            walltime = 86000

        nreplica_files=replica_pks.value
        nreplica_files=len(nreplica_files.split())
        inp = get_cp2k_input(atoms=atoms,
                             cell=cell,
                             fixed_atoms=fixed_atoms,
                             max_force=max_force,
                             machine_cores=machine_cores*num_machines,
                             align=align,
                             endpoints=endpoints,
                             nproc_rep=nproc_rep,
                             nreplicas=nreplicas,
                             nstepsit=nstepsit,
                             rotate=rotate,
                             spring=spring,
                             calc_type=calc_type,
                             mgrid_cutoff=mgrid_cutoff,
                             nreplica_files=nreplica_files,
                             first_slab_atom=first_slab_atom,
                             last_slab_atom=len(atoms),
                             walltime=walltime*0.97,
                             workchain=workchain,
                             remote_calc_folder=remote_calc_folder
                            )

        if remote_calc_folder is not None:
            inputs['parent_folder'] = remote_calc_folder

        inputs['parameters'] = Dict(dict=inp)

        # settings
        settings = Dict(dict={'additional_retrieve_list': ['*.xyz',
                                                                    '*.out',
                                                                    '*.ener']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
        }
        if len(wfn_cp_commands) > 0:
            inputs['_options']["prepend_text"] = ""
            for wfn_cp_command in wfn_cp_commands:
                inputs['_options']["prepend_text"] += wfn_cp_command + "\n"
        return inputs

    # ==========================================================================

