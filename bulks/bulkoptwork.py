from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.base import Int, Float, Str, Bool
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.code import Code

from aiida.work.workchain import WorkChain, ToContext, Calc, while_
from aiida.work.run import submit

from aiida_cp2k.calculations import Cp2kCalculation

import tempfile
import shutil
import itertools
import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase import Atoms


from apps.surfaces.widgets.get_cp2k_input import get_cp2k_input

class BulkOptWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(BulkOptWorkChain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("max_force", valid_type=Float, default=Float(0.001))
        spec.input("calc_type", valid_type=Str, default=Str('Full DFT'))
        spec.input("workchain", valid_type=Str, default=Str('BulkOptWorkChain'))
        spec.input("vdw_switch", valid_type=Bool, default=Bool(False))
        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600))
        spec.input("fixed_atoms", valid_type=Str, default=Str(''))
        spec.input("num_machines", valid_type=Int, default=Int(1))
        spec.input("system_charge", valid_type=Int, default=Int(0))
        spec.input("uks_switch", valid_type=Str, default=Str('RKS'))
        spec.input("multiplicity", valid_type=Int, default=Int(0))
        spec.input("calc_name", valid_type=Str)

        spec.outline(
            #cls.print_test,
            cls.run_bulkopt,
            while_(cls.not_converged)(
                cls.run_bulkopt_again
            ),
        )
        spec.dynamic_output()

    # ==========================================================================
    def not_converged(self):
        return self.ctx.bulk_opt.res.exceeded_walltime
    # ==========================================================================
    def print_test(self):
        self.report("Reporting test")

    # ==========================================================================
    def run_bulkopt(self):
        self.report("Running CP2K Bulk optimization")

        inputs = self.build_calc_inputs(structure          = self.inputs.structure,
                                        cp2k_code          = self.inputs.cp2k_code,
                                        max_force          = self.inputs.max_force,
                                        calc_type          = self.inputs.calc_type,
                                        workchain          = self.inputs.workchain,
                                        mgrid_cutoff       = self.inputs.mgrid_cutoff,
                                        vdw_switch         = self.inputs.vdw_switch,
                                        fixed_atoms        = self.inputs.fixed_atoms,
                                        num_machines       = self.inputs.num_machines,
                                        system_charge      = self.inputs.system_charge,
                                        uks_switch         = self.inputs.uks_switch,
                                        multiplicity       = self.inputs.multiplicity,
                                        remote_calc_folder = None)

        self.report("inputs: "+str(inputs))
        self.report("parameters: "+str(inputs['parameters'].get_dict()))
        self.report("settings: "+str(inputs['settings'].get_dict()))
        
        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(bulk_opt=Calc(future))

    # ==========================================================================
    def run_bulkopt_again(self):
        # TODO: make this nicer.
        inputs_new = self.build_calc_inputs(structure          = self.inputs.structure,
                                            cp2k_code          = self.inputs.cp2k_code,
                                            max_force          = self.inputs.max_force,
                                            calc_type          = self.inputs.calc_type,
                                            workchain          = self.inputs.workchain,
                                            mgrid_cutoff       = self.inputs.mgrid_cutoff,
                                            vdw_switch         = self.inputs.vdw_switch,
                                            fixed_atoms        = self.inputs.fixed_atoms,
                                            num_machines       = self.inputs.num_machines,
                                            system_charge      = self.inputs.system_charge,
                                            uks_switch         = self.inputs.uks_switch,
                                            multiplicity       = self.inputs.multiplicity,
                                            remote_calc_folder = self.ctx.bulk_opt.out.remote_folder)
        

        self.report("inputs (restart): "+str(inputs_new))
        future_new = submit(Cp2kCalculation.process(), **inputs_new)
        return ToContext(bulk_opt=Calc(future_new))

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls, 
                          structure          = None, 
                          cp2k_code          = None, 
                          max_force          = None, 
                          calc_type          = None,
                          workchain          = None,
                          mgrid_cutoff       = None, 
                          vdw_switch         = None, 
                          fixed_atoms        = None,
                          num_machines       = None,
                          system_charge      = None,
                          uks_switch         = None,
                          multiplicity       = None, 
                          remote_calc_folder = None,
                          **not_used
                         ):

        inputs = {}
        inputs['_label'] = "bulk_opt"
        inputs['code'] = cp2k_code
        inputs['file'] = {}

 
        atoms = structure.get_ase()  # slow
    
        molslab_f = cls.mk_aiida_file(atoms, "bulk.xyz")
        inputs['file']['input_xyz'] = molslab_f


        # parameters
        cell    =[atoms.cell[0, 0],atoms.cell[0, 1], atoms.cell[0, 2],
                  atoms.cell[1, 0],atoms.cell[1, 1], atoms.cell[1, 2],
                  atoms.cell[2, 0],atoms.cell[2, 1], atoms.cell[2, 2]]

#        remote_computer = code.get_remote_computer()
 #       machine_cores = remote_computer.get_default_mpiprocs_per_machine()
     
        walltime = 86000

        inp =     get_cp2k_input(cell               = cell,
                                 atoms              = atoms,
                                 max_force          = max_force,
                                 calc_type          = calc_type, 
                                 mgrid_cutoff       = mgrid_cutoff,
                                 vdw_switch         = vdw_switch,
                                 fixed_atoms        = fixed_atoms,
                                 system_charge      = system_charge,
                                 uks_switch         = uks_switch, 
                                 multiplicity       = multiplicity,
                                 walltime           = walltime*0.97,
                                 workchain          = workchain,
                                 remote_calc_folder = remote_calc_folder
                                )

        if remote_calc_folder is not None:
            inp['EXT_RESTART'] = {
                'RESTART_FILE_NAME': './parent_calc/aiida-1.restart'
            }
            inputs['parent_folder'] = remote_calc_folder

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.pdb']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
        }

        return inputs

    # ==========================================================================
    @classmethod
    def mk_aiida_file(cls, atoms, name):
        tmpdir = tempfile.mkdtemp()
        atoms_file_name = tmpdir + "/" + name
        atoms.write(atoms_file_name)
        atoms_aiida_f = SinglefileData(file=atoms_file_name)
        shutil.rmtree(tmpdir)
        return atoms_aiida_f

    # ==========================================================================

    def _check_prev_calc(self, prev_calc):
        error = None
        if prev_calc.get_state() != 'FINISHED':
            error = "Previous calculation in state: "+prev_calc.get_state()
        elif "aiida.out" not in prev_calc.out.retrieved.get_folder_list():
            error = "Previous calculation did not retrive aiida.out"
        else:
            fn = prev_calc.out.retrieved.get_abs_path("aiida.out")
            content = open(fn).read()
            if "exceeded requested execution time" in content:
                error = "Previous calculation's aiida.out exceeded walltime"

        if error:
            self.report("ERROR: "+error)
            self.abort(msg=error)
            raise Exception(error)

# EOF
