from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm import Int, Float, Str, Bool
from aiida.orm import SinglefileData
from aiida.orm import Code

from aiida.engine import WorkChain, ToContext, Calc, while_
from aiida.engine import submit

from aiida_cp2k.calculations import Cp2kCalculation

import tempfile
import shutil
import itertools
import numpy as np

from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase import Atoms

from apps.surfaces.widgets import find_mol

from apps.surfaces.widgets.get_cp2k_input import get_cp2k_input

#workchain_dict={'cp2k_code':Code,'structure':StructureData,'max_force':Float}

class SlabGeoOptWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(SlabGeoOptWorkChain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("max_force", valid_type=Float, default=Float(0.001))
        spec.input("calc_type", valid_type=Str, default=Str('Mixed DFTB'))
        spec.input("workchain", valid_type=Str, default=Str('SlabGeoOptWorkChain'))
        spec.input("vdw_switch", valid_type=Bool, default=Bool(False))
        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600))
        spec.input("fixed_atoms", valid_type=Str, default=Str(''))
        spec.input("center_switch", valid_type=Bool, default=Bool(False))
        spec.input("num_machines", valid_type=Int, default=Int(1))
        spec.input("calc_name", valid_type=Str)

        spec.outline(
            cls.run_geopt,
            while_(cls.not_converged)(
                cls.run_geopt_again
            ),
        )
        spec.dynamic_output()

    # ==========================================================================
    def not_converged(self):
        return self.ctx.geo_opt.res.exceeded_walltime

    # ==========================================================================
    def run_geopt(self):
        self.report("Running CP2K geometry optimization")

        inputs = self.build_calc_inputs(structure          = self.inputs.structure,
                                        cp2k_code          = self.inputs.cp2k_code,
                                        max_force          = self.inputs.max_force,
                                        calc_type          = self.inputs.calc_type,
                                        workchain          = self.inputs.workchain,
                                        mgrid_cutoff       = self.inputs.mgrid_cutoff,
                                        vdw_switch         = self.inputs.vdw_switch,
                                        fixed_atoms        = self.inputs.fixed_atoms,
                                        center_switch      = self.inputs.center_switch,
                                        num_machines       = self.inputs.num_machines,
                                        remote_calc_folder = None)

        self.report("inputs: "+str(inputs))
        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(geo_opt=Calc(future))

    # ==========================================================================
    def run_geopt_again(self):
        # TODO: make this nicer.
        inputs_new = self.build_calc_inputs(structure = self.inputs.structure,
                                            cp2k_code = self.inputs.cp2k_code,
                                            max_force = self.inputs.max_force,
                                            calc_type = self.inputs.calc_type,
                                            workchain = self.inputs.workchain,
                                            mgrid_cutoff = self.inputs.mgrid_cutoff,
                                            vdw_switch = self.inputs.vdw_switch,
                                            fixed_atoms = self.inputs.fixed_atoms,
                                            center_switch = self.inputs.center_switch,
                                            num_machines = self.inputs.num_machines,
                                            remote_calc_folder = self.ctx.geo_opt.out.remote_folder)
        

        self.report("inputs (restart): "+str(inputs_new))
        future_new = submit(Cp2kCalculation.process(), **inputs_new)
        return ToContext(geo_opt=Calc(future_new))

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
                          center_switch      = None,
                          num_machines       = None,
                          remote_calc_folder = None):

        inputs = {}
        inputs['_label'] = "slab_geo_opt"
        inputs['code'] = cp2k_code
        inputs['file'] = {}
        

        atoms = structure.get_ase()  # slow

        # parameters
        
        cell=[atoms.cell[0, 0],atoms.cell[0, 1], atoms.cell[0, 2],
              atoms.cell[1, 0],atoms.cell[1, 1], atoms.cell[1, 2],
              atoms.cell[2, 0],atoms.cell[2, 1], atoms.cell[2, 2]]

        remote_computer = cp2k_code.get_remote_computer()
        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
        
        walltime = 86000
        
        molslab_f = cls.mk_aiida_file(atoms, "mol_on_slab.xyz")
        inputs['file']['molslab_coords'] = molslab_f
        first_slab_atom = None
        
        
        if calc_type != 'Full DFT':
            
            # Au potential
            pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
            inputs['file']['au_pot'] = pot_f
            
            mol_indexes = find_mol.extract_mol_indexes_from_slab(atoms)
            
            first_slab_atom = len(mol_indexes) + 1
            
            mol_f = cls.mk_aiida_file(atoms[mol_indexes], "mol.xyz")
            inputs['file']['mol_coords'] = mol_f
            
            if calc_type == 'Mixed DFTB':
                walltime = 18000
   

        inp = get_cp2k_input(cell               = cell,
                             atoms              = atoms,
                             first_slab_atom    = first_slab_atom,
                             last_slab_atom     = len(atoms),
                             max_force          = max_force,
                             calc_type          = calc_type,
                             mgrid_cutoff       = mgrid_cutoff,
                             vdw_switch         = vdw_switch,
                             machine_cores      = machine_cores*num_machines,
                             fixed_atoms        = fixed_atoms,
                             walltime           = walltime*0.97,
                             workchain          = workchain,
                             center_switch      = center_switch,
                             remote_calc_folder = remote_calc_folder
                             )
        if remote_calc_folder is not None:
            inputs['parent_folder'] = remote_calc_folder

        inputs['parameters'] = Dict(dict=inp)

        # settings
        settings = Dict(dict={'additional_retrieve_list': ['*.pdb']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
        }

        return inputs

    # ==========================================================================
    @classmethod
    def mk_coord_files(cls, atoms):
        mol_indexes = find_mol.extract_mol_indexes_from_slab(atoms)
        mol = atoms[mol_indexes]

        tmpdir = tempfile.mkdtemp()
        molslab_fn = tmpdir + '/mol_on_slab.xyz'
        mol_fn = tmpdir + '/mol.xyz'

        atoms.write(molslab_fn)
        mol.write(mol_fn)

        molslab_f = SinglefileData(file=molslab_fn)
        mol_f = SinglefileData(file=mol_fn)

        shutil.rmtree(tmpdir)

        return molslab_f, mol_f
    
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
