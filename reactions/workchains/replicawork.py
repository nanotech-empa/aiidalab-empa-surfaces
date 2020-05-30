from aiida.orm import StructureData, Dict, Int, Str, Float, Bool, ArrayData
from aiida.orm import SinglefileData, RemoteData, Code

from aiida.engine import WorkChain, ToContext, while_
from aiida.engine import submit

from aiida.plugins import WorkflowFactory, CalculationFactory

Cp2kBaseWorkChain = WorkflowFactory('cp2k.base')
Cp2kCalculation = CalculationFactory('cp2k')

#from aiida.orm import Workflow

import tempfile
import shutil

import numpy as np

from apps.surfaces.reactions import analyze_structure
from apps.surfaces.widgets.get_cp2k_input import Get_CP2K_Input

ATOMIC_KINDS = {
    'H' :('TZV2P-MOLOPT-GTH','GTH-PBE-q1'),
    'Au':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q11'),
    'Ag':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q11'),
    'Cu':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q11'),
    'Al':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q3'),
    'B' :('TZV2P-MOLOPT-GTH','GTH-PBE-q1'),
    'Br':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q7'),
    'C' :('TZV2P-MOLOPT-GTH','GTH-PBE-q4'),
    'Ga':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q13'),        
    'N' :('TZV2P-MOLOPT-GTH','GTH-PBE-q5'),
    'O' :('TZV2P-MOLOPT-GTH','GTH-PBE-q6'),
    'Pd':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q18'),
    'S' :('TZV2P-MOLOPT-GTH','GTH-PBE-q6')
}

# possible metal atoms for empirical substrate
METAL_ATOMS = ['Au', 'Ag', 'Cu']

class ReplicaWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(ReplicaWorkChain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("num_machines", valid_type=Int, default=lambda: Int(54))
        spec.input("calc_name", valid_type=Str)
        spec.input("cell", valid_type=Str, default=lambda: Str(''))
        spec.input("fixed_atoms", valid_type=Str, default=lambda: Str(''))
        spec.input("colvar_targets", valid_type=Str)
        spec.input("target_unit", valid_type=Str)
        spec.input("spring", valid_type=Float, default=lambda: Float(75.0))
        spec.input("spring_unit", valid_type=Str)
        spec.input("subsys_colvar", valid_type=Dict)
        spec.input("calc_type", valid_type=Str)
        spec.input("mgrid_cutoff", valid_type=Int, default=lambda: Int(600))
        spec.input("max_force", valid_type=Float, default=lambda: Float(0.0001))
        spec.input("dftd3_switch", valid_type=Bool, default=lambda: Bool(True))

        spec.outline(
            cls.initialize,
            while_(cls.next_replica)(
                cls.generate_replica,
                while_(cls.not_converged)(
                    cls.generate_replica
                ),
                cls.store_replica
            ),
            cls.finalize
        )
        
        spec.outputs.dynamic = True
        
    # ==========================================================================
    def initialize(self):
        self.report('Init generate replicas')
        self.ctx.total_num_replica = len(self.inputs.colvar_targets.value.split())
        self.ctx.replica_list = self.inputs.colvar_targets.value.split()
        self.ctx.replicas_done = 0
        self.ctx.this_name = self.inputs.calc_name.value
        self.ctx.prev_converged = True
        self.ctx.energy_list = []
        
        self.ctx_init_energy_ran = False

        self.report('#{} replicas'.format(len(self.ctx.replica_list)))
        
    # ==========================================================================
    def next_replica(self):
        self.report('Go to replica - {}'.format(len(self.ctx.replica_list)))
        self.report('Remaining list: {} ({})'.format(self.ctx.replica_list,
                                                    len(self.ctx.replica_list)))
        if len(self.ctx.replica_list) > 0:
            self.ctx.this_replica = self.ctx.replica_list.pop(0)
        else:
            return False

        if self.ctx.replicas_done == 0:
            self.ctx.remote_calc_folder = None
            self.ctx.structure = self.inputs.structure
        else:
            self.ctx.remote_calc_folder = self.ctx.replica.outputs.remote_folder
            self.ctx.structure = self.ctx.replica.outputs.output_structure

        return True
        
    # ==========================================================================
    def generate_replica(self):
        
        if not self.ctx_init_energy_ran:
            # together with the initial replica, submit the SCF for initial structure
            self.report('Running SCF to get inital geometry energy')
            inputs = self.build_calc_inputs(self.inputs.structure,
                                            self.inputs.cell.value,
                                            self.inputs.cp2k_code,
                                            self.inputs.num_machines.value,
                                            self.ctx.this_name,
                                            self.inputs.calc_type.value,
                                            self.inputs.mgrid_cutoff.value,
                                            self.inputs.dftd3_switch.value,
                                            initial_energy = True)
            self.report("inputs: "+str(inputs))
            future = self.submit(Cp2kCalculation, **inputs)
            self.report("future: "+str(future))
            self.to_context(initial_scf=future)
            self.ctx_init_energy_ran = True
            
        self.report("Running CP2K geometry optimization - Target: {}"
                    .format(self.ctx.this_replica))
        
        inputs = self.build_calc_inputs(self.ctx.structure,
                                        self.inputs.cell.value,
                                        self.inputs.cp2k_code,
                                        self.inputs.num_machines.value,
                                        self.ctx.this_name,
                                        self.inputs.calc_type.value,
                                        self.inputs.mgrid_cutoff.value,
                                        self.inputs.dftd3_switch.value,
                                        
                                        self.ctx.remote_calc_folder,
                                        self.ctx.prev_converged,
                                        self.inputs.fixed_atoms.value,
                                        self.inputs.max_force.value,
                                        self.inputs.spring.value,
                                        self.inputs.spring_unit.value,
                                        self.inputs.target_unit.value,
                                        dict(self.inputs.subsys_colvar),
                                        self.ctx.this_replica,
                                        initial_energy = False)

        self.report("inputs: "+str(inputs))
        future = self.submit(Cp2kCalculation, **inputs)
        self.report("future: "+str(future))
        return ToContext(replica=future)
        
    # ==========================================================================
    def not_converged(self):
        self.ctx.structure = self.ctx.replica.outputs.output_structure
        if self.ctx.replica.res.exceeded_walltime:
            self.report("Replica didn't converge!")
            # Even if geometry did not converge, update remote_calc_folder and structure
            # to continue from the stopped part
            self.ctx.prev_converged = False
            self.ctx.remote_calc_folder = self.ctx.replica.outputs.remote_folder
        else:
            self.ctx.prev_converged = True
        return self.ctx.replica.res.exceeded_walltime

    
    # ==========================================================================
    def store_replica(self):
        
        n_dig = len(str(self.ctx.total_num_replica))
        label = "replica_{:0{}}".format(self.ctx.replicas_done + 1, n_dig)
        self.ctx.replicas_done += 1
        
        self.report("Storing %s" % label)
        self.out(label, self.ctx.replica.outputs.output_structure)
        
        en = self.ctx.replica.outputs['output_parameters']['energy']
        self.ctx.energy_list.append(en)
        
    # ==========================================================================
    def finalize(self):
        # Store initial geometry as replica_0
        n_dig = len(str(self.ctx.total_num_replica))
        self.out("replica_{:0{}}".format(0, n_dig), self.inputs.structure)
        initial_energy = self.ctx.initial_scf.outputs['output_parameters']['energy']
        
        en_arr = ArrayData()
        en_arr.set_array("energies", np.array([initial_energy] + self.ctx.energy_list))
        en_arr.store()
        self.out("energies", en_arr)
        self.report("Finish!")

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls,
                          structure,
                          cell,
                          code,
                          num_machines,
                          calc_name,
                          calc_type,
                          mgrid_cutoff,
                          dftd3_switch,
                          remote_calc_folder = None,
                          prev_converged = None,
                          fixed_atoms = None,
                          max_force = None,
                          spring = None,
                          spring_unit = None,
                          target_unit = None,
                          subsys_colvar = None,
                          colvar_target = None,
                          initial_energy = False):

        inputs = {}
        inputs['metadata'] = {}
        
        if initial_energy:
            inputs['metadata']['label'] = "initial_energy"
            inputs['metadata']['description'] = "initial_energy"
        else:
            inputs['metadata']['label'] = "replica_geo_opt"
            inputs['metadata']['description'] = "replica_{}_{}".format(calc_name, colvar_target)

        inputs['code'] = code
        inputs['file'] = {}

        atoms = structure.get_ase()  # slow
        
        walltime = 86000
        
        molslab_f = cls.mk_aiida_file(atoms, "bulk.xyz")
        inputs['file']['molslab_coords'] = molslab_f
        
        
        first_slab_atom = None        
        if calc_type != 'Full DFT':
            
            slab_analyzed = analyze_structure.analyze(atoms)
            
            # Au potential
            pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
            inputs['file']['au_pot'] = pot_f
            
            mol_indexes = list(itertools.chain(*slab_analyzed['all_molecules']))
            
            if len(mol_indexes) != np.max(mol_indexes) + 1:
                raise Exception("For mixed calculation, the molecule indexes " +
                                "need to be in the beginning of the file.")
            first_slab_atom = len(mol_indexes) + 2
            
            mol_f = cls.mk_aiida_file(atoms[mol_indexes], "mol.xyz")
            inputs['file']['mol_coords'] = mol_f
            
            if calc_type == 'Mixed DFTB':
                walltime = 18000

        # parameters
        # if no cell is given use the one from the xyz file.
        if cell == '' or len(str(cell)) < 3:
            cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                       atoms.cell[1, 1],
                                       atoms.cell[2, 2])
        else:
            cell_abc = cell
            
        remote_computer = code.computer
        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
        
        elements = list(set(atoms.symbols))
        
        inp_dict = {
            'cell': cell_abc,
            'calc_type': calc_type,
            'mgrid_cutoff': mgrid_cutoff,
            'max_force': max_force,
            'vdw_switch': dftd3_switch,
            'elements': elements,
            'atoms': atoms,
            'mpi_tasks': machine_cores*num_machines,
            'workchain': 'MoleculeKSWorkChain',
            'walltime': walltime,
        }
        
        if not initial_energy:
            additional = {
                'fixed_atoms': fixed_atoms,
                'first_slab_atom': first_slab_atom,
                'last_slab_atom': len(atoms),
                'parent_folder': None if prev_converged else True,
                'colvar_target': float(colvar_target),
                'spring': spring,
                'spring_unit': spring_unit,
                'target_unit': target_unit,
                'subsys_colvar': subsys_colvar,
            }
            inp_dict = {**inp_dict, **additional}
            inp_dict['workchain'] = 'ReplicaWorkChain'
        
        inp = Get_CP2K_Input(inp_dict).inp
        
        inputs['parameters'] = Dict(dict=inp)

        if remote_calc_folder is not None:
            inputs['parent_calc_folder'] = remote_calc_folder

        # settings
        settings = Dict(dict={'additional_retrieve_list': ['*.xyz']})
        inputs['settings'] = settings

        # resources
        inputs['metadata']['options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": 86000,
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
    
