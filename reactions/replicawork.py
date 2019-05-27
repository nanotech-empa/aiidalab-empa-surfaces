from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.base import Int, Str, Float
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.remote import RemoteData
from aiida.orm.code import Code

from aiida.work.workchain import WorkChain, ToContext, Calc, while_
from aiida.work.run import submit

from aiida_cp2k.calculations import Cp2kCalculation

import tempfile
import shutil

import numpy as np

import find_mol

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

class ReplicaWorkchain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(ReplicaWorkchain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("num_machines", valid_type=Int, default=Int(54))
        spec.input("calc_name", valid_type=Str)
        spec.input("cell", valid_type=Str, default=Str(''))
        spec.input("fixed_atoms", valid_type=Str, default=Str(''))
        spec.input("colvar_targets", valid_type=Str)
        spec.input("target_unit", valid_type=Str)
        spec.input("spring", valid_type=Float, default=Float(75.0))
        spec.input("spring_unit", valid_type=Str)
        spec.input("subsys_colvar", valid_type=ParameterData)
        spec.input("calc_type", valid_type=Str)
        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600))

        spec.outline(
            cls.init,
            while_(cls.next_replica)(
                cls.generate_replica,
                while_(cls.not_converged)(
                    cls.generate_replica
                ),
                cls.store_replica
            )
        )
        spec.dynamic_output()

    # ==========================================================================
    def not_converged(self):
        return self.ctx.replica.res.exceeded_walltime
        #try:
        #    self.ctx.remote_calc_folder = self.ctx.replica.out.remote_folder
        #    self.ctx.structure = self.ctx.replica.out.output_structure
        #    self.report('Convergence check: {}'.format(self.ctx.replica.res.exceeded_walltime))
        #    return self.ctx.replica.res.exceeded_walltime
        #except AttributeError:
        #    return True

    # ==========================================================================
    def init(self):
        self.report('Init generate replicas')

        self.ctx.replica_list = str(self.inputs.colvar_targets).split()
        self.ctx.replicas_done = 0
        self.ctx.this_name = self.inputs.calc_name

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
            self.ctx.remote_calc_folder = self.ctx.replica.out.remote_folder
            self.ctx.structure = self.ctx.replica.out.output_structure

        self.ctx.replicas_done += 1

        return True

    # ==========================================================================
    def generate_replica(self):
        self.report("Running CP2K geometry optimization - Target: "
                    .format(self.ctx.this_replica))

        inputs = self.build_calc_inputs(self.ctx.structure,
                                        self.inputs.cell,
                                        self.inputs.cp2k_code,
                                        self.ctx.this_replica,
                                        self.inputs.fixed_atoms,
                                        self.inputs.num_machines,
                                        self.ctx.remote_calc_folder,
                                        self.ctx.this_name,
                                        self.inputs.spring,
                                        self.inputs.spring_unit,
                                        self.inputs.target_unit,
                                        self.inputs.subsys_colvar,
                                        self.inputs.calc_type,
                                        self.inputs.mgrid_cutoff)

        self.report(" ")
        self.report("inputs: "+str(inputs))
        self.report(" ")
        future = submit(Cp2kCalculation.process(), **inputs)
        self.report("future: "+str(future))
        self.report(" ")
        return ToContext(replica=Calc(future))

    # ==========================================================================
    def store_replica(self):
        return self.out('replica_{}_{}'.format(self.ctx.this_replica,
                                               self.ctx.this_name),
                        self.ctx.replica.out.output_structure)

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls, structure, cell, code, colvar_target,
                          fixed_atoms, num_machines, remote_calc_folder,
                          calc_name, spring, spring_unit, target_unit,
                          subsys_colvar, calc_type,mgrid_cutoff):

        inputs = {}
        inputs['_label'] = "replica_geo_opt"
        inputs['_description'] = "replica_{}_{}".format(calc_name,
                                                        colvar_target)

        inputs['code'] = code
        inputs['file'] = {}

        atoms = structure.get_ase()  # slow
        
        walltime = 86000
        
        molslab_f = cls.mk_aiida_file(atoms, "mol_on_slab.xyz")
        inputs['file']['molslab_coords'] = molslab_f
        
        
        first_slab_atom = None        
        if calc_type != 'Full DFT':
            
            # Au potential
            pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
            inputs['file']['au_pot'] = pot_f
            
            mol_indexes = find_mol.extract_mol_indexes_from_slab(atoms)
            
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
            
        remote_computer = code.get_remote_computer()
        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
        
        inp = cls.get_cp2k_input(cell_abc,
                                 colvar_target,
                                 fixed_atoms,
                                 spring, spring_unit,
                                 target_unit,
                                 subsys_colvar,
                                 calc_type,
                                 mgrid_cutoff,
                                 machine_cores*num_machines,
                                 first_slab_atom,
                                 len(atoms),atoms)

        if remote_calc_folder is not None:
            inputs['parent_folder'] = remote_calc_folder

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.xyz']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": 86000,
        }

        return inputs

    # ==========================================================================
    @classmethod
    def get_cp2k_input(cls, cell_abc,
                       colvar_target, fixed_atoms,
                       spring, spring_unit, target_unit, subsys_colvar,
                       calc_type,mgrid_cutoff, machine_cores, first_slab_atom,
                       last_slab_atom,atoms):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'GEO_OPT',
                'WALLTIME': 85500,
                'PRINT_LEVEL': 'LOW',
                'EXTENDED_FFT_LENGTHS': ''
            },
            'MOTION': cls.get_motion(colvar_target, fixed_atoms, spring,
                                     spring_unit, target_unit),
            'FORCE_EVAL': [],
        }
        
        if calc_type == 'Mixed DFTB':
            inp['FORCE_EVAL'] = [cls.force_eval_mixed(cell_abc,
                                                      first_slab_atom,
                                                      last_slab_atom,
                                                      machine_cores,
                                                      subsys_colvar),
                                 cls.force_eval_fist(cell_abc,atoms),
                                 cls.get_force_eval_qs_dftb(cell_abc)]
            inp['MULTIPLE_FORCE_EVALS'] = {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            }

        elif calc_type == 'Mixed DFT':
            inp['FORCE_EVAL'] = [cls.force_eval_mixed(cell_abc,
                                                      first_slab_atom,
                                                      last_slab_atom,
                                                      machine_cores,
                                                      subsys_colvar),
                                 cls.force_eval_fist(cell_abc,atoms),
                                 cls.get_force_eval_qs_dft(cell_abc,mgrid_cutoff,atoms, only_molecule=True)]
            inp['MULTIPLE_FORCE_EVALS'] = {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            }

        elif calc_type == 'Full DFT':
            inp['FORCE_EVAL'] = [cls.get_force_eval_qs_dft(cell_abc,mgrid_cutoff,atoms, only_molecule=False,
                                                           subsys_colvar=subsys_colvar)]
        return inp

    # ==========================================================================
    @classmethod
    def force_eval_mixed(cls, cell_abc, first_slab_atom, last_slab_atom,
                         machine_cores, subsys_colvar):
        first_mol_atom = 1
        last_mol_atom = first_slab_atom - 1

        mol_delim = (first_mol_atom, last_mol_atom)
        slab_delim = (first_slab_atom, last_slab_atom)

        force_eval = {
            'METHOD': 'MIXED',
            'MIXED': {
                'MIXING_TYPE': 'GENMIX',
                'GROUP_PARTITION': '2 %d' % (machine_cores-2),
                'GENERIC': {
                    'ERROR_LIMIT': '1.0E-10',
                    'MIXING_FUNCTION': 'E1+E2',
                    'VARIABLES': 'E1 E2'
                },
                'MAPPING': {
                    'FORCE_EVAL_MIXED': {
                        'FRAGMENT':
                            [{'_': '1', ' ': '%d  %d' % mol_delim},
                             {'_': '2', ' ': '%d  %d' % slab_delim}],
                    },
                    'FORCE_EVAL': [{'_': '1', 'DEFINE_FRAGMENTS': '1 2'},
                                   {'_': '2', 'DEFINE_FRAGMENTS': '1'}],
                }
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                },
                'COLVAR': subsys_colvar.get_attrs()
            }
        }

        return force_eval
    
    # ==========================================================================
    @classmethod
    def force_eval_fist(cls, cell_abc,atoms):
        ff = {
            'SPLINE': {
                'EPS_SPLINE': '1.30E-5',
                'EMAX_SPLINE': '0.8',
            },
            'CHARGE': [],
            'NONBONDED': {
                'GENPOT': [],
                'LENNARD-JONES': [],
                'EAM': {
                    'ATOMS': 'Au Au',
                    'PARM_FILE_NAME': 'Au.pot',
                },
            },
        }

        element_list = list(np.unique(atoms.get_chemical_symbols()))
        
        metal_atom = None
        for el in element_list:
            if el in METAL_ATOMS:
                metal_atom = el
                element_list.remove(el)
                break
                
        if metal_atom is None:
            raise Exception("No valid metal atom found.")
        
        for x in element_list + [metal_atom]:
            ff['CHARGE'].append({'ATOM': x, 'CHARGE': 0.0})

        genpot_fun = 'A*exp(-av*r)+B*exp(-ac*r)-C/(r^6)/( 1+exp(-20*(r/R-1)) )'
        genpot_val = '4.13643 1.33747 115.82004 2.206825'\
                     ' 113.96850410723008483218 5.84114'
        for x in ('C', 'N', 'O', 'H'):
            ff['NONBONDED']['GENPOT'].append(
                {'ATOMS': 'Au ' + x,
                 'FUNCTION': genpot_fun,
                 'VARIABLES': 'r',
                 'PARAMETERS': 'A av B ac C R',
                 'VALUES': genpot_val,
                 'RCUT': '15'}
            )

        for x in ('C H', 'H H', 'H N', 'C C', 'C O', 'C N', 'N N', 'O H',
                  'O N', 'O O'):
            ff['NONBONDED']['LENNARD-JONES'].append(
                {'ATOMS': x,
                 'EPSILON': '0.0',
                 'SIGMA': '3.166',
                 'RCUT': '15'}
            )

        force_eval = {
            'METHOD': 'FIST',
            'MM': {
                'FORCEFIELD': ff,
                'POISSON': {
                    'EWALD': {
                      'EWALD_TYPE': 'none',
                    },
                },
            },
            'SUBSYS': {
                'CELL': {
                    'ABC': cell_abc,
                },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                },
            },
        }
        return force_eval
    
    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dftb(cls, cell_abc):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'QS': {
                    'METHOD': 'DFTB',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'DFTB': {
                        'SELF_CONSISTENT': 'T',
                        'DISPERSION': 'T',
                        'ORTHOGONAL_BASIS': 'F',
                        'DO_EWALD': 'F',
                        'PARAMETER': {
                            'PARAM_FILE_PATH': 'DFTB/scc',
                            'PARAM_FILE_NAME': 'scc_parameter',
                            'UFF_FORCE_FIELD': '../uff_table',
                        },
                    },
                },
                'SCF': {
                    'MAX_SCF': '30',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-6',
                    'OT': {
                        'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                        'MINIMIZER': 'CG',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '20',
                        'EPS_SCF': '1.0E-6',
                    },
                    'PRINT': {
                        'RESTART': {
                            'EACH': {
                                'QS_SCF': '0',
                                'GEO_OPT': '1',
                            },
                            'ADD_LAST': 'NUMERIC',
                            'FILENAME': 'RESTART'
                        },
                        'RESTART_HISTORY': {'_': 'OFF'}
                    }
                }
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol.xyz',
                    'COORDINATE': 'xyz'
                }
            }
        }

        return force_eval
    
    # ==========================================================================
    @classmethod
    def get_motion(cls, colvar_target, fixed_atoms, spring, spring_unit,
                   target_unit):
        motion = {
            'CONSTRAINT': {
                'COLLECTIVE': {
                    'COLVAR': 1,
                    'RESTRAINT': {
                        'K': '[{}] {}'.format(spring_unit, spring)
                    },
                    'TARGET': '[{}] {}'.format(target_unit, colvar_target),
                    'INTERMOLECULAR': ''
                },
                'FIXED_ATOMS': {
                    'LIST': '{}'.format(fixed_atoms)
                }
            },
            'GEO_OPT': {
                'MAX_FORCE': '0.0001',
                'MAX_ITER': '5000',
                'OPTIMIZER': 'BFGS',
                     'BFGS' : {
                         'TRUST_RADIUS' : '[bohr] 0.1'
                     }
            },
        }

        return motion

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell_abc,mgrid_cutoff,atoms, only_molecule,
                              subsys_colvar=None):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
                'RESTART_FILE_NAME': './parent_calc/aiida-RESTART.wfn',
                'QS': {
                    'METHOD': 'GPW',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'EPS_DEFAULT': '1.0E-14',
                },
                'MGRID': {
                    'CUTOFF': '%d' % (mgrid_cutoff),
                    'NGRIDS': '5',
                },
                'SCF': {
                    'MAX_SCF': '20',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-7',
                    'OT': {
                        'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                        'MINIMIZER': 'CG',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '15',
                        'EPS_SCF': '1.0E-7',
                    },
                    'PRINT': {
                        'RESTART': {
                            'EACH': {
                                'QS_SCF': '0',
                                'GEO_OPT': '1',
                            },
                            'ADD_LAST': 'NUMERIC',
                            'FILENAME': 'RESTART'
                        },
                        'RESTART_HISTORY': {'_': 'OFF'}
                    }
                },
                'XC': {
                    'XC_FUNCTIONAL': {'_': 'PBE'},
                    'VDW_POTENTIAL': {
                        'DISPERSION_FUNCTIONAL': 'PAIR_POTENTIAL',
                        'PAIR_POTENTIAL': {
                            'TYPE': 'DFTD3',
                            'CALCULATE_C9_TERM': '.TRUE.',
                            'PARAMETER_FILE_NAME': 'dftd3.dat',
                            'REFERENCE_FUNCTIONAL': 'PBE',
                            'R_CUTOFF': '[angstrom] 15',
                        }
                    }
                },
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'xyz',
                },
                'KIND': [],
            }
        }
        if only_molecule:
            force_eval['SUBSYS']['TOPOLOGY']['COORD_FILE_NAME'] = 'mol.xyz'

        if subsys_colvar is not None:
            force_eval['SUBSYS']['COLVAR'] = subsys_colvar.get_attrs()
        
        
        kinds_used = np.unique(atoms.get_chemical_symbols())
        
        for kind in kinds_used:
            bs, pp = ATOMIC_KINDS[kind] 
            force_eval['SUBSYS']['KIND'].append({
                '_': kind,
                'BASIS_SET': bs,
                'POTENTIAL': pp
            })   


        return force_eval
    
    # ==========================================================================
    @classmethod
    def mk_coord_files(cls, atoms, first_slab_atom):
        mol = atoms[:first_slab_atom-1]

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
#    @classmethod
#    def extract_mol_indexes_from_slab(cls, atoms):
#        exclude=['Au','Ag','Cu','Pd','Ga','Ni']
#        the_mol=Atoms()
#        cov_radii = [covalent_radii[a.number] for a in atoms]
#        nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
#        nl.update(atoms)
#        vec = np.zeros(3)
#        ismol=[]
#        tofollow=[]
#        followed=[]
#        
#        all_carbon_atoms = np.argwhere(atoms.numbers == 6)
#        for c in all_carbon_atoms:
#            tofollow.append(c[0])
#            ismol.append(c[0])
#        while len(tofollow) > 0:
#            indices, offsets = nl.get_neighbors(tofollow[0])
#            indices=list(indices)
#            followed.append(tofollow[0])
#            for inew in indices:
#                tofollow.append(inew)
#            for i in indices:
#                if (i not in ismol) and (i not in exclude):
#                    ismol.append(i)
#            for i in followed:
#                if i in tofollow:
#                    tofollow.remove(i)
#        
#        return ismol
#
    # ==========================================================================
