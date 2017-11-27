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
import numpy as np


class DFTGeoOptWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(DFTGeoOptWorkChain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("max_force", valid_type=Float, default=Float(0.001))
        spec.input("vdw_switch", valid_type=Bool, default=Bool(False))
        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600))
        spec.input("fixed_atoms", valid_type=Str, default=Str(''))

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

        inputs = self.build_calc_inputs(self.inputs.structure,
                                        self.inputs.cp2k_code,
                                        self.inputs.max_force,
                                        self.inputs.mgrid_cutoff,
                                        self.inputs.vdw_switch,
                                        self.inputs.fixed_atoms,
                                        None)

        self.report("inputs: "+str(inputs))
        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(geo_opt=Calc(future))

    # ==========================================================================
    def run_geopt_again(self):
        # TODO: make this nicer.
        inputs_new = self.build_calc_inputs(self.inputs.structure,
                                            self.inputs.cp2k_code,
                                            self.inputs.max_force,
                                            self.inputs.mgrid_cutoff,
                                            self.inputs.vdw_switch,
                                            self.inputs.fixed_atoms,
                                            self.ctx.geo_opt.out.remote_folder)

        self.report("inputs (restart): "+str(inputs_new))
        future_new = submit(Cp2kCalculation.process(), **inputs_new)
        return ToContext(geo_opt=Calc(future_new))

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls, structure, code, max_force,
                          mgrid_cutoff, vdw_switch, fixed_atoms,
                          remote_calc_folder=None):

        inputs = {}
        inputs['_label'] = "slab_geo_opt"
        inputs['code'] = code
        inputs['file'] = {}

        # make sure we're really dealing with a gold slab
        atoms = structure.get_ase()  # slow

        # structure
        tmpdir = tempfile.mkdtemp()
        mol_fn = tmpdir + '/mol.xyz'
        atoms.write(mol_fn)
        mol_f = SinglefileData(file=mol_fn)
        shutil.rmtree(tmpdir)

        inputs['file']['mol_coords'] = mol_f

        # parameters
        cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                   atoms.cell[1, 1],
                                   atoms.cell[2, 2])

        num_machines = int(np.round(1. + len(atoms)/120.))
        walltime = 86000

        inp = cls.get_cp2k_input(cell_abc,
                                 max_force,
                                 mgrid_cutoff,
                                 vdw_switch,
                                 fixed_atoms,
                                 walltime*0.97)

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
    def get_cp2k_input(cls, cell_abc,
                       max_force, mgrid_cutoff, vdw_switch,
                       fixed_atoms, walltime):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'GEO_OPT',
                'WALLTIME': '%d' % walltime,
                'PRINT_LEVEL': 'LOW'
            },
            'MOTION': cls.get_motion(max_force, fixed_atoms),
            'FORCE_EVAL': cls.get_force_eval_qs_dft(cell_abc, mgrid_cutoff,
                                                    vdw_switch),
        }

        return inp

    # ==========================================================================
    @classmethod
    def get_motion(cls, max_force, fixed_atoms):
        motion = {
            'CONSTRAINT': {
                'FIXED_ATOMS': {
                    'LIST': '%s' % (fixed_atoms),
                }
            },
            'GEO_OPT': {
                'MAX_FORCE': '%f' % (max_force),
                'MAX_ITER': '5000'
            },
        }

        return motion

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell_abc, mgrid_cutoff, vdw_switch):
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
                },
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol.xyz',
                    'COORDINATE': 'xyz',
                    'CENTER_COORDINATES': {'_': ''},
                },
                'KIND': [],
            }
        }

        if vdw_switch is True:
            force_eval['DFT']['XC']['VDW_POTENTIAL'] = {
                'DISPERSION_FUNCTIONAL': 'PAIR_POTENTIAL',
                'PAIR_POTENTIAL': {
                    'TYPE': 'DFTD3',
                    'CALCULATE_C9_TERM': '.TRUE.',
                    'PARAMETER_FILE_NAME': 'dftd3.dat',
                    'REFERENCE_FUNCTIONAL': 'PBE',
                    'R_CUTOFF': '15',
                }
            }

        force_eval['SUBSYS']['KIND'].append({
            '_': 'Au',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q11'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'C',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q4'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'Br',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q7'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'H',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q1'
        })

        return force_eval

# EOF
