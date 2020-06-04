from aiidalab_widgets_base.utils import string_range_to_list, list_to_string_range
import numpy as np
import itertools
import copy

#from aiida.orm import Int, Float, Str, Bool, List
#from aiida.orm import  Dict
import copy

ATOMIC_KINDS = {
    'H' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q1'   },
    'Au':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q11'  },
    'Ag':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q11'  },
    'Cu':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q11'  },
    'Al':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q3'   },
    'B' :{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q3'   },
    'Br':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q7'   },
    'C' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q4'   },
    'Si':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-GTH'    , 'pseudo' : 'GTH-PBE-q4'   },
    'Ga':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q13'  },        
    'N' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q5'   },
    'O' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q6'   },
    'Pd':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q18'  },
    'S' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q6'   },
    'Zn':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q12'  },
}

for element in ATOMIC_KINDS.keys():
    ATOMIC_KINDS[element]['RI_AUX'] = None
    ATOMIC_KINDS[element]['RI_HFX_BASIS_all'] = None
    
ATOMIC_KINDS['H']['RI_AUX'] = 'aug-cc-pVDZ-RIFIT'
ATOMIC_KINDS['C']['RI_AUX'] = 'aug-cc-pVDZ-RIFIT'
ATOMIC_KINDS['H']['GW_BASIS_SET'] = 'aug-cc-pVDZ-GTH'
ATOMIC_KINDS['C']['GW_BASIS_SET'] = 'aug-cc-pVDZ-GTH'

# possible metal atoms for empirical substrate
METAL_ATOMS = ['Au', 'Ag', 'Cu']

DEFAULT_INPUT_DICT ={
# GENERAL
        'added_mos'             : False                 ,
        'atoms'                 : None                  ,
        'calc_type'             : 'Full DFT'            ,
        'cell'                  : None                  ,
        'cell_free'             : None                  ,
        'cell_sym'              : 'ORTHORHOMBIC'        , 
        'center_coordinates'    : False                 ,
        'charge'                : 0                     , 
        'corr_occ'              : 10                    , 
        'corr_virt'             : 10                    , 
        'first_slab_atom'       : None                  , 
        'fixed_atoms'           : ''                    ,
        'functional'            : 'PBE'                 , 
        'gw_type'               : None                  , 
        'last_slab_atom'        : None                  , 
        'max_force'             : 0.0001                ,
        'max_memory'            : 0                     ,
        'mgrid_cutoff'          : 600                   , 
        'mpi_tasks'             : None                  , 
        'multiplicity'          : 0                     , 
        'diag_method'             : 'OT'                  ,
        'parent_folder'         : None                  , # why is ext_restart named this?
        'periodic'              : None                  ,
        'poisson_solver'        : None                  ,
        #'remote_calc_folder'    : None                  , 
        'smear'                 : False                 ,
        'spin_d'                : ''                    ,
        'spin_u'                : ''                    , 
        #'struc_folder'          : None                  , 
        'vdw_switch'            : None                  , 
        'walltime'              : 86000                 , 
        'workchain'             : 'SlabGeoOptWorkChain' , 
# NEB & Replica chain
        'align'                 : False                 ,
        'colvar_target'         : None                  ,   
        'endpoints'             : True                  ,
        'nproc_rep'             : None                  , 
        'nreplicas'             : None                  , 
        'nreplica_files'        : None                  , 
        'nstepsit'              : 5                     ,
        'rotate'                : False                 , 
        'spring'                : 0.05                  , 
        'spring_unit'           : None                  , 
        'subsys_colvar'         : None                  , 
        'target_unit'           : None                  , 
}
    

########py_type_conversion={type(Str(''))   : str   ,
########                   type(Bool(True)) : bool  ,
########                   type(Float(1.0)) : float ,
########                   type(Int(1))     : int
########                  }
########
########def to_py_type(aiida_obj):
########    if type(aiida_obj) in py_type_conversion.keys():
########        return py_type_conversion[type(aiida_obj)](aiida_obj)
########    elif type(aiida_obj) == type(List()):                
########        pylist=list(aiida_obj)
########        return pylist
########    elif type(aiida_obj) == type(Dict()):                
########        pydict=aiida_obj.get_dict()
########        return pydict
########    else:
########        return aiida_obj

class Get_CP2K_Input():
    def __init__(self,input_dict = None):
    
        self.inp_dict = copy.deepcopy(DEFAULT_INPUT_DICT)
        for inp_key in input_dict:
            #self.inp_dict[inp_key] = to_py_type(input_dict[inp_key]) 
            self.inp_dict[inp_key] = input_dict[inp_key]

        self.qs_default={
                    'METHOD': 'GPW',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'EPS_DEFAULT': '1.0E-14',
                    } 
        self.qs_gw={
                    'METHOD'             : 'GPW',
                    'EPS_PGF_ORB'        : '1.0E-80',
                    'EPS_FILTER_MATRIX'  : '1.0E-80'
                    }
        self.qs_neb={
                                  'METHOD': 'GPW',
                                  'EXTRAPOLATION': 'USE_GUESS',
                                  'EPS_DEFAULT': '1.0E-14',
                                }

        self.xc_default={
                    'XC_FUNCTIONAL': {'_': 'PBE'},
                }

        ### XC FOR GW
        self.xc_gw = {}
        
        #maybe we will reintroduce EPS_DEFAULT
        #if self.inp_dict['gw_type'] is not None:
        #            self.qs_gw['EPS_DEFAULT']=self.inp_dict['eps_default']
        if self.inp_dict['gw_type']=='GW':                   
            self.xc_gw={
                    'XC_FUNCTIONAL': {'_': 'PBE'}, 
                    'WF_CORRELATION': {
                        'RI_RPA': {
                            'RPA_NUM_QUAD_POINTS': '100', 
                            'GW': {
                                'CORR_MOS_OCC': '20', 
                                'CORR_MOS_VIRT': '20', 
                                'EV_GW_ITER': '10', 
                                'NPARAM_PADE': '16', 
                                'RI_SIGMA_X': ''
                            }
                        }
                    }
            }
        elif self.inp_dict['gw_type']=='GW-IC':           
            self.xc_gw={
                    'XC_FUNCTIONAL': {'_': 'PBE'}, 
                    'WF_CORRELATION': {
                        'RI': {
                            'RI_METRIC': {
                                'POTENTIAL_TYPE': 'IDENTITY'
                            }
                        }, 
                        'LOW_SCALING': {}, 
                        'RI_RPA': {
                            'GW': {
                                'CORR_MOS_OCC': '10', 
                                'CORR_MOS_VIRT': '10', 
                                'IC': ['', {}]
                            }
                        }
                    }
            }            

        elif self.inp_dict['gw_type']=='GW-LS': 
#    &XC
#      &XC_FUNCTIONAL PBE
#      &END XC_FUNCTIONAL
#      &WF_CORRELATION
#        METHOD RI_RPA_GPW
#        RI OVERLAP
#        ERI_METHOD OS
#        &WFC_GPW
#          ! normally, this EPS_FILTER controls the accuracy and
#          ! the time for the cubic_scaling RPA calculation
#          ! values like this should be really safe
#          EPS_FILTER  1.0E-20
#          EPS_GRID 1.0E-30
#          EPS_PGF_ORB_S 1.0E-30
#        &END
#        &RI_RPA
#          RPA_NUM_QUAD_POINTS    12   ! same as for N^4-scaling GW. For low-scaling GW, we do computations in imaginary time 
#                                      ! and imaginary frequency and we use a special time/frequency grid ("minimax grid"). 
#                                      ! The highest number is 20 because the generation for grids with more than 20 points 
#                                      ! is numerically unstable. 
#                                      ! 12 points is a good compromise between good accuracy, good numerical stability 
#                                      ! and fast computation.
#          MINIMAX
#          IM_TIME
#          &IM_TIME
#           EPS_FILTER_IM_TIME 1.0E-20
#           GROUP_SIZE_3C 32            ! a group size to do computations
#           GROUP_SIZE_P 4              ! another group size to do computations
#           MEMORY_CUT 12               ! high memory cut reduces memory to prevent out of memory but the computation takes longer. 
#                                       ! For Patrick's tensors which may come soon, this number can be lowered.
#           GW
#           MEMORY_INFO
#          &END
#          &RI_G0W0                     ! all the parameters below are as in normal GW
#            FIT_ERROR
#            CORR_OCC 2
#            CORR_VIRT 2
#            CROSSING_SEARCH NEWTON
#            CHECK_FIT
#            EV_SC_ITER 1
#            OMEGA_MAX_FIT 1.0
#            ANALYTIC_CONTINUATION PADE
#            RI OVERLAP
#            RI_SIGMA_X
#            PRINT_GW_DETAILS
#          &END RI_G0W0
#        &END RI_RPA
#      &END
#    &END XC
    
           self.xc_gw={
                       'XC_FUNCTIONAL': {'_': 'PBE'},
                       'WF_CORRELATION' : {
                                          'METHOD'     : 'RI_RPA_GPW',
                                          'RI'         : 'OVERLAP', 
                                          'ERI_METHOD' : 'OS',
                                         'WFC_GPW' : {
                                                     'EPS_FILTER'    : str(self.inp_dict['eps_filter']),
                                                     'EPS_GRID'      : str(self.inp_dict['eps_grid']),
                                                     'EPS_PGF_ORB_S' : str(self.inp_dict['eps_pgf_orb_s'])
                                         },#END
                                         'RI_RPA' : {
                                                    'RPA_NUM_QUAD_POINTS' : '%d' %(np.min([self.inp_dict['rpa_num_quad_points'],20])),
                                                    'MINIMAX'             : '',
                                                    'IM_TIME '             : '',
                                                    'IM_TIME' : {
                                                                'EPS_FILTER_IM_TIME' : str(self.inp_dict['eps_filter_im_time']),
                                                                'GROUP_SIZE_3C'      : str(self.inp_dict['group_size_3c']),
                                                                'GROUP_SIZE_P'       : str(self.inp_dict['group_size_p']),
                                                                'MEMORY_CUT'         : str(self.inp_dict['memory_cut']),
                                                                'GW'                 : '',
                                                                'MEMORY_INFO'        : ''
                                                    },#END IM_TIME
                                                    'RI_G0W0' : {
                                                                'FIT_ERROR'             : '',
                                                                'CORR_OCC'              : '%d' %(self.inp_dict['corr_occ']),
                                                                'CORR_VIRT'             : '%d' %(self.inp_dict['corr_virt']),
                                                                'CROSSING_SEARCH'       : 'NEWTON',
                                                                'CHECK_FIT'             : '',
                                                                'EV_SC_ITER'            : '%d' %(self.inp_dict['ev_sc_iter']),
                                                                'OMEGA_MAX_FIT'         : '1.0',
                                                                'ANALYTIC_CONTINUATION' : 'PADE',
                                                                'RI'                    : 'OVERLAP',
                                                                'RI_SIGMA_X'            : '',
                                                                'PRINT_GW_DETAILS'      : ''
                                                    }#END RI_G0W0
                                         }#END RI_RPA
                       }#END WF_CORRELATION
           }#END XC    
    
        ###END XC FOR GW


        self.sections_dict={
           'SlabGeoOptWorkChain'  :{'run_type' : 'GEO_OPT' , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True},
           'ReplicaWorkChain'     :{'run_type' : 'GEO_OPT' , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True},
           'CellOptWorkChain'     :{'run_type' : 'CELL_OPT', 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True}, 
           'BulkOptWorkChain'     :{'run_type' : 'GEO_OPT' , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True}, 
           'MoleculeOptWorkChain' :{'run_type' : 'GEO_OPT' , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True},
           'GWWorkChain'          :{'run_type' : 'ENERGY'  , 'xc' : self.xc_gw      , 'qs' : self.qs_gw      , 'motion' : False},
           'MoleculeKSWorkChain'  :{'run_type' : 'ENERGY'  , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : False}, # name to just SCF ?
           'NEBWorkChain'         :{'run_type' : 'BAND'    , 'xc' : self.xc_default , 'qs' : self.qs_neb     , 'motion' : True},

        }    
    
################ START INPUT SECTIONS
        self.workchain=self.inp_dict['workchain']
        self.cell=self.inp_dict['cell'].split()
        if len(self.cell) == 3:
            self.cell=np.diag(np.array(self.cell, dtype=float)).flatten().tolist()
        else :
            self.cell=np.array(self.cell, dtype=float).flatten().tolist()
        self.inp = {
            'GLOBAL': {
                'RUN_TYPE': self.sections_dict[self.workchain]['run_type'],
                'WALLTIME': '%d' % (int(self.inp_dict['walltime'])*0.97),
                'PRINT_LEVEL': 'LOW',
                'EXTENDED_FFT_LENGTHS': ''
            },
            'FORCE_EVAL': [],
        }
        
        if self.inp_dict['gw_type']:
            self.inp['GLOBAL']['PRINT_LEVEL']='MEDIUM'
                         
                         
        ### CHECK WHETHER MOTION SECTION NEEDED OR NOT
        if self.sections_dict[self.workchain]['motion']:
            self.inp['MOTION']=self.get_motion()
            
        ### EXTERNAL RESTART from parent folder
        if self.inp_dict['parent_folder'] is not None:
            self.inp['EXT_RESTART'] = {
                'RESTART_FILE_NAME': '   ./parent_calc/aiida-1.restart'
            }    
        ### FORCEVAL case MIXED DFTB
        if self.inp_dict['calc_type'] == 'Mixed DFTB':
            self.inp['FORCE_EVAL'] = [self.force_eval_mixed(),
                                 self.force_eval_fist(),
                                 self.get_force_eval_qs_dftb()
                                ]
            self.inp['MULTIPLE_FORCE_EVALS'] = {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            }
            
        ### FORCEVAL case MIXED DFT    
        elif self.inp_dict['calc_type'] == 'Mixed DFT':
            self.inp_dict['topology'] = 'mol.xyz' 
            self.inp['FORCE_EVAL'] = [self.force_eval_mixed(),
                                 self.force_eval_fist(),
                                 self.get_force_eval_qs_dft()
                                ]

            self.inp['MULTIPLE_FORCE_EVALS'] = {
                'FORCE_EVAL_ORDER': '2 3',
                'MULTIPLE_SUBSYS': 'T'
            }

        ### FULL DFT CALCULATIONS
        elif self.inp_dict['calc_type'] == 'Full DFT':
            ## XYZ file name for DFT section    

            if self.workchain == 'NEBWorkChain':
                full_dft_topology = 'replica1.xyz'
            elif self.workchain == 'SlabGeoOptWorkChain':
                full_dft_topology = 'mol_on_slab.xyz'
            elif self.workchain == 'MoleculeOptWorkChain':
                full_dft_topology = 'mol.xyz'  
            elif self.workchain == 'GWWorkChain':
                full_dft_topology = 'mol.xyz'
            else:
                full_dft_topology = 'bulk.xyz'
            self.inp_dict['topology'] = full_dft_topology    
            self.inp['FORCE_EVAL'] = self.get_force_eval_qs_dft()
            
        # ----
        # add the colvar subsystem for Replica calcs
        subsys_cv = self.inp_dict['subsys_colvar']
        if subsys_cv is not None:
            if isinstance(self.inp['FORCE_EVAL'],list):
                # mixed environment, add only to first force_eval
                self.inp['FORCE_EVAL'][0]['SUBSYS']['COLVAR'] = subsys_cv
            else:
                self.inp['FORCE_EVAL']['SUBSYS']['COLVAR'] = subsys_cv
        # ----

    ### MOTION SECTION
    def get_motion(self):

#        motion = {
#                   'PRINT' : {
#                      'RESTART_HISTORY' :{'_': 'OFF'},
#                   },
#            'CONSTRAINT': {
#                'FIXED_ATOMS': {
#                    'LIST': '%s' % (self.inp_dict['fixed_atoms']),
#                }
#            }
#        }

        motion = {
                   'PRINT' : {
                      'RESTART_HISTORY' :{'_': 'OFF'},
                   }
        }
        if len(self.inp_dict['fixed_atoms'].split()) >0:
            motion['CONSTRAINT']={
                'FIXED_ATOMS': {
                    'LIST': '%s' % (self.inp_dict['fixed_atoms']),
                }
            }

        ### GEO_OPT
        if self.workchain == 'SlabGeoOptWorkChain' or self.workchain == 'BulkOptWorkChain':
            motion['GEO_OPT'] = {
                    'MAX_FORCE': '%f' % (self.inp_dict['max_force']),
                    'MAX_ITER': '1000',
                    'OPTIMIZER': 'BFGS',
                         'BFGS' : {
                             'TRUST_RADIUS' : '[bohr] 0.1',
                         },
            }
        ### END GEO_OPT

        ### CELL_OPT
        if self.workchain == 'CellOptWorkChain':
            motion['CELL_OPT'] = {
                    'OPTIMIZER': 'BFGS',
                    'TYPE': 'DIRECT_CELL_OPT',
                    'MAX_FORCE': '%f' % (self.inp_dict['max_force']),
                    'EXTERNAL_PRESSURE' : '0',
                    'MAX_ITER': '1000'
                     }
            if self.inp_dict['cell_free'] !='FREE':
                motion['CELL_OPT'][str(self.inp_dict['cell_free'])] = ''
        #### END CELL_OPT

        ### NEB
        if self.workchain == 'NEBWorkChain':

            motion['BAND']= {
                    'NPROC_REP'           : self.inp_dict['nproc_rep'],
                    'BAND_TYPE'           : 'CI-NEB',
                    'NUMBER_OF_REPLICA'   : self.inp_dict['nreplicas'],
                    'K_SPRING'            : str(self.inp_dict['spring']),
                    'CONVERGENCE_CONTROL' : {
                        'MAX_FORCE'       : str(self.inp_dict['max_force']),
                        'RMS_FORCE'       : str(float(self.inp_dict['max_force'])*10),
                        'MAX_DR'          : str(float(self.inp_dict['max_force'])*20),
                        'RMS_DR'          : str(float(self.inp_dict['max_force'])*50)
                    },
                    'ROTATE_FRAMES'       : str(self.inp_dict['rotate']),
                    'ALIGN_FRAMES'        : str(self.inp_dict['align']),
                    'CI_NEB': {
                        'NSTEPS_IT'       : str(self.inp_dict['nstepsit'])
                    },
                    'OPTIMIZE_BAND': {
                        'OPT_TYPE': 'DIIS',
                        'OPTIMIZE_END_POINTS': str(self.inp_dict['endpoints']),
                        'DIIS': {
                            'MAX_STEPS': '1000'
                        }
                    },
                    'PROGRAM_RUN_INFO': {
                        'INITIAL_CONFIGURATION_INFO': ''
                    },
                    'CONVERGENCE_INFO': {
                        '_': ''
                    },
                    'REPLICA': []
                }


            # The fun part
            for r in range(int(self.inp_dict['nreplica_files'])):
                motion['BAND']['REPLICA'].append({
                   'COORD_FILE_NAME': 'replica{}.xyz'.format(r+1)
               })

        ### END NEB 

        ### REPLICA CHAIN
        if self.workchain == 'ReplicaWorkChain':
            
            cv_section = {
                'COLLECTIVE': {
                    'COLVAR': 1,
                    'RESTRAINT': {
                        'K': '[{}] {}'.format(self.inp_dict['spring_unit'], self.inp_dict['spring'])
                    },
                    'TARGET': '[{}] {}'.format(self.inp_dict['target_unit'], self.inp_dict['colvar_target']),
                    'INTERMOLECULAR': ''
                }
            }
            if 'CONSTRAINT' in motion:
                motion['CONSTRAINT'] = [motion['CONSTRAINT'], cv_section]
            else:
                motion['CONSTRAINT'] = cv_section
            
        ### END REPLICA CHAIN

        return motion 
    
    ### MULTI FORCEVAL FOR MIXED
    def force_eval_mixed(self):
        first_mol_atom = 1
        last_mol_atom = self.inp_dict['first_slab_atom'] - 1

        mol_delim = (first_mol_atom, last_mol_atom)
        slab_delim = (self.inp_dict['first_slab_atom'], self.inp_dict['last_slab_atom'])

        force_eval = {
            'METHOD': 'MIXED',
            'MIXED': {
                'MIXING_TYPE': 'GENMIX',
                'GROUP_PARTITION': '2 %d' % (int(self.inp_dict['mpi_tasks'])-2),
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
                    'CELL': {'A': '%f %f %f' % (self.cell[0],self.cell[1],self.cell[2]),
                             'B': '%f %f %f' % (self.cell[3],self.cell[4],self.cell[5]),
                             'C': '%f %f %f' % (self.cell[6],self.cell[7],self.cell[8]),                         
                             },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORD_FILE_FORMAT': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                }
            }
        }

        return force_eval
    
    ### FIST FOR MIXED
    def force_eval_fist(self):
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

        element_list = list(set(self.inp_dict['elements']))

        metal_atom = None
        for el in element_list:
            if el in METAL_ATOMS:
                metal_atom = el
                element_list.remove(el)
                break

        if metal_atom is None:
            raise Exception("No valid metal atom found.")

        for x in element_list + [metal_atom]:
            ff['CHARGE'].append({'ATOM': x, 'CHARGE': '0.0'})

        genpot_fun = 'A*exp(-av*r)+B*exp(-ac*r)-C/(r^6)/( 1+exp(-20*(r/R-1)) )'

        genpot_val = {
            'H': '0.878363 1.33747 24.594164 2.206825 32.23516124268186181470 5.84114',
            'else':  '4.13643 1.33747 115.82004 2.206825 113.96850410723008483218 5.84114'
        }

        for x in element_list:
            ff['NONBONDED']['GENPOT'].append(
                {'ATOMS': metal_atom+' ' + x,
                 'FUNCTION': genpot_fun,
                 'VARIABLES': 'r',
                 'PARAMETERS': 'A av B ac C R',
                 'VALUES': genpot_val[x] if x in genpot_val else genpot_val['else'],
                 'RCUT': '15'}
            )

        for x in itertools.combinations_with_replacement(element_list, 2):
            ff['NONBONDED']['LENNARD-JONES'].append(
                {'ATOMS': " ".join(x),
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
                'CELL': {'A': '%f %f %f' % (self.cell[0],self.cell[1],self.cell[2]),
                         'B': '%f %f %f' % (self.cell[3],self.cell[4],self.cell[5]),
                         'C': '%f %f %f' % (self.cell[6],self.cell[7],self.cell[8]),
                         },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORD_FILE_FORMAT': 'XYZ',
                    'CONNECTIVITY': 'OFF',
                },
            },
        }
        return force_eval

    ### DFTB for MIXED
    def get_force_eval_qs_dftb(self):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'QS': {
                    'METHOD': 'DFTB',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'DFTB': {
                        'SELF_CONSISTENT': 'T',
                        'DISPERSION': '%s' % (str(self.inp_dict['vdw_switch'])),
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
                'CELL': {'A': '%f %f %f' % (self.cell[0],self.cell[1],self.cell[2]),
                         'B': '%f %f %f' % (self.cell[3],self.cell[4],self.cell[5]),
                         'C': '%f %f %f' % (self.cell[6],self.cell[7],self.cell[8]),
                         },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol.xyz',
                    'COORD_FILE_FORMAT': 'xyz'
                }
            }
        }

        return force_eval

    # ==========================================================================


    def get_force_eval_qs_dft(self):

        if not self.inp_dict['gw_type']:
            basis_set = 'BASIS_MOLOPT'
        else:
            basis_set = 'GW_BASIS_SET'        

        

        
        ### SCF PRINT
        print_scf={'RESTART': {'EACH': {'QS_SCF' : '0',
                                        'GEO_OPT': '1',
                               },
                               'ADD_LAST': 'NUMERIC',
                               'FILENAME': 'RESTART'
                   },
                   'RESTART_HISTORY': {'_': 'OFF'}
                  }
        
        ### DIAGONALIZATION AND OT
        scf_opt={'OT'   : {'MAX_SCF'   : '20',
                           'SCF_GUESS' : 'RESTART',
                           'EPS_SCF'   : '1.0E-7',
                           'OT': {'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                                  'MINIMIZER'     : 'CG'
                           },
                           'OUTER_SCF': {'MAX_SCF': '50',
                                         'EPS_SCF': '1.0E-7',                                         
                           },
                           'PRINT'    : print_scf
                 },
                 'DIAG' : {'MAX_SCF'         : '500'                      ,
                           'SCF_GUESS'       : 'RESTART'                  ,
                           'EPS_SCF'         : '1.0E-7'                   ,
                           'CHOLESKY'        : 'INVERSE'                  ,
                           'DIAGONALIZATION' : {'ALGORITHM' : 'STANDARD'} ,
                           'MIXING'          : {
                                                'METHOD'    : 'BROYDEN_MIXING' ,
                                                'ALPHA'     : '0.1',
                                                'BETA'      : '1.5',
                                                'NBROYDEN'  : '8'
                           },
                           'PRINT'    : print_scf
                  },
                 'DEFAULT' : {'MAX_SCF'         : '200'                      ,
                              'SCF_GUESS'       : 'RESTART'                  ,
                              'EPS_SCF'         : '1.0E-6'                   ,
                              #'LEVEL_SHIFT'     : 0.1                        ,
                              'PRINT'           : print_scf
                             }
                              
        }

        
        ### SMEARING
        smear = {'_' : 'ON',
                 'METHOD' : 'FERMI_DIRAC',
                 'ELECTRONIC_TEMPERATURE' : '[K] 300'
        }                
        if self.inp_dict['smear']:
               scf_opt['DIAG']['SMEAR'] = smear

        ### ADDED_MOS
        if self.inp_dict['added_mos']:
                scf_opt['ADDED_MOS'] = self.inp_dict['added_mos']
        
        ### FORCEVAL MAIN        
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
                'CHARGE':str(self.inp_dict['charge']),
                'QS': self.sections_dict[self.workchain]['qs'],
                'MGRID': {
                    'CUTOFF': str(self.inp_dict['mgrid_cutoff']),
                    'NGRIDS': '5',
                },
                'SCF': scf_opt[self.inp_dict['diag_method']],
                'XC': self.sections_dict[self.workchain]['xc'],
            },
            'SUBSYS': {
                'CELL': {'A': '%f %f %f' % (self.cell[0],self.cell[1],self.cell[2]),
                         'B': '%f %f %f' % (self.cell[3],self.cell[4],self.cell[5]),
                         'C': '%f %f %f' % (self.cell[6],self.cell[7],self.cell[8]),
                         'SYMMETRY' : self.inp_dict['cell_sym']
                         },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': str(self.inp_dict['topology']),
                    'COORD_FILE_FORMAT': 'xyz',
                },
                'KIND': [],
            }
        }
        if self.inp_dict['multiplicity'] > 0:
            force_eval['DFT']['UKS']=''
            force_eval['DFT']['MULTIPLICITY']=self.inp_dict['multiplicity']
            
        ### POISSON SOLVER
        if self.inp_dict['periodic']:
            force_eval['DFT']['POISSON']={'PERIODIC':self.inp_dict['periodic'],'PSOLVER':self.inp_dict['poisson_solver']}
            force_eval['SUBSYS']['CELL'].update({'PERIODIC':self.inp_dict['periodic']})
        
        ### VDW
        if self.inp_dict['vdw_switch']:
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
            
        ### CENTER COORDINATES
        if self.inp_dict['center_coordinates']:
            force_eval['SUBSYS']['TOPOLOGY']['CENTER_COORDINATES'] = {'_': ''},

        ### KINDS SECTIONS    
        kinds_used = list(set(self.inp_dict['elements']))

        for kind in kinds_used:
            pp = ATOMIC_KINDS[kind]['pseudo']
            bs = ATOMIC_KINDS[kind][basis_set] 
            if  self.inp_dict['gw_type']  in {'GW', 'GW-IC'}:
                bs = ATOMIC_KINDS[kind]['GW_BASIS_SET']            
            force_eval['SUBSYS']['KIND'].append({
                '_': kind,
                'BASIS_SET': bs,
                'POTENTIAL': pp
            })
            if  self.inp_dict['gw_type'] :
                ba = ATOMIC_KINDS[kind]['RI_AUX']
                force_eval['SUBSYS']['KIND'][-1]['BASIS_SET RI_AUX'] = ba
            if  self.inp_dict['gw_type']=='GW-IC' :### ADD SECTION FOR GHOST ATOMS
                force_eval['SUBSYS']['KIND'].append({
                '_': kind+'G',
                'BASIS_SET': bs,
                'POTENTIAL': pp
                })
                ba = ATOMIC_KINDS[kind]['RI_AUX']
                force_eval['SUBSYS']['KIND'][-1]['GHOST'] = 'TRUE'
                force_eval['SUBSYS']['KIND'][-1]['ELEMENT'] = kind
                force_eval['SUBSYS']['KIND'][-1]['BASIS_SET RI_AUX'] = ba


        ### ADD KINDS for SPIN GUESS : DFT AND GW cases
        spin_dict={
            'C'  : {'n':'2','l':'1','nel':[1,-1]},
            'N'  : {'n':'2','l':'1','nel':[1,-1]},
            'Co' : {'n':'3','l':'2','nel':[1,-1]}
        }
        if (self.inp_dict['spin_u'] !='' or self.inp_dict['spin_d'] !='') and self.inp_dict['multiplicity'] > 0  :
            spin_elements=string_range_to_list(self.inp_dict['spin_u'])[0]+string_range_to_list(self.inp_dict['spin_u'])[0]
            spin_elements=list(set([self.inp_dict['elements'][j] for j in spin_elements]))
            for element in spin_elements:
                for u in [1,2]:
                    pp = ATOMIC_KINDS[element]['pseudo']
                    bs = ATOMIC_KINDS[element][basis_set] 
                    force_eval['SUBSYS']['KIND'].append(
                                                        {
                                                        '_': element+str(u),
                                                        'ELEMENT' : element,
                                                        'BASIS_SET': bs,
                                                        'POTENTIAL': pp,
                                                         'BS':
                                                            {
                                                             'ALPHA':
                                                                {
                                                                 'NEL' : spin_dict[element]['nel'][u-1],
                                                                 'L'   : spin_dict[element]['l'],
                                                                 'N'   : spin_dict[element]['n']
                                                                 },
                                                             ####BETA CONSTRAINED TO ALPHA
                                                             'BETA':
                                                                {
                                                                 'NEL' : -1*+spin_dict[element]['nel'][u-1],
                                                                 'L'   : spin_dict[element]['l'],
                                                                 'N'   : spin_dict[element]['n']
                                                                 }
                                                             }
                                                        }
                                                       )           
                    if self.inp_dict['gw_type']:
                        ba = ATOMIC_KINDS[element]['RI_AUX_BASIS_SET']
                        force_eval['SUBSYS']['KIND'][-1]['RI_AUX_BASIS_SET'] = ba                  
            ##### END ADD KINDS

        ### STRESS TENSOR for CELL_OPT
        if self.workchain == 'CellOptWorkChain':
             force_eval['STRESS_TENSOR']= 'ANALYTICAL'
                
        ### RESTART from .wfn IF NOT NEB        
        if self.workchain != 'NEBWorkChain':
            force_eval['DFT']['RESTART_FILE_NAME']='./parent_calc/aiida-RESTART.wfn'

        return force_eval