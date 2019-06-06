from apps.surfaces.widgets.find_mol import mol_ids_range
import numpy as np
import itertools

from aiida.orm.data.base import Int, Float, Str, Bool, List
from aiida.orm.data.parameter import  ParameterData

ATOMIC_KINDS = {
    'H' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q1'  , 'RI_HFX_BASIS_all': 'aug-DZVP-GTH-up-up-up-up'  },
    'Au':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q11' , 'RI_HFX_BASIS_all': None  },
    'Ag':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q11' , 'RI_HFX_BASIS_all': None  },
    'Cu':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q11' , 'RI_HFX_BASIS_all': None  },
    'Al':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q3'  , 'RI_HFX_BASIS_all': None  },
    'B' :{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q1'  , 'RI_HFX_BASIS_all': None  },
    'Br':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q7'  , 'RI_HFX_BASIS_all': None  },
    'C' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q4'  , 'RI_HFX_BASIS_all': 'aug-DZVP-GTH-up-up-up-up'  },
    'Ga':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q13' , 'RI_HFX_BASIS_all': None  },        
    'N' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q5'  , 'RI_HFX_BASIS_all': None  },
    'O' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q6'  , 'RI_HFX_BASIS_all': None  },
    'Pd':{'BASIS_MOLOPT' : 'DZVP-MOLOPT-SR-GTH' , 'pseudo' : 'GTH-PBE-q18' , 'RI_HFX_BASIS_all': None  },
    'S' :{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-GTH'   , 'pseudo' : 'GTH-PBE-q6'  , 'RI_HFX_BASIS_all': None  },
    'Zn':{'BASIS_MOLOPT' : 'TZV2P-MOLOPT-SR-GTH', 'pseudo' : 'GTH-PBE-q12' , 'RI_HFX_BASIS_all': None  },
}

for element in ATOMIC_KINDS.keys():
    ATOMIC_KINDS[element]['RI_AUX_BASIS_SET'] = None
    
ATOMIC_KINDS['H']['RI_AUX_BASIS_SET'] = 'RI_aug_DZ'
ATOMIC_KINDS['C']['RI_AUX_BASIS_SET'] = 'RI_aug_DZ'

# possible metal atoms for empirical substrate
METAL_ATOMS = ['Au', 'Ag', 'Cu']

DEFAULT_INPUT_DICT ={'added_mos'             : False                 ,
                     'align'                 : False                 ,
                     'atoms'                 : None                  ,
                     'calc_type'             : 'Full DFT'            ,
                     'cell'                  : None                  ,
                     'cell_free'             : None                  ,
                     'cell_sym'              : None                  , 
                     'center_switch'         : False                 ,
                     'charge'                : 0                     ,
                     'colvar_target'         : None                  ,    
                     'corr_occ'              : 10                    , 
                     'corr_virt'             : 10                    , 
                     'endpoints'             : True                  ,
                     'eps_default'           : -15                   ,
                     'eps_filter'            : -12                   ,
                     'eps_filter_im_time'    : -12                   ,
                     'eps_grid'              : -12                   , 
                     'eps_pgf_orb_s'         : -30                   ,
                     'eps_schwarz'           : -13                   , 
                     'ev_sc_iter'            : 1                     , 
                     'first_slab_atom'       : None                  , 
                     'fixed_atoms'           : ''                    , 
                     'functional'            : 'PBE'                 , 
                     'gw_type'               : None                  , 
                     'group_size'            : 12                    ,
                     'group_size_3c'         : 32                    ,
                     'group_size_p'          : 4                     ,
                     'ic_plane'              : 3.05                  ,
                     'last_slab_atom'        : None                  , 
                     'max_force'             : 0.0001                ,
                     'max_memory'            : 0                     ,
                     'memory_cut'            : 12                    ,
                     'mgrid_cutoff'          : 600                   , 
                     'machine_cores'         : None                  , 
                     'multiplicity'          : 0                     , 
                     'nproc_rep'             : None                  , 
                     'nreplicas'             : None                  , 
                     'nreplica_files'        : None                  , 
                     'nstepsit'              : 5                     ,
                     'ot_switch'             : 'OT'                  ,
                     'parent_folder'         : None                  ,
                     'periodic'              : None                  ,
                     'psolver'               : None                  ,
                     'remote_calc_folder'    : None                  , 
                     'replica_name'          : None                  , 
                     'rotate'                : False                 , 
                     'rpa_num_quad_points'   : 200                   ,
                     'size_freq_integ_group' : 1200                  , 
                     'smear'                 : False                 ,
                     'spin_guess'            : ''                    , 
                     'spring'                : 0.05                  , 
                     'spring_unit'           : None                  , 
                     #'struc_folder'          : None                  , 
                     'subsys_colvar'         : None                  , 
                     'target_unit'           : None                  , 
                     'uks_switch'            : 'RKS'                 , 
                     'vdw_switch'            : None                  , 
                     'walltime'              : 86000                 , 
                     'workchain'             : 'SlabGeoOptWorkChain' , 
                     #'wfn_cp_commands'       : None
}
    

py_type_conversion={type(Str(''))   : str   ,
                   type(Bool(True)) : bool  ,
                   type(Float(1.0)) : float ,
                   type(Int(1))     : int
                  }

def to_py_type(aiida_obj):
    if type(aiida_obj) in py_type_conversion.keys():
        return py_type_conversion[type(aiida_obj)](aiida_obj)
    elif type(aiida_obj) == type(List()):                
        pylist=list(aiida_obj)
        return pylist
    elif type(aiida_obj) == type(ParameterData()):                
        pydict=aiida_obj.get_dict()
        return pydict
    else:
        return aiida_obj

class Get_CP2K_Input():
    def __init__(self,input_dict = None):
    
        self.inp_dict = DEFAULT_INPUT_DICT
        for inp_key in input_dict:
            self.inp_dict[inp_key] = to_py_type(input_dict[inp_key]) 

        self.qs_default={
                    'METHOD': 'GPW',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'EPS_DEFAULT': '1.0E-14',
                    } 
        self.qs_gw={
                    'METHOD'       : 'GPW',
                    'EPS_DEFAULT'  : '1.0E'+str(self.inp_dict['eps_default']),
                    'EPS_PGF_ORB'  : '1.0E-290'
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
        if self.inp_dict['gw_type']=='GW':
#      &XC
#         &WF_CORRELATION
#            GROUP_SIZE  12     ! this is the group size for computing matrix elements. small number: 
#                               !fast computation, but can run out of memory, big number: computation can be slow. 
#                               12 seems fine. Normally, computing integrals is not the dominating part in the GW calculation.
#            METHOD  RI_RPA_GPW
#            &RI_RPA
#               GW
#               &HF
#                  FRACTION  1.0000000      ! for exchange self-energy, we need 100 % Fock exchange
#                  &MEMORY
#                     MAX_MEMORY  0         ! memory reserved for storing 4-center matrix elements. 
#                                           !Can be set to zero because 4c-integrals are only used once here 
#                                           !(in contrast to HF-SCF where one needs them every SCF step).
#                  &END MEMORY
#                  &SCREENING
#                     EPS_SCHWARZ  1.0E-13   ! screening parameter for 4c-integrals. 
#                                            !Determines accuracy and speed of the Hartree-Fock calculation 
#                                            !for the exchange self-energy. 
#                                            !1.0E-13 should give accurate results at reasonable computational cost. 
#                                            !But normally, HFX is not the dominating part in a GW calculation.
#                     SCREEN_ON_INITIAL_P  FALSE
#                  &END SCREENING
#               &END HF
#               &RI_G0W0
#                  ANALYTIC_CONTINUATION  PADE   ! the self-energy Sigma(iw) is computed for imaginary frequency iw, 
#                                                ! then a fit is done on Sigma(iw). The fitting function is a Pade approximant. 
#                                                ! We evaluate this fit afterwards at a real frequency/energy w 
#                                                ! to solve the quasiparticle equation 
#                                                !eps_n^GW = eps_n^DFT + fit(epsilon_n^GW) + Sigma_x - v_xc. 
#                                                !The Pade approximant should be the most reliable fitting function.
#                  CHECK_FIT
#                  CORR_OCC  10                  ! how many quasiparticle energies of occupied molecular orbitals are computed. 
#                                                ! Bigger numbers shouldn't affect the performance too much. 
#                                                ! However, for huge number (>100), computation time can increase linearly.
#                  CORR_VIRT  10                 ! see CORR_OCC, only for unoccupied molecular orbitals
#                  CROSSING_SEARCH  NEWTON       ! Method to solve the quasiparticle equation 
#                                                ! eps_n^GW = eps_n^DFT + fit(epsilon_n^GW) + Sigma_x - v_xc. 
#                                                ! Newton should be the most reliable scheme.
#                  EV_SC_ITER  10                ! We do eigenvalue-selfconsistent GW. At most 10 iterations in the eigenvalues are done.
#                  FIT_ERROR                     ! print some more information
#                  OMEGA_MAX_FIT  1.0            ! the fitting of Sigma(iw) with the Pade approximant is done in an interval 
#                                                ! iw \in i*[-1 Hartree, +1 Hartree]. 
#                                                ! Points outside this interval can be inaccurate because 
#                                                ! of numerical integration scheme...
#                  PRINT_GW_DETAILS              ! print some more information
#               &END RI_G0W0
#               RPA_NUM_QUAD_POINTS  200         ! number of imaginary frequency points iw. 
#                                                ! Main parameter (besides basis set) for determinining accuracy and speed of calculation.
#               SIZE_FREQ_INTEG_GROUP  1200      ! frequency calculation runs in parallel, this number says how many processors 
#                                                ! are used for each frequency group. Could be set to -1 because then 
#                                                ! it is decided automatically. 
#                                                ! A small number can speedup the calculation but may lead to out of memory. 
#                                                ! A big number can slow down the calculation. Maybe set the parameter to -1...
#            &END RI_RPA
#            &WFC_GPW
#               EPS_FILTER  1.0E-12              ! some parameter for computing integrals. 1.0E-12 should be safe. 
#                                                ! Computation time hardly affected by theses parameters.
#               EPS_GRID  1.0E-12
#            &END WFC_GPW
#         &END WF_CORRELATION
#         &XC_FUNCTIONAL PBE
#         &END XC_FUNCTIONAL
#      &END XC            
            
            self.xc_gw={
                    'XC_FUNCTIONAL': {'_': 'PBE'},
                    'WF_CORRELATION':{
                                      'METHOD' : 'RI_RPA_GPW',
                                      'WFC_GPW':{
                                                 'EPS_FILTER' : '1.0E'+str(self.inp_dict['eps_filter']),
                                                 'EPS_GRID' :   '1.0E'+str(self.inp_dict['eps_grid'])
                                      },
                                      'RI_RPA' : {
                                                  'RPA_NUM_QUAD_POINTS' :    '%d' %(self.inp_dict['rpa_num_quad_points']),
                                                  'SIZE_FREQ_INTEG_GROUP' :  '%d' %(self.inp_dict['size_freq_integ_group']) ,
                                                  'GW' :' ',
                                                  'HF' :{
                                                           'FRACTION' :  '1.0000000',
                                                           'SCREENING' :{
                                                                         'EPS_SCHWARZ' :  '1.0E'+str(self.inp_dict['eps_schwarz']),
                                                                         'SCREEN_ON_INITIAL_P' : 'FALSE'
                                                           },
                                                            'MEMORY': {
                                                                       'MAX_MEMORY' : '0' # '%d' %(self.inp_dict['max_memory'])
                                                            },
                                                    },
                                                   'RI_G0W0' :{
                                                               'FIT_ERROR' : ' ',
                                                               'CORR_OCC'  :  '%d' %(self.inp_dict['corr_occ']),
                                                               'CORR_VIRT' :  '%d' %(self.inp_dict['corr_virt']),
                                                               'CROSSING_SEARCH'  : 'NEWTON',
                                                               'CHECK_FIT' : ' ',
                                                               'EV_SC_ITER' : '%d' %(self.inp_dict['ev_sc_iter']),
                                                               'OMEGA_MAX_FIT' : '1.0',
                                                               'ANALYTIC_CONTINUATION' : 'PADE',
                                                               'PRINT_GW_DETAILS' : ' ' 
                                                   },
                                      },
                'GROUP_SIZE' : '%d' %(self.inp_dict['group_size']),
              }
            }
        elif self.inp_dict['gw_type']=='GW-IC':
#! in general, image charge calculations are super-good converged and the parameters shouldn't matter at all
#      &XC
#         &WF_CORRELATION
#            ERI_METHOD  OS
#            GROUP_SIZE  12
#            METHOD  RI_RPA_GPW
#            RI  OVERLAP
#            &RI_RPA
#               &HF
#                  FRACTION  1.0000000
#                  &MEMORY
#                     MAX_MEMORY  0
#                  &END MEMORY
#                  &SCREENING
#                     SCREEN_ON_INITIAL_P  FALSE
#                  &END SCREENING
#               &END HF
#               IM_TIME
#               &IM_TIME
#                  EPS_FILTER_IM_TIME  1.0E-12
#                  GW
#               &END IM_TIME
#               MINIMAX
#               &RI_G0W0
#                  CORR_OCC  3
#                  CORR_VIRT  3
#                  EV_SC_ITER  1
#                  IC
#                  OMEGA_MAX_FIT  .36749843813163794053 ! this parameter is actually not needed in the image 
#                                                       ! charge calculation and doesn't do anything
#                  PRINT_GW_DETAILS
#                  RI  OVERLAP
#               &END RI_G0W0
#               SIZE_FREQ_INTEG_GROUP  1200             ! this parameter is also not doing anything 
#                                                       ! in the image charge calculation and you can delete it
#            &END RI_RPA
#            &WFC_GPW
#               EPS_FILTER  1.0E-12
#               EPS_GRID  1.0E-12
#            &END WFC_GPW
#         &END WF_CORRELATION
#         &XC_FUNCTIONAL PBE-
#         &END XC_FUNCTIONAL-
#      &END XC            
            self.xc_gw={
                     'XC_FUNCTIONAL': {'_': 'PBE'},
                     'WF_CORRELATION' : {
                                           'ERI_METHOD' :  'OS',
                                           'GROUP_SIZE' : '%d' %(self.inp_dict['group_size']),
                                           'METHOD'     : 'RI_RPA_GPW',
                                           'RI'         : 'OVERLAP',
                                           'RI_RPA' : {
                                                      'HF' : {
                                                              'FRACTION' :  '1.0',
                                                              'MEMORY' : {
                                                                 'MAX_MEMORY' :  '0'
                                                              },#END MEMORY
                                                              'SCREENING' : {
                                                                 'SCREEN_ON_INITIAL_P' :  'FALSE'
                                                              },#END SCREENING
                                                      },#END HF
                                                      'IM_TIME ' : '',
                                                      'IM_TIME' : {
                                                                  'EPS_FILTER_IM_TIME' : '1.0E'+str(self.inp_dict['eps_filter_im_time']),
                                                                  'GW' : ''
                                                      },#END IM_TIME
                                                      'MINIMAX' : '',
                                                      'RI_G0W0' : {
                                                                  'CORR_OCC'         : '%d' %(self.inp_dict['corr_occ']),
                                                                  'CORR_VIRT'        : '%d' %(self.inp_dict['corr_virt']),
                                                                  'EV_SC_ITER'       : '1',
                                                                  'IC'               : '',
                                                                  'PRINT_GW_DETAILS' : '',
                                                                  'RI'               : 'OVERLAP',
                                                      },#END RI_G0W0
                                           },#END RI_RPA
                                           'WFC_GPW' : {
                                                       'EPS_FILTER' : '1.0E'+str(self.inp_dict['eps_filter']),
                                                       'EPS_GRID' :   '1.0E'+str(self.inp_dict['eps_grid'])
                                           }#END WFC_GPW
                     }#END WF_CORRELATION                    
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
                                                     'EPS_FILTER'    : '1.0E'+str(self.inp_dict['eps_filter']),
                                                     'EPS_GRID'      : '1.0E'+str(self.inp_dict['eps_grid']),
                                                     'EPS_PGF_ORB_S' : '1.0E'+str(self.inp_dict['eps_pgf_orb_s'])
                                         },#END
                                         'RI_RPA' : {
                                                    'RPA_NUM_QUAD_POINTS' : '%d' %(np.min([self.inp_dict['rpa_num_quad_points'],20])),
                                                    'MINIMAX'             : '',
                                                    'IM_TIME '             : '',
                                                    'IM_TIME' : {
                                                                'EPS_FILTER_IM_TIME' : '1.0E'+str(self.inp_dict['eps_filter_im_time']),
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
              'ReplicaWorkchain'     :{'run_type' : 'GEO_OPT' , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True},
              'CellOptWorkChain'     :{'run_type' : 'CELL_OPT', 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True}, 
              'BulkOptWorkChain'     :{'run_type' : 'GEO_OPT' , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True}, 
              'MoleculeOptWorkChain' :{'run_type' : 'GEO_OPT' , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : True},
              'GWWorkChain'          :{'run_type' : 'ENERGY'  , 'xc' : self.xc_gw      , 'qs' : self.qs_gw      , 'motion' : False},
              'MoleculeKSWorkChain'  :{'run_type' : 'ENERGY'  , 'xc' : self.xc_default , 'qs' : self.qs_default , 'motion' : False},
              'NEBWorkchain'         :{'run_type' : 'BAND'    , 'xc' : self.xc_default , 'qs' : self.qs_neb     , 'motion' : True},

        }    
    
    
    
    
    
    
    

        self.workchain=self.inp_dict['workchain']
        self.cell=self.inp_dict['cell']
        self.inp = {
            'GLOBAL': {
                'RUN_TYPE': self.sections_dict[self.workchain]['run_type'],
                'WALLTIME': '%d' % (int(self.inp_dict['walltime'])*0.97),
                'PRINT_LEVEL': 'LOW',
                'EXTENDED_FFT_LENGTHS': ''
            },
            'FORCE_EVAL': [],
        }
        
        ### CHECK WHETHER MOTION SECTION NEEDED OR NOT
        if self.sections_dict[self.workchain]['motion']:
            self.inp['MOTION']=self.get_motion()
            
        ### EXTERNAL RESTART from parent folder
        if self.inp_dict['parent_folder'] is not None:
            self.inp['EXT_RESTART'] = {
                'RESTART_FILE_NAME': './parent_calc/aiida-1.restart'
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

            if self.workchain == 'NEBWorkchain':
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
            self.inp['FORCE_EVAL'] = [self.get_force_eval_qs_dft()]

        


    ### MOTION SECTION
    def get_motion(self):

        motion = {
                   'PRINT' : {
                      'RESTART_HISTORY' :{'_': 'OFF'},
                   },
            'CONSTRAINT': {
                'FIXED_ATOMS': {
                    'LIST': '%s' % (self.inp_dict['fixed_atoms']),
                }
            }
        }


        ### GEO_OPT
        if self.workchain == 'SlabGeoOptWorkChain' or self.workchain == 'BulkOptWorkChain':
            motion['GEO_OPT'] = {
                    'MAX_FORCE': '%f' % (self.inp_dict['max_force']),
                    'MAX_ITER': '5000',
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
                    'MAX_ITER': '500'
                     }
            if cell_free !='FREE':
                motion['CELL_OPT'][str(self.inp_dict['cell_free'])] = ''
        #### END CELL_OPT

        ### NEB
        if self.workchain == 'NEBWorkchain':

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
        if self.workchain == 'ReplicaWorkchain':

            motion['CONSTRAINT'].append({
                               'COLLECTIVE': {
                                   'COLVAR': 1,
                                   'RESTRAINT': {
                                       'K': '[{}] {}'.format(self.inp_dict['spring_unit'], self.inp_dict['spring'])
                                   },
                                   'TARGET': '[{}] {}'.format(self.inp_dict['target_unit'], self.inp_dict['colvar_target']),
                                   'INTERMOLECULAR': ''
                               }                                              
            })
        ### END REPLICA CHAIN

        return motion 
    
    ### MULTI FORCEVAL FOR MIXED
    def force_eval_mixed(self):
        first_mol_atom = 1
        last_mol_atom = self.inp_dict['first_slab_atom'] - 1

        mol_delim = (first_mol_atom, last_mol_atom)
        slab_delim = (first_slab_atom, last_slab_atom)

        force_eval = {
            'METHOD': 'MIXED',
            'MIXED': {
                'MIXING_TYPE': 'GENMIX',
                'GROUP_PARTITION': '2 %d' % (int(self.inp_dict['machine_cores'])-2),
                'GENERIC': {
                    'ERROR_LIMIT': '1.0E-10',
                    'MIXING_FUNCTION': 'E1+E2',
                    'VARIABLES': 'E1 E2'
                },
                'MAPPING': {
                    'FORCE_EVAL_MIXED': {
                        'FRAGMENT':
                            [{'_': '1', ' ': '%d  %d' % int(mol_delim)},
                             {'_': '2', ' ': '%d  %d' % int(slab_delim)}],
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
                    'COORDINATE': 'XYZ',
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

        element_list = list(self.inp_dict['elements'])

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
                    'COORDINATE': 'XYZ',
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
                        'DISPERSION': '%s' % (str(slef.inp_dict['vdw_switch'])),
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
                    'COORDINATE': 'xyz'
                }
            }
        }

        return force_eval

    # ==========================================================================


    def get_force_eval_qs_dft(self):
        uks_logic='.False.'
        if self.inp_dict['uks_switch'] == 'UKS':
            uks_logic='.True.'
        else:
            uks_logic='.False.'
            self.inp_dict['multiplicity']='0'

        if not self.inp_dict['gw_type']:
            basis_set = 'BASIS_MOLOPT'
        else:
            basis_set = 'RI_HFX_BASIS_all'        

        

        
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
                           'OUTER_SCF': {'MAX_SCF': '15',
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
                'UKS': uks_logic,
                'MULTIPLICITY': str(self.inp_dict['multiplicity']),
                'BASIS_SET_FILE_NAME': basis_set,
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
                'CHARGE':str(self.inp_dict['charge']),
                'QS': self.sections_dict[self.workchain]['qs'],
                'MGRID': {
                    'CUTOFF': str(self.inp_dict['mgrid_cutoff']),
                    'NGRIDS': '5',
                },
                'SCF': scf_opt[self.inp_dict['ot_switch']],
                'XC': self.sections_dict[self.workchain]['xc'],
            },
            'SUBSYS': {
                'CELL': {'A': '%f %f %f' % (self.cell[0],self.cell[1],self.cell[2]),
                         'B': '%f %f %f' % (self.cell[3],self.cell[4],self.cell[5]),
                         'C': '%f %f %f' % (self.cell[6],self.cell[7],self.cell[8]),
                         },
                'TOPOLOGY': {
                    'COORD_FILE_NAME': str(self.inp_dict['topology']),
                    'COORDINATE': 'xyz',
                },
                'KIND': [],
            }
        }

        ### POISSON SOLVER
        if self.inp_dict['periodic']:
            force_eval['DFT']['POISSON']={'PERIODIC':self.inp_dict['periodic'],'PSOLVER':self.inp_dict['psolver']}
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
        if self.inp_dict['center_switch']:
            force_eval['SUBSYS']['TOPOLOGY']['CENTER_COORDINATES'] = {'_': ''},

        ### KINDS SECTIONS    
        kinds_used = list(self.inp_dict['elements'])

        for kind in kinds_used:
            pp = ATOMIC_KINDS[kind]['pseudo']
            bs = ATOMIC_KINDS[kind][basis_set] 
            force_eval['SUBSYS']['KIND'].append({
                '_': kind,
                'BASIS_SET': bs,
                'POTENTIAL': pp
            })
            if  self.inp_dict['gw_type'] :
                ba = ATOMIC_KINDS[kind]['RI_AUX_BASIS_SET']
                force_eval['SUBSYS']['KIND'][-1]['RI_AUX_BASIS_SET'] = ba
            if  self.inp_dict['gw_type']=='GW-IC' :### ADD SECTION FOR GHOST ATOMS
                force_eval['SUBSYS']['KIND'].append({
                '_': kind+'G',
                'BASIS_SET': bs,
                'POTENTIAL': pp
                })
                ba = ATOMIC_KINDS[kind]['RI_AUX_BASIS_SET']
                force_eval['SUBSYS']['KIND'][-1]['GHOST'] = 'TRUE'
                force_eval['SUBSYS']['KIND'][-1]['ELEMENT'] = kind
                force_eval['SUBSYS']['KIND'][-1]['RI_AUX_BASIS_SET'] = ba


        ### ADD KINDS for SPIN GUESS : DFT AND GW cases
        if self.inp_dict['spin_guess'] !='' and self.inp_dict['spin_guess']:
            spin_splitted = str(self.inp_dict['spin_guess']).split() ## e.g. ['C1','-1','1','2','C2','1','1','2']
            for ii,C1 in enumerate(spin_splitted[0::4]):
                element=C1[0:-1]
                pp = ATOMIC_KINDS[element]['pseudo']
                bs = ATOMIC_KINDS[element][basis_set] 
                force_eval['SUBSYS']['KIND'].append(
                                                    {
                                                    '_': C1,
                                                    'ELEMENT' : element,
                                                    'BASIS_SET': bs,
                                                    'POTENTIAL': pp,
                                                     'BS':
                                                        {
                                                         'ALPHA':
                                                            {
                                                             'NEL': spin_splitted[4*ii+1],
                                                             'L': spin_splitted[4*ii+2],
                                                             'N': spin_splitted[4*ii+3]
                                                             },
                                                         ####BETA CONSTRAINED TO ALPHA
                                                         'BETA':
                                                            {
                                                             'NEL': str(-1*int(spin_splitted[4*ii+1])),
                                                             'L': spin_splitted[4*ii+2],
                                                             'N': spin_splitted[4*ii+3]
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
        if self.workchain != 'NEBWorkchain':
            force_eval['DFT']['RESTART_FILE_NAME']='./parent_calc/aiida-RESTART.wfn'

        return force_eval
    
