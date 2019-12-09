from apps.surfaces.widgets.find_mol import mol_ids_range
import numpy as np
import itertools

from aiida.orm import Int, Float, Str, Bool

ATOMIC_KINDS = {
    'H' :('TZV2P-MOLOPT-GTH','GTH-PBE-q1'),
    'Au':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q11'),
    'Ag':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q11'),
    'Cu':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q11'),
    'Al':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q3'),
    'B' :('DZVP-MOLOPT-SR-GTH','GTH-PBE-q3'),
    'Br':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q7'),
    'C' :('TZV2P-MOLOPT-GTH','GTH-PBE-q4'),
    'Ga':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q13'),        
    'N' :('TZV2P-MOLOPT-GTH','GTH-PBE-q5'),
    'O' :('TZV2P-MOLOPT-GTH','GTH-PBE-q6'),
    'Pd':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q18'),
    'S' :('TZV2P-MOLOPT-GTH','GTH-PBE-q6'),
    'Zn':('DZVP-MOLOPT-SR-GTH','GTH-PBE-q12')
}

# possible metal atoms for empirical substrate
METAL_ATOMS = ['Au', 'Ag', 'Cu']

def get_cp2k_input(align                 = None, 
                   atoms                 = None,
                   calc_type             = None,                   
                   cell                  = None,
                   cell_free             = None,
                   cell_sym              = None,
                   center_switch         = None,
                   colvar_target         = None, 
                   corr_occ              = None,
                   corr_virt             = None,
                   endpoints             = None,
                   eps_filter            = None,
                   eps_grid              = None,
                   eps_schwarz           = None,
                   ev_sc_iter            = None,
                   first_slab_atom       = None,
                   fixed_atoms           = None,
                   functional            = None,
                   gw_type               = None,
                   group_size            = None,
                   last_slab_atom        = None,
                   max_force             = None,                    
                   mgrid_cutoff          = None,                    
                   machine_cores         = None,
                   multiplicity          = None,
                   nproc_rep             = None,
                   nreplicas             = None,
                   nreplica_files        = None,
                   nstepsit              = None,
                   remote_calc_folder    = None,
                   replica_name          = None,
                   rotate                = None,
                   rpa_num_quad_points   = None,
                   size_freq_integ_group = None,
                   spin_guess            = None,
                   spring                = None,
                   spring_unit           = None,
                   struc_folder          = None,
                   subsys_colvar         = None,
                   system_charge         = None,
                   target_unit           = None,
                   uks_switch            = None,
                   vdw_switch            = None,
                   walltime              = None, 
                   workchain             = None,
                   wfn_cp_commands       = None
                   ):
    
    print("CALC TYPE ",calc_type)
    RUN_TYPE={'SlabGeoOptWorkChain'  : 'GEO_OPT',
              'NEBWorkchain'         : 'BAND',
              'ReplicaWorkchain'     : 'GEO_OPT',
              'CellOptWorkChain'     : 'CELL_OPT',
              'BulkOptWorkChain'     : 'GEO_OPT',
              'MoleculeOptWorkChain' : 'GEO_OPT'
             }


    inp = {
        'GLOBAL': {
            'RUN_TYPE': RUN_TYPE[workchain.value],
            'WALLTIME': '%d' % walltime,
            'PRINT_LEVEL': 'LOW',
            'EXTENDED_FFT_LENGTHS': ''
        },
        'MOTION': get_motion(align           = align,
                             cell_free       = cell_free,
                             endpoints       = endpoints,
                             fixed_atoms     = fixed_atoms,
                             max_force       = max_force,
                             nproc_rep       = nproc_rep,
                             nreplicas       = nreplicas,
                             nstepsit        = nstepsit,
                             rotate          = rotate,
                             spring          = spring,
                             nreplica_files  = nreplica_files,
                             workchain=workchain
                            ),
        'FORCE_EVAL': [],
    }

    if remote_calc_folder is not None:
        inp['EXT_RESTART'] = {
            'RESTART_FILE_NAME': './parent_calc/aiida-1.restart'
        }    

    if calc_type == 'Mixed DFTB':
        inp['FORCE_EVAL'] = [force_eval_mixed(
                                cell            = cell,
                                first_slab_atom = first_slab_atom,
                                last_slab_atom  = last_slab_atom,
                                machine_cores   = machine_cores),
                                force_eval_fist(
                                          atoms = atoms,
                                          cell  = cell
                                ),
                                get_force_eval_qs_dftb(
                                          cell  = cell,
                                          vdw_switch = vdw_switch
                                )
                            ]
        inp['MULTIPLE_FORCE_EVALS'] = {
            'FORCE_EVAL_ORDER': '2 3',
            'MULTIPLE_SUBSYS': 'T'
        }
    elif calc_type == 'Mixed DFT':
        inp['FORCE_EVAL'] = [force_eval_mixed(
                                              cell             = cell,
                                              first_slab_atom  = first_slab_atom,
                                              last_slab_atom   = last_slab_atom,
                                              machine_cores    = machine_cores),
                                              force_eval_fist(
                                                         atoms = atoms,
                                                         cell  = cell
                                              ),
                                              get_force_eval_qs_dft(
                                                         atoms         = atoms,
                                                         cell          = cell,
                                                         center_switch = center_switch,
                                                         mgrid_cutoff  = mgrid_cutoff,
                                                         vdw_switch    = vdw_switch,
                                                         topology      = 'mol.xyz'
                                                       )
                            ]
        
        inp['MULTIPLE_FORCE_EVALS'] = {
            'FORCE_EVAL_ORDER': '2 3',
            'MULTIPLE_SUBSYS': 'T'
        }
        
###FULL DFT CALCULATIONS
    elif calc_type == 'Full DFT':
        ## XYZ file name for DFT section    

        if workchain.value == 'NEBWorkchain':
            full_dft_topology = 'replica1.xyz'
        elif workchain.value == 'SlabGeoOptWorkChain':
            full_dft_topology = 'mol_on_slab.xyz'
        elif workchain.value == 'MoleculeOptWorkChain':
            full_dft_topology = 'mol.xyz'            
        else:
            full_dft_topology = 'bulk.xyz'
            
        inp['FORCE_EVAL'] = [get_force_eval_qs_dft(
                                atoms                 = atoms                 ,
                                cell                  = cell                  ,
                                cell_sym              = cell_sym              ,
                                center_switch         = center_switch         ,
                                corr_occ              = corr_occ              ,
                                corr_virt             = corr_virt             ,
                                eps_filter            = eps_filter            ,
                                eps_grid              = eps_grid              ,
                                eps_schwarz           = eps_schwarz           ,
                                ev_sc_iter            = ev_sc_iter            ,
                                gw_type               = gw_type               ,
                                group_size            = group_size            ,
                                mgrid_cutoff          = mgrid_cutoff          ,
                                multiplicity          = multiplicity          ,
                                rpa_num_quad_points   = rpa_num_quad_points   ,
                                size_freq_integ_group = size_freq_integ_group ,
                                spin_guess            = spin_guess            ,
                                system_charge         = system_charge         ,
                                topology              = full_dft_topology     ,
                                uks_switch            = uks_switch            ,                               
                                vdw_switch            = vdw_switch            ,                                
                                workchain             = workchain
                             )
                            ]

    return inp


###MOTION SECTION
def get_motion(fixed_atoms    = None,
               cell_free      = None,
               max_force      = None,
               nproc_rep      = None,
               nreplicas      = None,
               nreplica_files = None,
               spring         = None,
               rotate         = None,
               align          = None,
               nstepsit       = None,
               endpoints      = None,
               workchain      = None
              ):
    
    motion = {
               'PRINT' : {
                  'RESTART_HISTORY' :{'_': 'OFF'},
               },
        'CONSTRAINT': {
            'FIXED_ATOMS': {
                'LIST': '%s' % (fixed_atoms),
            }
        }
    }

  
    #GEO_OPT
    if workchain.value == 'SlabGeoOptWorkChain' or workchain.value == 'BulkOptWorkChain':
        motion['GEO_OPT'] = {
                'MAX_FORCE': '%f' % (max_force),
                'MAX_ITER': '5000',
                'OPTIMIZER': 'BFGS',
                     'BFGS' : {
                         'TRUST_RADIUS' : '[bohr] 0.1',
                     },
        }
    #END GEO_OPT
    
    #CELL_OPT
    if workchain.value == 'CellOptWorkChain':
        motion['CELL_OPT'] = {
                'OPTIMIZER': 'BFGS',
                'TYPE': 'DIRECT_CELL_OPT',
                'MAX_FORCE': '%f' % (max_force),
                'EXTERNAL_PRESSURE' : '0',
                'MAX_ITER': '500'
                 }
        if cell_free !='FREE':
            motion['CELL_OPT'][str(cell_free)] = ''
    #END CELL_OPT
    
    #NEB
    if workchain.value == 'NEBWorkchain':
            
        motion['BAND']= {
                'NPROC_REP': nproc_rep,
                'BAND_TYPE': 'CI-NEB',
                'NUMBER_OF_REPLICA': nreplicas,
                'K_SPRING': str(spring),
                'CONVERGENCE_CONTROL': {
                    'MAX_FORCE': str(max_force),
                    'RMS_FORCE': str(Float(max_force)*10),
                    'MAX_DR': str(Float(max_force)*20),
                    'RMS_DR': str(Float(max_force)*50)
                },
                'ROTATE_FRAMES': str(rotate)[0],
                'ALIGN_FRAMES': str(align)[0],
                'CI_NEB': {
                    'NSTEPS_IT': str(nstepsit)
                },
                'OPTIMIZE_BAND': {
                    'OPT_TYPE': 'DIIS',
                    'OPTIMIZE_END_POINTS': str(endpoints)[0],
                    'DIIS': {
                        'MAX_STEPS': 1000
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
        for r in range(nreplica_files):
            motion['BAND']['REPLICA'].append({
               'COORD_FILE_NAME': 'replica{}.xyz'.format(r+1)
           })

    ##END NEB 
           
    ##REPLICA CHAIN
    if workchain.value == 'ReplicaWorkchain':
        
        motion['CONSTRAINT'].append({
                           'COLLECTIVE': {
                               'COLVAR': 1,
                               'RESTRAINT': {
                                   'K': '[{}] {}'.format(spring_unit, spring)
                               },
                               'TARGET': '[{}] {}'.format(target_unit, colvar_target),
                               'INTERMOLECULAR': ''
                           }                                              
        })
        ##END REPLICA CHAIN
    
    return motion 

def force_eval_mixed(cell=None, first_slab_atom=None, last_slab_atom=None,
                     machine_cores=None):
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
                'CELL': {'A': '%f %f %f' % (cell[0],cell[1],cell[2]),
                         'B': '%f %f %f' % (cell[3],cell[4],cell[5]),
                         'C': '%f %f %f' % (cell[6],cell[7],cell[8]),                         
                         },
            'TOPOLOGY': {
                'COORD_FILE_NAME': 'mol_on_slab.xyz',
                'COORDINATE': 'XYZ',
                'CONNECTIVITY': 'OFF',
            }
        }
    }

    return force_eval

def force_eval_fist(atoms=None,cell=None):
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
            'CELL': {'A': '%f %f %f' % (cell[0],cell[1],cell[2]),
                     'B': '%f %f %f' % (cell[3],cell[4],cell[5]),
                     'C': '%f %f %f' % (cell[6],cell[7],cell[8]),
                     },
            'TOPOLOGY': {
                'COORD_FILE_NAME': 'mol_on_slab.xyz',
                'COORDINATE': 'XYZ',
                'CONNECTIVITY': 'OFF',
            },
        },
    }
    return force_eval

def get_force_eval_qs_dftb(cell=None, vdw_switch=None):
    force_eval = {
        'METHOD': 'Quickstep',
        'DFT': {
            'QS': {
                'METHOD': 'DFTB',
                'EXTRAPOLATION': 'ASPC',
                'EXTRAPOLATION_ORDER': '3',
                'DFTB': {
                    'SELF_CONSISTENT': 'T',
                    'DISPERSION': '%s' % (str(vdw_switch)[0]),
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
            'CELL': {'A': '%f %f %f' % (cell[0],cell[1],cell[2]),
                     'B': '%f %f %f' % (cell[3],cell[4],cell[5]),
                     'C': '%f %f %f' % (cell[6],cell[7],cell[8]),
                     },
            'TOPOLOGY': {
                'COORD_FILE_NAME': 'mol.xyz',
                'COORDINATE': 'xyz'
            }
        }
    }

    return force_eval

# ==========================================================================
 

def get_force_eval_qs_dft(
                          atoms                 = None,
                          cell                  = None,
                          cell_sym              = None,
                          center_switch         = None,
                          corr_occ              = None,
                          corr_virt             = None,
                          eps_filter            = None,
                          eps_grid              = None,
                          eps_schwarz           = None,
                          ev_sc_iter            = None,
                          gw_type               = None,
                          group_size            = None,
                          mgrid_cutoff          = None,
                          multiplicity          = None,
                          rpa_num_quad_points   = None,
                          size_freq_integ_group = None,
                          spin_guess            = None,
                          system_charge         = None,
                          topology              = None,
                          uks_switch            = None,
                          vdw_switch            = None,
                          workchain             = None
                         ):
    uks_logic='.False.'
    if uks_switch == 'UKS':
        uks_logic='.True.'
    else:
        uks_logic='.False.'
        multiplicity=int(0)
    if system_charge == None:
        system_charge = int(0)
    qs_default={
                'METHOD': 'GPW',
                'EXTRAPOLATION': 'ASPC',
                'EXTRAPOLATION_ORDER': '3',
                'EPS_DEFAULT': '1.0E-14',
                }    
    qs_dict={'SlabGeoOptWorkChain'  : qs_default,
             'ReplicaWorkchain'     : qs_default,
             'CellOptWorkChain'     : qs_default, 
             'BulkOptWorkChain'     : qs_default, 
             'MoleculeOptWorkChain' : qs_default,
             'NEBWorkchain' :{
                              'METHOD': 'GPW',
                              'EXTRAPOLATION': 'USE_GUESS',
                              'EPS_DEFAULT': '1.0E-14',
                            },
             
            }
    xc_default={
                'XC_FUNCTIONAL': {'_': 'PBE'},
            }

######XC FOR GW    
#    xc_gw={
#            'XC_FUNCTIONAL': {'_': 'PBE'},
#            'WF_CORRELATION':{
#                              'METHOD' : gw_type,
#                              'WFC_GPW':{
#                                         'EPS_FILTER' : '1.0E'+str(eps_filter),
#                                         'EPS_GRID' :   '1.0E'+str(eps_grid)
#                              },
#            'RI_RPA' : {
#                        'RPA_NUM_QUAD_POINTS' :    '%d' %(rpa_num_quad_points),
#                        'SIZE_FREQ_INTEG_GROUP' :  '%d' %(size_freq_integ_group) ,
#                        'GW' :' ',
#                        'HF' :{
#                               'FRACTION' :  '1.0000000',
#                               'SCREENING' :{
#                                             'EPS_SCHWARZ' :  '1.0E'+str(eps_schwarz),
#                                             'SCREEN_ON_INITIAL_P' : 'FALSE'
#                               },
#                                'MEMORY': {
#                                           'MAX_MEMORY' : '512'
#                                },
#                        },
#                       'RI_G0W0' :{
#                                   'FIT_ERROR' : ' ',
#                                   'CORR_OCC'  :  '%d' %(corr_occ),
#                                   'CORR_VIRT' :  '%d' %(corr_virt),
#                                   'CROSSING_SEARCH'  : 'NEWTON',
#                                   'CHECK_FIT' : ' ',
#                                   'EV_SC_ITER' : '%d' %(ev_sc_iter),
#                                   'OMEGA_MAX_FIT' : '1.0',
#                                   'ANALYTIC_CONTINUATION' : 'PADE',
#                                   'PRINT_GW_DETAILS' : ' ' 
#                       },
#          },
#        'GROUP_SIZE' : '%d' %(group_size),
#      }
#    }
######END XC FOR GW
    
    force_eval = {
        'METHOD': 'Quickstep',
        'DFT': {
            'UKS': uks_logic,
            'MULTIPLICITY': '%d' % (multiplicity),
            'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
            'POTENTIAL_FILE_NAME': 'POTENTIAL',
            'CHARGE':'%d' % (system_charge),
            'QS': qs_dict[workchain.value],
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
            'CELL': {'A': '%f %f %f' % (cell[0],cell[1],cell[2]),
                     'B': '%f %f %f' % (cell[3],cell[4],cell[5]),
                     'C': '%f %f %f' % (cell[6],cell[7],cell[8]),
                     },
            'TOPOLOGY': {
                'COORD_FILE_NAME': topology,
                'COORDINATE': 'xyz',
            },
            'KIND': [],
        }
    }

    if vdw_switch:
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

    if center_switch:
        force_eval['SUBSYS']['TOPOLOGY']['CENTER_COORDINATES'] = {'_': ''},

    kinds_used = np.unique(atoms.get_chemical_symbols())

    for kind in kinds_used:
        bs, pp = ATOMIC_KINDS[kind] 
        force_eval['SUBSYS']['KIND'].append({
            '_': kind,
            'BASIS_SET': bs,
            'POTENTIAL': pp
        })
        
##### ADD KINDS for SPIN GUESS
        if spin_guess !='' and spin_guess:
            spin_splitted = str(spin_guess).split() ## e.g. ['C1','-1','1','2','C2','1','1','2']
            for ii,C1 in enumerate(spin_splitted[0::4]):
                element=C1[0:-1]
                bs, pp = ATOMIC_KINDS[element]
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
                                                             'NEL': str(-1*int(spin_splitted[4*ii+1])), ## -1*NEL of ALPHA
                                                             'L': spin_splitted[4*ii+2],
                                                             'N': spin_splitted[4*ii+3]
                                                             }
                                                         }
                                                    }
                                                   )
##### END ADD KINDS
        
        
    if workchain.value == 'CellOptWorkChain':
         force_eval['STRESS_TENSOR']= 'ANALYTICAL'
    if workchain.value != 'NEBWorkchain':
        force_eval['DFT']['RESTART_FILE_NAME']='./parent_calc/aiida-RESTART.wfn'
    
    return force_eval