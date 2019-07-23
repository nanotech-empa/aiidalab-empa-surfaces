from aiida.orm import Code, Dict, RemoteData, SinglefileData
from aiida.orm import Int, Float, Str, Bool
from aiida.engine import WorkChain, ToContext, submit, while_
from aiida_cp2k.calculations import Cp2kCalculation

#workchain_dict={'cp2k_code':Code,'structure':StructureData,'max_force':Float}

class SlabGeoOptWorkChain(WorkChain):

    @classmethod

    def define(cls, spec):
        super(SlabGeoOptWorkChain, cls).define(spec)
        
        spec.input('parameters', valid_type    = Dict)
        spec.input('code'      , valid_type    = Code)
        spec.input('structure' , valid_type    = SinglefileData)
        spec.input('parent_calc_folder', valid_type = RemoteData, required=False)
        spec.input('my_metadata', valid_type=dict, non_db=True, required=False)

        spec.outline(
            cls.build_calc_inputs,            
            while_(cls.not_converged)(
                cls.run_geopt,
                cls.parse_geopt,
            ),
        )
        spec.outputs.dynamic = True    
               
    # ==================================================
    def build_calc_inputs(self):

        self.ctx.n_runs = 0
        self.ctx.inputs = {}
        self.ctx.inputs['file'] = {}
        self.ctx.cp2k_input_dict = self.inputs.parameters.get_dict()
        self.ctx.inputs['file']['input_xyz'] = self.inputs.structure

        # parent folder
        if 'parent_calc_folder' in self.inputs:
            self.ctx.parent_calc_folder = self.inputs.parent_calc_folder        
        
        # settings
        settings = Dict(dict={'additional_retrieve_list': ['*.pdb']})
        self.ctx.inputs['settings'] = settings
        self.ctx.inputs['code'] = self.inputs.code

        #resources
        if 'my_metadata' in self.inputs:
            self.ctx.inputs['metadata'] = self.inputs.my_metadata
            
        else:
            self.ctx.inputs['metadata'] = {
                'options': {
                    "resources": {"num_machines": 4, "num_mpiprocs_per_machine" : 36},
                    "max_wallclock_seconds": 40000,
                }
            }

    # ==========================================================================
    def not_converged(self):
        if not self.ctx.n_runs:
            return True
        return self.ctx.molecule_opt.res.exceeded_walltime

    # ==========================================================================        
        
        
        
        
    
#    def define(cls, spec):
#        super(SlabGeoOptWorkChain, cls).define(spec)
#        spec.input("cp2k_code", valid_type=Code)
#        spec.input("structure", valid_type=StructureData)
#        spec.input("max_force", valid_type=Float, default=Float(0.001))
#        spec.input("calc_type", valid_type=Str, default=Str('Mixed DFTB'))
#        spec.input("workchain", valid_type=Str, default=Str('SlabGeoOptWorkChain'))
#        spec.input("vdw_switch", valid_type=Bool, default=Bool(False))
#        spec.input("mgrid_cutoff", valid_type=Int, default=Int(600))
#        spec.input("fixed_atoms", valid_type=Str, default=Str(''))
#        spec.input("center_switch", valid_type=Bool, default=Bool(False))
#        spec.input("num_machines", valid_type=Int, default=Int(1))
#        spec.input("calc_name", valid_type=Str)
#
#        spec.outline(
#            cls.run_geopt,
#            while_(cls.not_converged)(
#                cls.run_geopt_again
#            ),
#        )
#        spec.dynamic_output()
#
#    # ==========================================================================
#    def not_converged(self):
#        return self.ctx.geo_opt.res.exceeded_walltime
#
    # ==========================================================================
    def run_geopt(self):
        self.report("Running CP2K geometry optimization")
        
        
        self.report("Running CP2K molecule optimization")

        if 'parent_calc_folder' in self.ctx:
            self.ctx.cp2k_input_dict['EXT_RESTART'] = {
                'RESTART_FILE_NAME': './parent_calc/aiida-1.restart'
            }
            self.ctx.inputs['parent_calc_folder'] = self.ctx.parent_calc_folder

        self.ctx.inputs['parameters'] = Dict(dict=self.ctx.cp2k_input_dict)
        self.report("inputs: "+str(self.ctx.inputs))
        self.report("parameters: "+str(self.ctx.inputs['parameters'].get_dict()))
        self.report("settings: "+str(self.ctx.inputs['settings'].get_dict()))
        
        running = self.submit(Cp2kCalculation, **self.ctx.inputs)
        self.ctx.n_runs += 1
        return ToContext(geo_opt=running)        
        

#        inputs = self.build_calc_inputs(structure          = self.inputs.structure,
#                                        cp2k_code          = self.inputs.cp2k_code,
#                                        max_force          = self.inputs.max_force,
#                                        calc_type          = self.inputs.calc_type,
#                                        workchain          = self.inputs.workchain,
#                                        mgrid_cutoff       = self.inputs.mgrid_cutoff,
#                                        vdw_switch         = self.inputs.vdw_switch,
#                                        fixed_atoms        = self.inputs.fixed_atoms,
#                                        center_switch      = self.inputs.center_switch,
#                                        num_machines       = self.inputs.num_machines,
#                                        remote_calc_folder = None)
#
#        self.report("inputs: "+str(inputs))
#        future = submit(Cp2kCalculation.process(), **inputs)
#        return ToContext(geo_opt=Calc(future))


#    @classmethod
#    def build_calc_inputs(cls,
#                          structure          = None, 
#                          cp2k_code          = None, 
#                          max_force          = None, 
#                          calc_type          = None,
#                          workchain          = None,
#                          mgrid_cutoff       = None,
#                          vdw_switch         = None,
#                          fixed_atoms        = None,
#                          center_switch      = None,
#                          num_machines       = None,
#                          remote_calc_folder = None):
#
#        inputs = {}
#        inputs['_label'] = "slab_geo_opt"
#        inputs['code'] = cp2k_code
#        inputs['file'] = {}
#        
#
#        atoms = structure.get_ase()  # slow
#
#        # parameters
#        
#        cell=[atoms.cell[0, 0],atoms.cell[0, 1], atoms.cell[0, 2],
#              atoms.cell[1, 0],atoms.cell[1, 1], atoms.cell[1, 2],
#              atoms.cell[2, 0],atoms.cell[2, 1], atoms.cell[2, 2]]
#
#        remote_computer = cp2k_code.get_remote_computer()
#        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
#        
#        walltime = 86000
#        
#        molslab_f = cls.mk_aiida_file(atoms, "mol_on_slab.xyz")
#        inputs['file']['molslab_coords'] = molslab_f
#        first_slab_atom = None
#        
#        
#        if calc_type != 'Full DFT':
#            
#            # Au potential
#            pot_f = SinglefileData(file='/project/apps/surfaces/slab/Au.pot')
#            inputs['file']['au_pot'] = pot_f
#            
#            mol_indexes = find_mol.extract_mol_indexes_from_slab(atoms)
#            
#            first_slab_atom = len(mol_indexes) + 1
#            
#            mol_f = cls.mk_aiida_file(atoms[mol_indexes], "mol.xyz")
#            inputs['file']['mol_coords'] = mol_f
#            
#            if calc_type == 'Mixed DFTB':
#                walltime = 18000
#   
#
#        inp = get_cp2k_input(cell               = cell,
#                             atoms              = atoms,
#                             first_slab_atom    = first_slab_atom,
#                             last_slab_atom     = len(atoms),
#                             max_force          = max_force,
#                             calc_type          = calc_type,
#                             mgrid_cutoff       = mgrid_cutoff,
#                             vdw_switch         = vdw_switch,
#                             machine_cores      = machine_cores*num_machines,
#                             fixed_atoms        = fixed_atoms,
#                             walltime           = walltime*0.97,
#                             workchain          = workchain,
#                             center_switch      = center_switch,
#                             remote_calc_folder = remote_calc_folder
#                             )
#        if remote_calc_folder is not None:
#            inputs['parent_folder'] = remote_calc_folder
#
#        inputs['parameters'] = Dict(dict=inp)
#
#        # settings
#        settings = Dict(dict={'additional_retrieve_list': ['*.pdb']})
#        inputs['settings'] = settings
#
#        # resources
#        inputs['_options'] = {
#            "resources": {"num_machines": num_machines},
#            "max_wallclock_seconds": walltime,
#        }
#
#        return inputs
#
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
