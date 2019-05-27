from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.folder import FolderData
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
from io import StringIO, BytesIO

from apps.surfaces.widgets.get_cp2k_input_dev import Get_CP2K_Input

class GWWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(GWWorkChain, cls).define(spec)
        
        spec.input('parameters', valid_type    = ParameterData)
        spec.input('code'      , valid_type    = Code)
        spec.input('structure' , valid_type    = StructureData)
        spec.input('parent_folder', valid_type = RemoteData, default=None, required=False)
        
        spec.outline(
            #cls.print_test,
            cls.run_gw #,
#            while_(cls.not_converged)(
#                cls.run_gw_again
#            ),
        )
        spec.dynamic_output()

    # ==========================================================================
    def not_converged(self):
        return self.ctx.gw_opt.res.exceeded_walltime
    # ==========================================================================
    def print_test(self):
        self.report("Reporting test")
        self.report("cell_str: %s %s" % (self.inputs.cell_str, str(self.inputs.cell_str)))

    # ==========================================================================
    def run_gw(self):
        self.report("Running CP2K GW")
        
        parameters_dict = self.inputs.parameters.get_dict()
        inputs = self.build_calc_inputs(code           = self.inputs.code,
                                        parent_folder  = None,
                                        structure      = self.inputs.structure,
                                        input_dict     = parameters_dict )

        self.report("inputs: "+str(inputs))
        self.report("parameters: "+str(inputs['parameters'].get_dict()))
        self.report("settings: "+str(inputs['settings'].get_dict()))
        
        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(gw_opt=Calc(future))

    # ==========================================================================
#    def run_gw_again(self):
#        # TODO: make this nicer.
#        the_dict = self.inputs.input_dict.get_dict()
#        the_dict['parent_folder'] = self.ctx.gw_opt.out.remote_folder
#        inputs_new = self.build_calc_inputs(input_dict=the_dict)
#        
#
#        self.report("inputs (restart): "+str(inputs_new))
#        future_new = submit(Cp2kCalculation.process(), **inputs_new)
#        return ToContext(gw_opt=Calc(future_new))

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls,
                          code          = None,
                          parent_folder = None,
                          structure     = None,
                          input_dict    = None):

        inputs = {}
        inputs['_label'] = "gw_opt"
        inputs['code'] = code
        inputs['file'] = {}

 
        atoms = structure.get_ase()# slow
        input_dict['atoms'] = atoms
        
        basis_f = SinglefileData(file='/project/apps/surfaces/Files/RI_HFX_BASIS_all')
        inputs['file']['ri_hfx_basis_all'] = basis_f
        
        if 'IC' not in input_dict['gw_type']:
            ic_plane_z = None
        else:
            ic_plane_z = cls.find_ic_plane_z(atoms, input_dict['ads_height'])
            
        spin_guess = cls.extract_spin_guess(structure)

        molslab_f = cls.make_geom_file(atoms, "mol.xyz", spin_guess=spin_guess, ic_plane_z=ic_plane_z)
        
        inputs['file']['input_xyz'] = molslab_f
        
        # parameters
        cell_ase = atoms.cell.flatten().tolist()
        
        if 'cell' in input_dict.keys():
            if input_dict['cell'] == '' or input_dict['cell'] == None :
                input_dict['cell'] = cell_ase   
            else:
                cell_abc=input_dict['cell'].split()
                input_dict['cell']=np.diag(np.array(cell_abc, dtype=float)).flatten().tolist()
        else:
            input_dict['cell'] = cell_ase

#        remote_computer = code.get_remote_computer()
#        machine_cores = remote_computer.get_default_mpiprocs_per_machine()
     
        #inp =     get_cp2k_input(input_dict = input_dict)
        inp = Get_CP2K_Input(input_dict = input_dict).inp                         
        

        if 'parent_folder' in input_dict.keys():
            inp['EXT_RESTART'] = {
                'RESTART_FILE_NAME': './parent_calc/aiida-1.restart'
            }
            inputs['parent_folder'] = input_dict['remote_calc_folder']

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.pdb']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            'resources' : {'num_machines'             : input_dict['num_machines'],
                           'num_mpiprocs_per_machine' : input_dict['num_mpiprocs_per_machine'],
                           'num_cores_per_mpiproc'    : input_dict['num_cores_per_mpiproc']
                          },
            'max_wallclock_seconds': int(input_dict['walltime']),
        }

        return inputs
    
    # ==========================================================================
    @classmethod
    def make_geom_file(cls, atoms, filename, spin_guess=None, ic_plane_z=None):
        # spin_guess = [[spin_up_indexes], [spin_down_indexes]]
        
        n_atoms = len(atoms)
        
        tmpdir = tempfile.mkdtemp()
        file_path = tmpdir + "/" + filename

        orig_file = BytesIO()
        atoms.write(orig_file, format='xyz')
        orig_file.seek(0)
        all_lines = orig_file.readlines()
        comment = all_lines[1].strip()
        orig_lines = all_lines[2:]
        
        modif_lines = []
        for i_line, line in enumerate(orig_lines):
            new_line = line
            lsp = line.split()
            if spin_guess is not None:
                if i_line in spin_guess[0]:
                    new_line = lsp[0]+"1 " + " ".join(lsp[1:])+"\n"
                if i_line in spin_guess[1]:
                    new_line = lsp[0]+"2 " + " ".join(lsp[1:])+"\n"
            modif_lines.append(new_line)
        
        imag_lines = []
        if ic_plane_z is not None:
            image = atoms.copy()
            image.positions[:, 2] = 2*ic_plane_z - atoms.positions[:, 2]
            
            imag_file = BytesIO()
            image.write(imag_file, format='xyz')
            imag_file.seek(0)
            imag_lines = imag_file.readlines()[2:]

            imag_lines = [r.split()[0]+"G "+" ".join(r.split()[1:])+"\n" for r in imag_lines]
            
            n_atoms = 2*len(atoms)
        
        final_str = "%d\n%s\n" % (n_atoms, comment) + "".join(modif_lines+imag_lines)

        with open(file_path, 'w') as f:
            f.write(final_str)
        aiida_f = SinglefileData(file=file_path)
        shutil.rmtree(tmpdir)
        return aiida_f
    
    # ==========================================================================
    @classmethod
    def extract_spin_guess(cls, struct_node):
        sites_list = struct_node.get_attrs()['sites']
        
        spin_up_inds = []
        spin_dw_inds = []
        
        for i_site, site in enumerate(sites_list):
            if site['kind_name'][-1] == '1':
                spin_up_inds.append(i_site)
            elif site['kind_name'][-1] == '2':
                spin_dw_inds.append(i_site)
        
        return [spin_up_inds, spin_dw_inds]
    
    # ==========================================================================
    @classmethod
    def find_ic_plane_z(cls, atoms, ads_height):
        au_image_plane_height = 1.42 # Kharche et al.
        au_surf = np.mean(atoms.positions[:, 2]) - ads_height
        return au_surf + au_image_plane_height
        

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
