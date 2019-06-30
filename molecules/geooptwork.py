from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.remote import RemoteData
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

class MoleculeOptWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(MoleculeOptWorkChain, cls).define(spec)
        
        spec.input('parameters', valid_type    = ParameterData)
        spec.input('code'      , valid_type    = Code)
        spec.input('structure' , valid_type    = StructureData)
        spec.input('parent_folder', valid_type = RemoteData, default=None, required=False)

        spec.outline(
            #cls.print_test,
            cls.run_moleculeopt,
            while_(cls.not_converged)(
                cls.run_moleculeopt_again
            ),
        )
        spec.dynamic_output()

    # ==========================================================================
    def not_converged(self):
        return self.ctx.molecule_opt.res.exceeded_walltime
    # ==========================================================================
    def print_test(self):
        self.report("Reporting test")
        self.report("cell_str: %s %s" % (self.inputs.cell_str, str(self.inputs.cell_str)))

    # ==========================================================================
    def run_moleculeopt(self):
        self.report("Running CP2K molecule optimization")

        the_dict = self.inputs.parameters.get_dict()
        inputs = self.build_calc_inputs(code           = self.inputs.code,
                                        structure      = self.inputs.structure,
                                        input_dict     = the_dict )

        self.report("inputs: "+str(inputs))
        self.report("parameters: "+str(inputs['parameters'].get_dict()))
        self.report("settings: "+str(inputs['settings'].get_dict()))        
        
        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(molecule_opt=Calc(future))

    # ==========================================================================
    def run_moleculeopt_again(self):
        # TODO: make this nicer.
        
        the_dict = self.inputs.parameters.get_dict()
        the_dict['parent_folder'] = self.ctx.molecule_opt.out.remote_folder
        inputs_new = self.build_calc_inputs(code       = self.inputs.code,
                                        structure      = self.inputs.structure,
                                        input_dict     = the_dict )        

        

        self.report("inputs (restart): "+str(inputs_new))
        future_new = submit(Cp2kCalculation.process(), **inputs_new)
        return ToContext(molecule_opt=Calc(future_new))

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls,
                          code          = None,
                          structure     = None,
                          input_dict    = None):

        inputs = {}
        inputs['_label'] = "molecule_opt"
        inputs['code'] = code
        inputs['file'] = {}

 
        atoms = structure.get_ase()  # slow
    
        spin_guess = cls.extract_spin_guess(structure)
        molslab_f = cls.make_geom_file(atoms, "mol.xyz", spin_guess)
        
        inputs['file']['input_xyz'] = molslab_f

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
     
        inp = Get_CP2K_Input(input_dict = input_dict).inp
        
        if 'parent_folder' in input_dict.keys():
            inp['EXT_RESTART'] = {
                'RESTART_FILE_NAME': './parent_calc/aiida-1.restart'
            }
            inputs['parent_folder'] = input_dict['parent_folder']

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.pdb']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": input_dict['num_machines']},
            "max_wallclock_seconds": input_dict['walltime'],
        }

        return inputs

    # ==========================================================================
    @classmethod
    def make_geom_file(cls, atoms, filename, spin_guess=None):
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
        
        
        final_str = "%d\n%s\n" % (n_atoms, comment) + "".join(modif_lines)

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
