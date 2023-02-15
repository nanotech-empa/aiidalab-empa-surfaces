
from aiida.engine import CalcJob
from aiida.common.utils import classproperty
from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm import SinglefileData
from aiida.orm import RemoteData
from aiida.common import CalcInfo, CodeInfo
from aiida.common import InputValidationError


class HrstmCalculation(CalcJob):
    
    # --------------------------------------------------------------------------
    @classmethod
    def define(cls, spec):
        super(HrstmCalculation, cls).define(spec)
        spec.input('parameters', valid_type=Dict, help='HRSTM input parameters')
        spec.input('parent_calc_folder', valid_type=RemoteData, help='scf folder')
        spec.input('ppm_calc_folder', valid_type=RemoteData, help='ppm folder')

        # Use mpi by default
        spec.input('metadata.options.withmpi', valid_type=bool, default=True)

    # --------------------------------------------------------------------------
    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.
        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        
        # create code info
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        
        param_dict = self.inputs.parameters.get_dict()
        
        cmdline = []
        for key in param_dict:
            cmdline += [key]
            if param_dict[key] != '':
                if isinstance(param_dict[key], list):
                    cmdline += param_dict[key]
                else:
                    cmdline += [param_dict[key]]
        
        codeinfo.cmdline_params = cmdline

        # create calc info
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.codes_info = [codeinfo]

        # file lists
        calcinfo.remote_symlink_list = []
        calcinfo.local_copy_list = []
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = ["*.npy", "*.npz"]
            
        # symlinks
        if 'parent_calc_folder' in self.inputs:
            comp_uuid = self.inputs.parent_calc_folder.computer.uuid
            remote_path = self.inputs.parent_calc_folder.get_remote_path()
            copy_info  = (comp_uuid, remote_path, 'parent_calc_folder/')
            if self.inputs.code.computer.uuid == comp_uuid:  # if running on the same computer - make a symlink
                # if not - copy the folder
                calcinfo.remote_symlink_list.append(copy_info)
            else:
                calcinfo.remote_copy_list.append(copy_info)
                
        if 'ppm_calc_folder' in self.inputs:
            comp_uuid = self.inputs.ppm_calc_folder.computer.uuid
            remote_path = self.inputs.ppm_calc_folder.get_remote_path()
            copy_info  = (comp_uuid, remote_path, 'ppm_calc_folder/')
            if self.inputs.code.computer.uuid == comp_uuid:  # if running on the same computer - make a symlink
                # if not - copy the folder
                calcinfo.remote_symlink_list.append(copy_info)
            else:
                calcinfo.remote_copy_list.append(copy_info)

        return calcinfo

# EOF
