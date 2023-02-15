
from aiida.engine import CalcJob
from aiida.common.utils import classproperty
from aiida.orm import StructureData
from aiida.orm import Dict
from aiida.orm import SinglefileData
from aiida.orm import RemoteData
from aiida.common import CalcInfo, CodeInfo
from aiida.common import InputValidationError


class AfmCalculation(CalcJob):

    # --------------------------------------------------------------------------
    @classmethod
    def define(cls, spec):
        super(AfmCalculation, cls).define(spec)
        spec.input('parameters', valid_type=Dict, help='AFM input parameters')
        spec.input('parent_calc_folder', valid_type=RemoteData, help='remote folder')
        spec.input('atomtypes', valid_type=SinglefileData, help='atomtypes.ini file')
        spec.input('geo_no_labels', valid_type=SinglefileData, help='geometry without spin labels file')
        
        # Don't use mpi by default
        spec.input('metadata.options.withmpi', valid_type=bool, default=False)


    # --------------------------------------------------------------------------
    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.
        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        
        settings = self.inputs.settings.get_dict() if 'settings' in self.inputs else {}
        
        param_dict = self.inputs.parameters.get_dict()
        
        # ---------------------------------------------------
        # Write params.ini file
        params_fn = folder.get_abs_path("params.ini")
        with open(params_fn, 'w') as f:
            for key, val in param_dict.items():
                line = str(key) + " "
                if isinstance(val, list):
                    line += " ".join(str(v) for v in val)
                else:
                    line += str(val)
                f.write(line + '\n')
        # ---------------------------------------------------
        
        # create code info
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.withmpi = False

        # create calc info
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.codes_info = [codeinfo]

        # file lists
        calcinfo.remote_symlink_list = []
        calcinfo.local_copy_list = [(self.inputs.atomtypes.uuid, self.inputs.atomtypes.filename, 'atomtypes.ini'),(self.inputs.geo_no_labels.uuid, self.inputs.geo_no_labels.filename, 'geom.xyz')]
        calcinfo.remote_copy_list = []
        calcinfo.retrieve_list = ["*/*/*.npy"]
        
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
        
        return calcinfo

# EOF