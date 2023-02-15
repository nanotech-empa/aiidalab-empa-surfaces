from aiida.common import CalcInfo, CodeInfo, InputValidationError
from aiida.common.utils import classproperty
from aiida.engine import CalcJob
from aiida.orm import Dict, RemoteData, SinglefileData, StructureData


class StmCalculation(CalcJob):
    """This is a StmCalculation."""

    # --------------------------------------------------------------------------
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=Dict, help="STM input parameters")
        spec.input("parent_calc_folder", valid_type=RemoteData, help="remote folder")
        spec.input("settings", valid_type=Dict, help="special settings")

        # Use mpi by default
        spec.input("metadata.options.withmpi", valid_type=bool, default=True)

    # --------------------------------------------------------------------------
    def prepare_for_submission(self, folder):
        """Create the input files from the input nodes passed to this instance of the `CalcJob`.
        :param folder: an `aiida.common.folders.Folder` to temporarily write files on disk
        :return: `aiida.common.datastructures.CalcInfo` instance
        """

        settings = self.inputs.settings.get_dict() if "settings" in self.inputs else {}

        # create code info
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid

        param_dict = self.inputs.parameters.get_dict()

        cmdline = []
        for key in param_dict:
            cmdline += [key]
            if param_dict[key] != "":
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

        calcinfo.retrieve_list = settings.pop("additional_retrieve_list", [])

        # symlinks
        if "parent_calc_folder" in self.inputs:
            comp_uuid = self.inputs.parent_calc_folder.computer.uuid
            remote_path = self.inputs.parent_calc_folder.get_remote_path()
            copy_info = (comp_uuid, remote_path, "parent_calc_folder/")
            if (
                self.inputs.code.computer.uuid == comp_uuid
            ):  # if running on the same computer - make a symlink
                # if not - copy the folder
                calcinfo.remote_symlink_list.append(copy_info)
            else:
                calcinfo.remote_copy_list.append(copy_info)

        # check for left over settings
        if settings:
            raise InputValidationError(
                "The following keys have been found "
                + f"in the settings input node {self.pk}, "
                + "but were not understood: "
                + ",".join(settings.keys())
            )

        return calcinfo


# EOF
