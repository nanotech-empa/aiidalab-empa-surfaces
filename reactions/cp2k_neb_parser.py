import re

import ase
import numpy as np
from aiida.common import NotExistent, OutputParsingError, exceptions
from aiida.engine import ExitCode
from aiida.orm import Dict
from aiida.parsers import Parser
from aiida.plugins import DataFactory

StructureData = DataFactory("structure")  # pylint: disable=invalid-name
ArrayData = DataFactory("array")


class Cp2kNebParser(Parser):
    """
    Parser for the output of CP2K.
    """

    def parse(self, **kwargs):
        """Receives in input a dictionary of retrieved nodes. Does all the logic here."""

        try:
            _ = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        exit_code = self._parse_stdout()
        if exit_code is not None:
            return exit_code

        exit_code = self._parse_trajectory()
        if exit_code is not None:
            return exit_code

        return ExitCode(0)

    def _parse_stdout(self):
        """Basic CP2K output file parser."""

        from aiida_cp2k.utils import parse_cp2k_output

        fname = self.node.process_class._DEFAULT_OUTPUT_FILE

        if fname not in self.retrieved.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        try:
            output_string = self.retrieved.get_object_content(fname)
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

        """Parse CP2K output into a dictionary."""
        lines = output_string.splitlines()

        result_dict = {"exceeded_walltime": False}

        for line in lines:
            if "The number of warnings for this run is" in line:
                result_dict["nwarnings"] = int(line.split()[-1])
            if "exceeded requested execution time" in line:
                result_dict["exceeded_walltime"] = True
            if "ABORT" in line:
                result_dict["aborted"] = True

        if "aborted" in result_dict:
            return self.exit_codes.ERROR_OUTPUT_CONTAINS_ABORT

        energy_str_list = re.findall(
            r"ENERGIES \[au\] = (.*?)BAND", output_string, re.DOTALL
        )
        energies_list = [
            np.array(e_str.split(), dtype=float) for e_str in energy_str_list
        ]

        energies_arr_data = ArrayData()
        energies_arr_data.set_array("energies", np.array(energies_list))

        dist_str_list = re.findall(
            r"DISTANCES REP = (.*?)ENERGIES", output_string, re.DOTALL
        )
        distances_list = [
            np.array(["0.0"] + ds.split(), dtype=float) for ds in dist_str_list
        ]

        distances_arr_data = ArrayData()
        distances_arr_data.set_array("distances", np.array(distances_list))

        self.out("output_parameters", Dict(dict=result_dict))
        self.out("replica_energies", energies_arr_data)
        self.out("replica_distances", distances_arr_data)

        return None

    def _parse_trajectory(self):
        """CP2K trajectory parser."""

        from aiida_cp2k.utils import parse_cp2k_trajectory
        from ase import Atoms

        fname = (
            self.node.process_class._DEFAULT_RESTART_FILE_NAME
        )  # pylint: disable=protected-access

        # Check if the restart file is present.
        if fname not in self.retrieved.list_object_names():
            raise exceptions.NotExistent(
                "No restart file available, so the output trajectory can't be extracted"
            )

        # Read the restart file.
        try:
            output_string = self.retrieved.get_object_content(fname)
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

        m = re.search(r"\n\s*&CELL\n(.*?)\n\s*&END CELL\n", output_string, re.DOTALL)
        cell_lines = [line.strip().split() for line in m.group(1).split("\n")]
        cell_str = [line[1:] for line in cell_lines if line[0] in "ABC"]
        cell = np.array(cell_str, np.float64)

        matches = re.findall(
            r"\n\s*&COORD\n(.*?)\n\s*&END COORD\n", output_string, re.DOTALL
        )

        coord_line_sets = [
            [line.strip().split() for line in m.split("\n")] for m in matches
        ]
        coord_set_with_elements = coord_line_sets[-1]
        replica_coord_line_sets = coord_line_sets[:-1]
        # remove integers from element names and create tags
        element_list = [
            re.sub(r"[0-9]+", "", line[0]) for line in coord_set_with_elements
        ]
        tags = [int(re.findall(r"\d+", line[0])[0]) for line in coord_set_with_elements]

        for i_rep, rep_coord_lines in enumerate(replica_coord_line_sets):
            positions = np.array(rep_coord_lines, np.float64)
            ase_atoms = ase.Atoms(symbols=element_list, positions=positions, cell=cell)
            ase_atoms.set_tags(tags)

            self.out(
                "opt_replica_%s" % str(i_rep).zfill(3), StructureData(ase=ase_atoms)
            )

        return None


# EOF
