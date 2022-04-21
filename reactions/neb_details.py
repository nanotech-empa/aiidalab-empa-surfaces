from __future__ import absolute_import, print_function

import shutil
import subprocess
import tempfile

import ipywidgets as ipw
import numpy as np
from aiida.orm import Code, Computer, FolderData, load_node
from apps.surfaces.reactions import analyze_structure
from IPython.display import clear_output, display

from .neb_utils import mk_coord_files, mk_wfn_cp_commands

style = {"description_width": "120px"}
layout = {"width": "70%"}
layout3 = {"width": "23%"}


class NebDetails(ipw.VBox):
    def __init__(self, code_drop_down, dft_details_w, **kwargs):
        """Dropdown for DFT details"""

        self.code_drop_down = code_drop_down
        self.dft_details_w = dft_details_w

        ### ---------------------------------------------------------
        ### Define all child widgets contained in this composite widget

        self.proc_rep = ipw.IntText(
            value=324,
            description="# Processors per replica",
            style=style,
            layout=layout,
        )

        self.num_rep = ipw.IntText(
            value=15, description="# replicas", style=style, layout=layout
        )

        self.spring_constant = ipw.FloatText(
            description="Spring constant",
            value=0.05,
            min=0.01,
            max=0.5,
            step=0.01,
            style=style,
            layout=layout,
        )

        self.nsteps_it = ipw.IntText(
            value=5, description="# steps before CI", style=style, layout=layout
        )

        self.optimize_endpoints = ipw.ToggleButton(
            value=False,
            description="Optimize endpoints",
            tooltip="Optimize endpoints",
            style=style,
            layout=layout3,
        )

        self.rotate_frames = ipw.ToggleButton(
            value=False,
            description="Rotate Frames",
            tooltip="Rotate Frames",
            style=style,
            layout=layout3,
        )

        self.align_frames = ipw.ToggleButton(
            value=False,
            description="Align Frames",
            tooltip="Align Frames",
            style=style,
            layout=layout3,
        )

        self.text_replica_pks = ipw.Text(
            placeholder="10000 10005 11113 11140",
            description="Replica pks",
            style=style,
            layout={"width": "50%"},
        )

        self.setup_btn = ipw.Button(
            description="Setup replicas", layout={"width": "20%"}
        )

        self.replica_setup_out = ipw.Output()

        self.setup_success = False

        ### ---------------------------------------------------------
        ### Logic

        def on_setup_btn_press(b):
            with self.replica_setup_out:
                clear_output()

                if self.code_drop_down.selected_code is None:
                    print("please select a computer")
                    self.setup_success = False
                    return

                selected_computer = self.code_drop_down.selected_code.computer
                replica_pks = [int(a) for a in self.text_replica_pks.value.split()]
                nreplicas = self.num_rep.value

                print("Find replica wavefunctions...")
                self.aiida_wfn_cp_list = self.mk_wfn_cp_commands(
                    nreplicas, replica_pks, selected_computer
                )

                print("Writing coordinate files...")

                self.struct_folder = self.generate_struct_folder(
                    calc_type=dft_details_w.calc_type.value
                )

                print(self.struct_folder)

        self.setup_btn.on_click(on_setup_btn_press)

        # output variables:
        self.aiida_wfn_cp_list = None
        self.struct_folder = None

        ### ---------------------------------------------------------
        ### Define the ipw structure and create parent VBOX

        children = [
            self.proc_rep,
            self.num_rep,
            self.spring_constant,
            self.nsteps_it,
            ipw.HBox([self.optimize_endpoints, self.rotate_frames, self.align_frames]),
            ipw.HBox([self.text_replica_pks, self.setup_btn]),
            self.replica_setup_out,
        ]

        super(NebDetails, self).__init__(children=children, **kwargs)

    def generate_struct_folder(self, calc_type="Full DFT"):

        replica_pks = [int(a) for a in self.text_replica_pks.value.split()]
        structures = [load_node(x) for x in replica_pks]

        tmpdir = tempfile.mkdtemp()

        if calc_type != "Full DFT":
            # We need mol0.xyz for the initial mixed force_eval
            mol_fn = tmpdir + "/mol.xyz"
            atoms = structures[0].get_ase()
            slab_analyzed = analyze_structure.analyze(atoms)
            mol_ids = [
                item for sublist in slab_analyzed["all_molecules"] for item in sublist
            ]
            mol = atoms[mol_ids]
            mol.write(mol_fn)

        # And we also write all the replicas up to the final geometry.
        for i, s in enumerate(structures):
            atoms = s.get_ase()
            molslab_fn = tmpdir + "/replica{}.xyz".format(i + 1)
            atoms.write(molslab_fn)

        fd = FolderData(tree=tmpdir)
        shutil.rmtree(tmpdir)
        return fd

    def structure_available_wfn(self, struct_pk, current_hostname):
        """
        Checks availability of .wfn file corresponding to a structure and returns the remote path.
        """

        struct_node = load_node(struct_pk)

        if struct_node.creator is None:
            print("Struct %d .wfn not avail: no creator." % struct_pk)
            return None

        parent_calc = struct_node.creator

        if parent_calc.computer is None:
            print("Struct %d .wfn not avail: creator has no computer." % struct_pk)
            return None

        hostname = parent_calc.computer.hostname

        if hostname != current_hostname:
            print("Struct %d .wfn not avail: different hostname." % struct_pk)
            return None

        if parent_calc.label == "neb":
            print("NEB wfn retrieve not configured")
            return None
            ## parent is NEB
            # imag_nr = int(key.split("_")[-1]) + 1
            # parent_calc = val
            # total_n_reps = parent_calc.get_inputs_dict()['CALL'].get_inputs_dict()['nreplicas']
            # n_digits = len(str(total_n_reps))
            # fmt = "%."+str(n_digits)+"d"
            # wfn_name = "aiida-BAND"+str(fmt % imag_nr)+"-RESTART.wfn"
        else:
            # In all other cases, e.g. geo opt, replica, ...
            # use the standard name
            wfn_name = "aiida-RESTART.wfn"

        wfn_search_path = parent_calc.get_remote_workdir() + "/" + wfn_name
        ssh_cmd = (
            "ssh "
            + hostname
            + " if [ -f "
            + wfn_search_path
            + " ]; then echo 1 ; else echo 0 ; fi"
        )
        wfn_exists = subprocess.check_output(ssh_cmd.split())

        if wfn_exists.decode()[0] != "1":
            print("Struct %d .wfn not avail: file deleted from remote." % struct_pk)
            return None

        return wfn_search_path

    def mk_wfn_cp_commands(self, nreplicas, replica_pks, selected_computer):

        available_wfn_paths = []
        list_wfn_available = []
        list_of_cp_commands = []

        for ir, node_pk in enumerate(replica_pks):

            avail_wfn = self.structure_available_wfn(
                node_pk, selected_computer.hostname
            )

            if avail_wfn:
                list_wfn_available.append(ir)  ## example:[0,4,8]
                available_wfn_paths.append(avail_wfn)

        if len(list_wfn_available) == 0:
            return []

        n_images_available = len(replica_pks)
        n_images_needed = nreplicas
        n_digits = len(str(n_images_needed))
        fmt = "%." + str(n_digits) + "d"

        # assign each initial replica to a block of created reps
        block_size = n_images_needed / float(n_images_available)

        for to_be_created in range(1, n_images_needed + 1):
            name = "aiida-BAND" + str(fmt % to_be_created) + "-RESTART.wfn"

            lwa = np.array(list_wfn_available)

            # index_wfn = np.abs(np.round(lwa*block_size + block_size/2) - to_be_created).argmin()
            index_wfn = np.abs(
                lwa * block_size + block_size / 2 - to_be_created
            ).argmin()

            closest_available = lwa[index_wfn]

            print(name, closest_available)

            list_of_cp_commands.append(
                "cp %s ./%s" % (available_wfn_paths[index_wfn], name)
            )

        return list_of_cp_commands

    def reset(
        self,
        proc_rep=324,
        num_rep=15,
        spring_constant=0.05,
        nsteps_it=5,
        optimize_endpoints=False,
        rotate_frames=False,
        align_frames=False,
        text_replica_pks="",
    ):

        self.proc_rep.value = proc_rep
        self.num_rep.value = num_rep
        self.spring_constant.value = spring_constant
        self.nsteps_it.value = nsteps_it
        self.optimize_endpoints.value = optimize_endpoints
        self.rotate_frames.value = rotate_frames
        self.align_frames.value = align_frames
        self.text_replica_pks.value = text_replica_pks
        # self.job_details={}
        # self.update_job_details()
