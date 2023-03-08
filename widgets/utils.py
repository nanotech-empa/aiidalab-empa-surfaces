import subprocess
import re
import numpy as np
from aiida.orm import StructureData, load_node


def find_first_workchain(node):
    """Find the first workchain in the provenance."""
    if isinstance(node, StructureData):
        previous_node = node.creator
    else:
        previous_node = node
    while previous_node is not None:
        lastcalling = previous_node
        previous_node = lastcalling.caller

    return lastcalling


def remote_file_exists(hostname, filepath):
    """Checks if a file exists on a remote host."""
    result = ""
    try:
        ssh = subprocess.check_output(
            ["ssh", "%s" % hostname, "ls %s" % filepath],
            shell=False,
            stderr=subprocess.DEVNULL,
        )
        result = ssh.decode()
    except subprocess.CalledProcessError:
        pass
    return result


def copy_wfn(hostname=None, wfn_search_path=None, wfn_name=None):
    """Cretaes a copy of a wfn in a folder"""
    result = ""
    try:
        if hostname == "localhost":
            result = (
                subprocess.call("cp %s %s" % (wfn_search_path, wfn_name), shell=True)
                == 0
            )
        else:
            ssh = subprocess.check_output(
                ["ssh", "%s" % hostname, "cp %s %s" % (wfn_search_path, wfn_name)],
                shell=False,
                stderr=subprocess.DEVNULL,
            )
            result = ssh.decode()
    except subprocess.CalledProcessError:
        pass
    return result


def structure_available_wfn(
    node_uuid=None,
    relative_replica_id=None,
    current_hostname=None,
    return_path=True,
    dft_params=None,
):
    """
    Checks availability of .wfn file corresponding to a structure and returns the remote path.
    :param node_uuid: uuid of the structure node
    :param relative_replica_id: if a structure has to do with  NEB replica_id / nreplica to
      account for teh case where in a NEB calculation we restart from a different number of replicas
    :param current_hostname: hostname of the current computer
    :param return_path: if True, returns the remote path with filename , otherwise returns the remote folder
    :param dft_params: dictionary with dft parameters needed to check if restart with UKS is possible
    :return: remote path or folder
    """

    struc_node = load_node(node_uuid)
    generating_workchain = find_first_workchain(struc_node)

    if generating_workchain is None:
        return None

    if generating_workchain.inputs.code.computer is None:
        return None

    hostname = generating_workchain.inputs.code.computer.hostname

    if hostname != current_hostname:
        return None

    # check if UKS or RKS and in ase of UKS if matching magnetization options
    orig_dft_params = generating_workchain.inputs.dft_params.get_dict()
    was_uks = "uks" in orig_dft_params and orig_dft_params["uks"]
    is_uks = "uks" in dft_params and orig_dft_params["uks"]
    if was_uks != is_uks:
        return None

    if is_uks:
        magnow = dft_params["magnetization_per_site"]
        nmagnow = [-i for i in magnow]
        same_magnetization = (
            orig_dft_params["magnetization_per_site"] == magnow
            or orig_dft_params["magnetization_per_site"] == nmagnow
        )
        same_multiplicity = (
            orig_dft_params["multiplicity"] == dft_params["multiplicity"]
        )
        if not (same_magnetization and same_multiplicity):
            return None

    was_charge = 0
    if "charge" in orig_dft_params:
        was_charge = orig_dft_params["charge"]
    charge = 0
    if "charge" in dft_params:
        charge = dft_params["charge"]

    if charge != was_charge:
        return None

    if generating_workchain.label == "CP2K_NEB":
        create_a_copy = False
        # it could be that the neb calculatio had a different number of replicas
        nreplica_parent = generating_workchain.inputs.neb_params["number_of_replica"]
        ndigits = len(str(nreplica_parent))
        # structure from a NEB but teh number of NEb images changed thus relative_replica_id must be an input
        if relative_replica_id is not None:
            eff_replica_number = int(
                round(relative_replica_id * nreplica_parent, 0) + 1
            )
        # structure from NEB but I am not doing a NEB calculation
        else:
            eff_replica_number = int(re.findall(r"\d+", struc_node.label)[0])
            create_a_copy = True
        # aiida-BAND2-RESTART.wfn 'replica_%s.xyz' % str(i +2 ).zfill(3)
        wfn_name = "aiida-BAND%s-RESTART.wfn" % str(eff_replica_number).zfill(ndigits)
    elif generating_workchain.label in ["CP2K_GeoOpt", "CP2K_CellOpt"]:
        # In all other cases, e.g. geo opt, replica, ...
        # use the standard name
        wfn_name = "aiida-RESTART.wfn"

    wfn_search_path = (
        generating_workchain.outputs.remote_folder.get_remote_path() + "/" + wfn_name
    )

    if hostname == "localhost":
        wfn_exists = (
            subprocess.call("test -e '{}'".format(wfn_search_path), shell=True) == 0
        )
    else:
        wfn_exists = remote_file_exists(hostname, wfn_search_path) != ""

    if not wfn_exists:
        return None

    if create_a_copy:
        copy_wfn(
            hostname=hostname,
            wfn_search_path=wfn_search_path,
            wfn_name=generating_workchain.outputs.remote_folder.get_remote_path()
            + "/"
            + "aiida-RESTART.wfn",
        )

    if return_path:
        return wfn_search_path
    else:
        return generating_workchain.outputs.remote_folder


def mk_wfn_cp_commands(nreplicas, replica_uuids, selected_computer):
    available_wfn_paths = []
    list_wfn_available = []
    list_of_cp_commands = []

    for ir, node_uuid in enumerate(replica_uuids):

        # in general the number of uuids is <= nreplicas
        relative_replica_id = ir / len(replica_uuids)
        avail_wfn = structure_available_wfn(
            node_uuid, relative_replica_id, selected_computer.hostname
        )

        if avail_wfn:
            list_wfn_available.append(ir)  # example:[0,4,8]
            available_wfn_paths.append(avail_wfn)

    if len(list_wfn_available) == 0:
        return []

    n_images_available = len(replica_uuids)
    n_images_needed = nreplicas
    n_digits = len(str(n_images_needed))
    fmt = "%." + str(n_digits) + "d"

    # assign each initial replica to a block of created reps
    block_size = n_images_needed / float(n_images_available)

    for to_be_created in range(1, n_images_needed + 1):
        name = "aiida-BAND" + str(fmt % to_be_created) + "-RESTART.wfn"

        lwa = np.array(list_wfn_available)

        index_wfn = np.abs(lwa * block_size + block_size / 2 - to_be_created).argmin()

        list_of_cp_commands.append(f"cp {available_wfn_paths[index_wfn]} ./{name}")

    return list_of_cp_commands
