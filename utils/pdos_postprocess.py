import copy
import re

import numpy as np


def read_and_process_pdos_file(pdos_path):
    try:
        header = open(pdos_path).readline()
    except TypeError:
        header = pdos_path.readline()
    # fermi = float(re.search("Fermi.* ([+-]?[0-9]*[.]?[0-9]+)", header).group(1))
    try:
        kind = re.search(r"atomic kind.(\S+)", header).group(1)
    except Exception:
        kind = None
    data = np.loadtxt(pdos_path)

    # Determine fermi by counting the number of electrons and
    # taking the middle of HOMO and LUMO.
    n_el = int(np.round(np.sum(data[:, 2])))
    if data[0, 2] > 1.5:
        n_el = int(np.round(n_el / 2))
    fermi = 0.5 * (data[n_el - 1, 1] + data[n_el, 1])

    out_data = np.zeros((data.shape[0], 2))
    out_data[:, 0] = (data[:, 1] - fermi) * 27.21138602  # energy
    out_data[:, 1] = np.sum(data[:, 3:], axis=1)  # "contracted pdos"
    return out_data, kind


def process_pdos_files(scf_calc, newversion):
    if newversion:
        retr_files = scf_calc.outputs.slab_retrieved.list_object_names()
        retr_folder = scf_calc.outputs.slab_retrieved
    else:
        retr_files = scf_calc.outputs.retrieved.list_object_names()
        retr_folder = scf_calc.outputs.retrieved

    nspin = 1
    for file in retr_files:
        if file.endswith(".pdos") and "BETA" in file:
            nspin = 2
            break

    dos = {
        "mol": [None] * nspin,
    }

    for file in sorted(retr_files):
        if file.endswith(".pdos"):
            with retr_folder.open(file) as fhandle:
                try:
                    path = fhandle.name
                except AttributeError:
                    path = fhandle

                if "BETA" in file:
                    i_spin = 1
                else:
                    i_spin = 0

                pdos, kind = read_and_process_pdos_file(path)

                if "list1" in file:
                    dos["mol"][i_spin] = pdos
                elif "list" in file:
                    num = re.search("list(.*)-", file).group(1)
                    label = f"sel_{num}"
                    if label not in dos:
                        dos[label] = [None] * nspin
                    dos[label][i_spin] = pdos
                elif "k" in file:
                    # remove any digits from kind
                    kind = "".join([c for c in kind if not c.isdigit()])
                    label = f"kind_{kind}"
                    if label not in dos:
                        dos[label] = [None] * nspin
                    if dos[label][i_spin] is not None:
                        dos[label][i_spin][:, 1] += pdos[:, 1]
                    else:
                        dos[label][i_spin] = pdos

    tdos = None
    for k in dos:
        if k.startswith("kind"):
            if tdos is None:
                tdos = copy.deepcopy(dos[k])
            else:
                for i_spin in range(nspin):
                    tdos[i_spin][:, 1] += dos[k][i_spin][:, 1]
    dos["tdos"] = tdos
    return dos


def create_series_w_broadening(x_values, y_values, x_arr, fwhm, shape="g"):
    spectrum = np.zeros(len(x_arr))

    def lorentzian(x_):
        # factor = np.pi*fwhm/2 # to make maximum 1.0
        return 0.5 * fwhm / (np.pi * (x_**2 + (0.5 * fwhm) ** 2))

    def gaussian(x_):
        sigma = fwhm / 2.3548
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x_**2) / (2 * sigma**2))

    for xv, yv in zip(x_values, y_values):
        if shape == "g":
            spectrum += yv * gaussian(x_arr - xv)
        else:
            spectrum += yv * lorentzian(x_arr - xv)
    return spectrum


def match_and_reduce_spin_channels(om):
    # In principle, for high-spin states such as triplet, we could assume
    # that alpha is always the higher-populated spin.
    # However, for uks-1, we have to check both configurations and pick higher overlap,
    # so might as well do it in all cases

    # In rks case, just remove the 2nd spin index
    if len(om[0]) == 1:
        return [om[0][0]]

    # Same spin channels.
    same_contrib = 0
    for i_spin in range(2):
        same_contrib = np.sum(om[i_spin][i_spin])

    # Opposite spin channels
    oppo_contrib = 0
    for i_spin in range(2):
        oppo_contrib = np.sum(om[i_spin][(i_spin + 1) % 2])

    if same_contrib >= oppo_contrib:
        return [om[i][i] for i in range(2)]
    else:
        return [om[i][(i + 1) % 2] for i in range(2)]


def load_overlap_npz_legacy(loaded_data):
    overlap_data = {
        "nspin_g1": 1,
        "nspin_g2": 1,
        "homo_i_g2": [int(loaded_data["homo_grp2"])],
        "overlap_matrix": [loaded_data["overlap_matrix"]],
        "energies_g1": [loaded_data["en_grp1"]],
        "energies_g2": [loaded_data["en_grp2"]],
    }
    return overlap_data


def load_overlap_npz(npz_path):
    loaded_data = np.load(npz_path, allow_pickle=True)

    if "metadata" not in loaded_data:
        return load_overlap_npz_legacy(loaded_data)

    metadata = loaded_data["metadata"][0]

    overlap_matrix = []

    for i_spin_g1 in range(metadata["nspin_g1"]):
        overlap_matrix.append([])
        for i_spin_g2 in range(metadata["nspin_g2"]):
            overlap_matrix[-1].append(
                loaded_data[f"overlap_matrix_s{i_spin_g1}s{i_spin_g2}"]
            )

    energies_g1 = []
    for i_spin_g1 in range(metadata["nspin_g1"]):
        energies_g1.append(loaded_data[f"energies_g1_s{i_spin_g1}"])

    energies_g2 = []
    orb_indexes_g2 = []
    for i_spin_g2 in range(metadata["nspin_g2"]):
        energies_g2.append(loaded_data[f"energies_g2_s{i_spin_g2}"])
        orb_indexes_g2.append(loaded_data[f"orb_indexes_g2_s{i_spin_g2}"])

    overlap_data = {
        "nspin_g1": metadata["nspin_g1"],
        "nspin_g2": metadata["nspin_g2"],
        "homo_i_g2": metadata["homo_i_g2"],
        "overlap_matrix": overlap_matrix,
        "energies_g1": energies_g1,
        "energies_g2": energies_g2,
        "orb_indexes_g2": orb_indexes_g2,
    }

    overlap_data["overlap_matrix"] = match_and_reduce_spin_channels(
        overlap_data["overlap_matrix"]
    )

    return overlap_data


def get_orbital_label(i_orb_wrt_homo):
    if i_orb_wrt_homo < 0:
        label = "HOMO%+d" % i_orb_wrt_homo
    elif i_orb_wrt_homo == 0:
        label = "HOMO"
    elif i_orb_wrt_homo == 1:
        label = "LUMO"
    elif i_orb_wrt_homo > 1:
        label = "LUMO%+d" % (i_orb_wrt_homo - 1)
    return label


def get_full_orbital_label(i_spin, i_orb, overlap_data):
    energy = overlap_data["energies_g2"][i_spin][i_orb]

    spin_letter = ""
    if overlap_data["nspin_g2"] == 2:
        spin_letter = "a-" if i_spin == 0 else "b-"

    i_wrt_homo = i_orb - overlap_data["homo_i_g2"][i_spin]
    label = get_orbital_label(i_wrt_homo)

    full_label = f"{spin_letter}{label:6} (E={energy:5.2f})"

    if "orb_indexes_g2" in overlap_data:
        index = overlap_data["orb_indexes_g2"][i_spin][i_orb]
        full_label = f"MO{index:2} {full_label}"
    return full_label


def get_full_orbital_labels(overlap_data):
    labels = []
    for i_spin in range(overlap_data["nspin_g2"]):
        labels.append([])
        for i_orb in range(len(overlap_data["energies_g2"][i_spin])):
            labels[-1].append(get_full_orbital_label(i_spin, i_orb, overlap_data))
    return labels
