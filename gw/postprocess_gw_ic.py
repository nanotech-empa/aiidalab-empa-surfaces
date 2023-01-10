import matplotlib.pyplot as plt
import numpy as np
from aiida.engine import ExitCode, ToContext, WorkChain, calcfunction
from aiida.orm import Bool, Code, Dict, Float, Int, List, Str, StructureData, load_node


def find_ref_i(gw_ic_params):
    mo = gw_ic_params["mo"]
    homo = gw_ic_params["homo"]
    homo_global_i = [m[h] for m, h in zip(mo, homo)]
    ref_i = int(np.round(np.mean(homo_global_i)))
    return ref_i


def select_orbitals(node, ref_i, n_occ=5, n_virt=5):

    gw_ic = node.outputs.gw_ic_parameters
    gw = node.outputs.gw_output_parameters

    mos = gw_ic["mo"]
    occs = gw_ic["occ"]
    nspin = len(mos)

    scf_levels = gw["g0w0_e_scf"]
    gw_ic_levels = gw_ic["gw_ic_levels"]

    i_start_mo = ref_i - n_occ + 1
    i_end_mo = ref_i + n_virt + 1

    sel_scf_levels = []
    sel_gw_ic_levels = []
    sel_occs = []

    for i_spin in range(nspin):

        try:
            i_start = mos[i_spin].index(i_start_mo)
        except ValueError:
            i_start = 0

        try:
            i_end = mos[i_spin].index(i_end_mo)
        except ValueError:
            i_end = None

        sel_scf_levels.append(scf_levels[i_spin][i_start:i_end])
        sel_gw_ic_levels.append(gw_ic_levels[i_spin][i_start:i_end])
        sel_occs.append(occs[i_spin][i_start:i_end])

    homos = [a.index(0) - 1 for a in sel_occs]

    selected = {
        "scf": np.array(sel_scf_levels),
        "gw+ic": np.array(sel_gw_ic_levels),
        "occ": np.array(sel_occs),
        "homo": np.array(homos),
    }

    return selected


def table(node_list, n_occ=4, n_virt=4, energy_ref_i=0):
    ref_i = find_ref_i(node_list[0].outputs.gw_ic_parameters)

    header = " MO |"
    templ = "%3d |"

    # Count starts from 1!
    mo_indexes = np.arange(ref_i - n_occ + 1, ref_i + n_virt + 1) + 1

    cols = [mo_indexes]

    for node in node_list:

        sel = select_orbitals(node, ref_i, n_occ, n_virt)

        nspin = len(sel["occ"])

        evals = sel["gw+ic"]

        evals -= evals[0][sel["homo"][0] + energy_ref_i]

        cols += [evals[0], sel["occ"][0]]
        label = "{}_gw+ic".format(node.pk)
        if nspin == 1:
            header += "{:^11s}|".format(label)
            templ += " %6.2f  %d |"
        else:
            header += "{:^20s}|".format(label)
            templ += " %6.2f %d %7.2f %d |"
            cols += [evals[1], sel["occ"][1]]

    for i in range(len(cols)):
        cols[i] = cols[i][::-1]

    print(header)
    for row in zip(*cols):
        print(templ % row)


def table_scf(node_list, n_occ=4, n_virt=4, energy_ref_i=0):
    # mode = 'gw' or 'gw+ic'

    ref_i = find_ref_i(node_list[0].outputs.gw_ic_parameters)

    header = " MO |"
    templ = "%3d |"

    # Count starts from 1!
    mo_indexes = np.arange(ref_i - n_occ + 1, ref_i + n_virt + 1) + 1

    cols = [mo_indexes]

    for node in node_list:

        sel = select_orbitals(node, ref_i, n_occ, n_virt)

        nspin = len(sel["occ"])

        evals = sel["scf"]

        evals -= evals[0][sel["homo"][0] + energy_ref_i]

        cols += [evals[0], sel["occ"][0]]
        label = "{}_scf".format(node.pk)
        if nspin == 1:
            header += "{:^11s}|".format(label)
            templ += " %6.2f  %d |"
        else:
            header += "{:^20s}|".format(label)
            templ += " %6.2f %d %7.2f %d |"
            cols += [evals[1], sel["occ"][1]]

    for i in range(len(cols)):
        cols[i] = cols[i][::-1]

    print(header)
    for row in zip(*cols):
        print(templ % row)


def make_levels_plot(node_list, n_occ=4, n_virt=4, energy_ref_i=0, ylim=None):
    plt.figure(figsize=(1.5 * len(node_list), 6))

    pos = 0

    for node in node_list:

        ref_i = find_ref_i(node.outputs.gw_ic_parameters)
        sel = select_orbitals(node, ref_i, n_occ, n_virt)

        occs = sel["occ"]
        nspin = len(occs)

        ens = sel["gw+ic"]

        ens -= ens[0][sel["homo"][0] + energy_ref_i]

        for i_spin in range(nspin):

            if nspin == 2:
                loc = [-0.4, -0.05] if i_spin == 0 else [0.05, 0.4]
            else:
                loc = [-0.4, 0.4]

            for e, o in zip(ens[i_spin], occs[i_spin]):

                color = "blue" if o < 0.5 else "red"
                plt.plot([pos + loc[0], pos + loc[1]], [e, e], "-", color=color, lw=2.0)

        pos += 1

    labels = [f"{n.pk}" for n in node_list]

    plt.title("gw+ic")
    plt.ylabel("Energy [eV]")
    plt.xlim([-0.5, len(labels) - 0.5])
    plt.ylim(ylim)
    plt.xticks(np.arange(0, len(labels)), labels)
    # plt.savefig(f"gw+ic.png", dpi=200, bbox_inches='tight')
    plt.show()
