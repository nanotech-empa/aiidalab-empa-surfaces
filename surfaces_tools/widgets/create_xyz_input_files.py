import io
import shutil
import tempfile

import aiidalab_widgets_base as awb
import numpy as np
from aiida import engine, orm


@engine.calcfunction
def make_geom_file(
    structure,
    filename,
    selection=None,
    spin_u=lambda: orm.Str(""),
    spin_d=lambda: orm.Str(""),
    ic_plane_z=None,
):
    filename = filename.value

    # Spins from widgets
    spin_guess = [
        awb.utils.string_range_to_list(spin_u.value)[0],
        awb.utils.string_range_to_list(spin_d.value)[0],
    ]
    if selection is None:
        atoms = structure.get_ase()
        tags = np.zeros(len(atoms))
        atoms.set_tags(tags)
    else:
        atoms = structure.get_ase()[selection]
        tags = np.zeros(len(atoms))
        atoms.set_tags(tags)
    n_atoms = len(atoms)
    tmpdir = tempfile.mkdtemp()
    file_path = tmpdir + "/" + filename
    orig_file = io.StringIO()
    atoms.write(orig_file, format="xyz")
    orig_file.seek(0)
    all_lines = orig_file.readlines()
    comment = all_lines[1].strip()
    orig_lines = all_lines[2:]

    # Modify specie of atoms for spin guess.
    modif_lines = []
    for i_line, line in enumerate(orig_lines):
        new_line = line
        lsp = line.split()
        if i_line in spin_guess[0]:
            new_line = lsp[0] + "1 " + " ".join(lsp[1:]) + "\n"
        if i_line in spin_guess[1]:
            new_line = lsp[0] + "2 " + " ".join(lsp[1:]) + "\n"
        modif_lines.append(new_line)

    # Adding ghost atoms for image charge calculations.
    # The chemical symbol will have a G at the end. No spin guess.
    imag_lines = []
    if ic_plane_z is not None:
        image = atoms.copy()
        image.positions[:, 2] = 2 * ic_plane_z.value - atoms.positions[:, 2]

        imag_file = io.StringIO()
        image.write(imag_file, format="xyz")
        imag_file.seek(0)
        imag_lines = imag_file.readlines()[2:]

        imag_lines = [
            r.split()[0] + "G " + " ".join(r.split()[1:]) + "\n" for r in imag_lines
        ]

        n_atoms = 2 * len(atoms)

    final_str = f"{n_atoms}\n{comment}\n" + "".join(modif_lines + imag_lines)

    with open(file_path, "w") as f:
        f.write(final_str)
    aiida_f = orm.SinglefileData(file=file_path)
    shutil.rmtree(tmpdir)
    return aiida_f


def extract_spin_guess(struct_node):
    sites_list = struct_node.attributes["sites"]

    spin_up_inds = []
    spin_dw_inds = []

    for i_site, site in enumerate(sites_list):
        if site["kind_name"][-1] == "1":
            spin_up_inds.append(i_site)
        elif site["kind_name"][-1] == "2":
            spin_dw_inds.append(i_site)

    return [spin_up_inds, spin_dw_inds]
