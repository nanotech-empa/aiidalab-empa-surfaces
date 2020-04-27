from apps.surfaces.widgets.analyze_structure import string_range_to_list
from aiida.orm import StructureData , SinglefileData, Str
from aiida.engine import calcfunction
from ase import Atoms

@calcfunction
def make_geom_file(structure, filename,
                   selection=None,
                   spin_u=lambda: orm.Str(''),
                   spin_d=lambda: orm.Str(''),
                   ic_plane_z=None):
    
    import tempfile
    import shutil
    from io import  StringIO
    
    filename = filename.value

    
    ### the two ways of defining spin seem not to be compatible
    ###spin from ase struct
    #spin_guess = extract_spin_guess(structure)
    
    ###spin_from widgets
    spin_guess = [string_range_to_list(spin_u.value),string_range_to_list(spin_d.value)]
    
    if selection is None:
        atoms = structure.get_ase()
    else:
        atoms = structure.get_ase()[selection]
    n_atoms = len(atoms)
    tmpdir = tempfile.mkdtemp()
    file_path = tmpdir + "/" + filename
    orig_file = StringIO()
    atoms.write(orig_file, format='xyz')
    orig_file.seek(0)
    all_lines = orig_file.readlines()
    comment = all_lines[1].strip()
    orig_lines = all_lines[2:]

    #### modify specie of atoms for spin guess
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
        
    #### adding ghost atoms for image charge calculations
    #### chemical symbol will have a G at the end. No spin guess
    imag_lines = []
    if ic_plane_z is not None:
        image = atoms.copy()
        image.positions[:, 2] = 2*ic_plane_z.value - atoms.positions[:, 2]

        imag_file = StringIO()
        image.write(imag_file, format='xyz')
        imag_file.seek(0)
        imag_lines = imag_file.readlines()[2:]

        imag_lines = [r.split()[0]+"G "+" ".join(r.split()[1:])+"\n" for r in imag_lines]

        n_atoms = 2*len(atoms)       


    final_str = "%d\n%s\n" % (n_atoms, comment) + "".join(modif_lines+imag_lines)

    with open(file_path, 'w') as f:
        f.write(final_str)
    aiida_f = SinglefileData(file=file_path)
    shutil.rmtree(tmpdir)
    return aiida_f

def extract_spin_guess(struct_node):
    sites_list = struct_node.attributes['sites']

    spin_up_inds = []
    spin_dw_inds = []

    for i_site, site in enumerate(sites_list):
        if site['kind_name'][-1] == '1':
            spin_up_inds.append(i_site)
        elif site['kind_name'][-1] == '2':
            spin_dw_inds.append(i_site)

    return [spin_up_inds, spin_dw_inds]
# ==========================================================================