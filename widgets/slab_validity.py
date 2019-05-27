
def slab_is_valid(slab_analyzed,dft_type):
    
    if dft_type in ['Mixed DFTB', 'Mixed DFT']:

        first_metal_atom=slab_analyzed['slabatoms'][0]

        if slab_analyzed['slab_elements'][first_metal_atom] not in ['Ag','Au','Cu']:
            return (False, "Didn't find Au, Ag or Cu.")

        if len(slab_analyzed['adatoms'])>0:
            return (False, "Found Adatoms")

        if len(slab_analyzed['unclassified'])>0:
            return (False, "Found unclassified atoms")

        for the_mol in slab_analyzed['all_molecules']:
            for mol_elem in the_mol:
                if mol_elem > first_metal_atom:
                    return (False, "Molecule is not at the beginning of the structure.")
        
    return (True, "")