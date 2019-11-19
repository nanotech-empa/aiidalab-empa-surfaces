from apps.surfaces.widgets.analyze_structure import mol_ids_range
import numpy as np

def suggested_parameters(slab_analyzed,dft_type):
    natoms=slab_analyzed['numatoms']
    atoms_to_fix=''
    num_nodes=12
    if slab_analyzed['system_type']=='Slab':
        if dft_type in ['Mixed DFTB', 'Mixed DFT']:
            full_slab=slab_analyzed['slabatoms'] + slab_analyzed['bottom_H']
            full_slab = [i for i in full_slab] 
            atoms_to_fix = mol_ids_range(full_slab)
            # -----------------------------------------------------------------------
            # Suggest optimal number of nodes
            if dft_type == 'Mixed DFTB':
                num_nodes = int(np.round(natoms/120.))
            elif dft_type == 'Mixed DFT':
                num_nodes = int(np.round(natoms/45.))

        if dft_type in ['Full DFT']:
            partial_slab = slab_analyzed['bottom_H'] + slab_analyzed['slab_layers'][0] + slab_analyzed['slab_layers'][1]
            partial_slab = [i for i in partial_slab] 
            atoms_to_fix = mol_ids_range(partial_slab)
            num_nodes = int(np.round(natoms/45.))
        
    return (atoms_to_fix,num_nodes)