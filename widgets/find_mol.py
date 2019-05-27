import numpy as np
import itertools

from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase import Atoms



def extract_mol_indexes_from_slab(atoms):
    atoms.set_pbc([True,True,True])
    exclude=['Au','Ag','Cu','Pd','Ga','Ni']
    possible_mol_atoms=['C','N','H','O','S','F'] #problem of H making everything slow due to bottom_H

    ismol=[]


    zmin=np.min(atoms.positions[:,2])
    all_guess_mol_atoms = [ia.index for ia in atoms if 
                           (ia.symbol in possible_mol_atoms and ia.position[2]>zmin+2.0)]
    
    for ma in all_guess_mol_atoms:
        if ma not in ismol:
            tobeadded=all_connected_to(ma,atoms,exclude)
            if len(tobeadded) >1:
                for tba in tobeadded:
                    ismol.append(tba)

    return ismol


def to_ranges(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]
        
def mol_ids_range(ismol):
    range_string=''
    ranges=list(to_ranges(ismol))
    for i in range(len(ranges)):
        if ranges[i][1]>ranges[i][0]:
            range_string+=str(ranges[i][0])+'..'+str(ranges[i][1])
        else:
            range_string+=' '+str(ranges[i][0])
    return range_string

def all_connected_to(id_atom,atoms,exclude):
    cov_radii = [covalent_radii[a.number] for a in atoms]
    
    atoms.set_pbc([False, False, False])
    nl_no_pbc = NeighborList(cov_radii, bothways = True, self_interaction = False)
    nl_no_pbc.update(atoms)
    atoms.set_pbc([True,True,True])
    
    tofollow=[]
    followed=[]
    isconnected=[]
    tofollow.append(id_atom)
    isconnected.append(id_atom)
    while len(tofollow) > 0:
        indices, offsets = nl_no_pbc.get_neighbors(tofollow[0])
        indices=list(indices)
        followed.append(tofollow[0])
        for i in indices:
            if (i not in isconnected) and (atoms[i].symbol not in exclude):
                tofollow.append(i)
                isconnected.append(i)
        for i in followed:
            if i in tofollow: ### do not remove this check
                tofollow.remove(i)
            #try:
            #    tofollow.remove(i)
            #except:
            #    pass
            #    
            

    return isconnected

def molecules(ismol,atoms):
    all_molecules=[]
    to_be_checked=[i for i in range(len(ismol))]
    all_found=[]
    exclude=['None']
    while len(to_be_checked) >0:
        one_mol=all_connected_to(to_be_checked[0],atoms[ismol],exclude)

        is_new_molecule = True
        for ia in one_mol:
            if ia in all_found:
                is_new_molecule=False
                break
                
        if is_new_molecule:
            all_molecules.append([ismol[ia] for ia in one_mol])
            for ia in one_mol:
                all_found.append(ia)
                to_be_checked.remove(ia)
                
            
    return all_molecules

def to_ranges(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]
        
def mol_ids_range(ismol):
    range_string=''
    shifted_list=[i+1 for i in ismol]
    ranges=list(to_ranges(shifted_list))
    for i in range(len(ranges)):
        if ranges[i][1]>ranges[i][0]:
            range_string+=str(ranges[i][0])+'..'+str(ranges[i][1])+' '
        else:
            range_string+=str(ranges[i][0])+' '
    return range_string

def analyze_slab(atoms):
    atoms.set_pbc([True,True,True])
    
    total_charge=np.sum(atoms.get_atomic_numbers())
    bottom_H=[]
    adatoms=[]
    remaining=[]
    metalatings=[]
    unclassified=[]
    slabatoms=[]
    slab_layers=[]    
    all_molecules=[[]]
    is_a_bulk=False
    is_a_molecule=False
    spins_up   = set(str(the_a.symbol)+str(the_a.tag) for the_a in atoms if the_a.tag == 1)
    spins_down = set(str(the_a.symbol)+str(the_a.tag) for the_a in atoms if the_a.tag == 2)
    #### check if there is vacuum otherwise classify as bulk and skip
    vacuum_x=np.max(atoms.positions[:,0]) - np.min(atoms.positions[:,0]) +7 < atoms.cell[0][0]
    vacuum_y=np.max(atoms.positions[:,1]) - np.min(atoms.positions[:,1]) +7 < atoms.cell[1][1]
    vacuum_z=np.max(atoms.positions[:,2]) - np.min(atoms.positions[:,2]) +7 < atoms.cell[2][2]
    all_elements=list(set(atoms.get_chemical_symbols()))
    cov_radii = [covalent_radii[a.number] for a in atoms]
    
    nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
    nl.update(atoms)
    
    metalating_atoms=['Ag','Au','Cu','Co','Ni','Fe']
    possible_slab_atoms=['Au','Ag','Cu','Pd','Ga','Ni']
    
    summary=''
    if (not vacuum_z) and (not vacuum_x) and (not vacuum_y):
        is_a_bulk=True
        sys_type='Bulk'
        summary='Bulk contains: \n'
        slabatoms=[ia for ia in range(len(atoms))]
        
    if vacuum_x and vacuum_y and vacuum_z:
        is_a_molecule=True
        sys_type='Molecule'
        summary='Molecule: \n'
        all_molecules=molecules([i for i in range(len(atoms))],atoms)
        com=np.average(atoms.positions,axis=0)
        summary+='COM: '+str(com)+', min z: '+str(np.min(atoms.positions[:,2]))
    ####END check
    if not (is_a_bulk or is_a_molecule or vacuum_y):
        sys_type='Slab'
        mol_atoms=extract_mol_indexes_from_slab(atoms)
        all_molecules=molecules(mol_atoms,atoms)


        ## bottom_H   
        zmin=np.min(atoms.positions[:,2])
        listh=[x[0] for x in np.argwhere(atoms.numbers == 1)]
        for ih in listh:
            if atoms[ih].position[2]<zmin+0.8:
                bottom_H.append(ih)


        #nice but too slow and does not distinguish top from bottom    
        #    for h in listh:
        #        indices, offsets = nl.get_neighbors(h)
        #        #aus=[i for i in indices if atoms[i].symbol=='Au' ]
        #        aus=[i for i in indices if (atoms[i].symbol in possible_slab_atoms) ]
        #        for the_au in aus:
        #            iau,oau = nl.get_neighbors(the_au)
        #            h_of_one_au=[ish for ish in iau if atoms[ish].symbol=='H']
        #            if len(h_of_one_au) >1:
        #                bottom_H.append(h)
        #                break

        for ra in range(len(atoms)):
            if ra not in mol_atoms+bottom_H:
                remaining.append(ra)
        for ra in remaining:
            nummetalsneigh=0
            indices, offsets = nl.get_neighbors(ra)
            classified=False
            if len(indices)==0: 
                if atoms[ra].symbol in metalating_atoms:
                    metalatings.append(ra)
                    classified=True
            else:
                for inra in indices:
                    if atoms[inra].symbol in possible_slab_atoms:
                        nummetalsneigh+=1
                if nummetalsneigh >0 and nummetalsneigh <4:
                    adatoms.append(ra)
                    classified=True
                else:
                    slabatoms.append(ra)
                    classified=True
            if not classified:
                unclassified.append(ra) 

        ##slab layers
        dz=1.0
        ddz=0.01

    #    zmin different from one already computed above
        id_zmin=np.argmin(atoms[slabatoms].positions[:,2]) # wrt only slab atoms
        id_zmin=slabatoms[id_zmin] # global index
        zmin=atoms[id_zmin].position[2]
        possible_next_z=[ia[0] for ia in np.argwhere(atoms[slabatoms].positions[:,2] > zmin +dz)]
        next_z=np.min(atoms[slabatoms][possible_next_z].positions[:,2])
        to_be_checked=list(slabatoms)
        while len(to_be_checked)>0:
            #within_dz=np.logical_and(atoms.positions[:,2] > zmin -ddz , atoms.positions[:,2] <next_z -ddz  )
            within_dz=np.logical_and(atoms[slabatoms].positions[:,2] >= zmin , atoms[slabatoms].positions[:,2] <next_z   )
            within_dz=[slabatoms[ia[0]] for ia in np.argwhere(within_dz)]
            slab_layers.append(within_dz)
            for ia in within_dz:
                to_be_checked.remove(ia)
            if len(to_be_checked)>0:
                id_zmin=np.argmin(atoms[to_be_checked].positions[:,2])
                id_zmin=to_be_checked[id_zmin]
                zmin=atoms[id_zmin].position[2]           
                possible_next_z=[ia[0] for ia in np.argwhere(atoms.positions[:,2] > zmin +dz)]
                next_z=np.min(atoms[possible_next_z].positions[:,2])        
        ##end slab layers
        summary='Slab contains: \n'
        
    slab_elements=set(atoms[slabatoms].get_chemical_symbols())

    if len(bottom_H) >0:
        summary+='bottom H: ' + mol_ids_range(bottom_H)   + '\n'
    if len(slabatoms) > 0:    
        summary+='slab atoms: '   + mol_ids_range(slabatoms)  + '\n' 
    for nlayer in range(len(slab_layers)):
        summary+='slab layer '+str(nlayer+1)+': '+mol_ids_range(slab_layers[nlayer])+'\n'    
    if len(adatoms)>0:
        summary+='adatoms: '  + mol_ids_range(adatoms)    + '\n' 
    summary+='#'+str(len(all_molecules))   + ' molecules: ' 
    for nmols in range(len(all_molecules)):
        summary+=mol_ids_range(all_molecules[nmols])
    summary+=' \n' 
    if len(mol_ids_range(metalatings))>0:
        summary+='metal atoms inside molecules: '+ mol_ids_range(metalatings) + '\n'
    if len(mol_ids_range(unclassified))>0:
        summary+='unclassified: ' + mol_ids_range(unclassified)

    return {'total_charge'  : total_charge,
            'the_cell'      : atoms.cell,
            'slab_layers'   : slab_layers,
            'system_type'   : sys_type,
            'bottom_H'      : sorted(bottom_H),
            'slabatoms'     : sorted(slabatoms),
            'adatoms'       : sorted(adatoms),
            'all_molecules' : sorted(all_molecules),
            'metalatings'   : sorted(metalatings),
            'unclassified'  : sorted(unclassified),
            'numatoms'      : len(atoms),
            'all_elements'  : all_elements,
            'slab_elements' : slab_elements,
            'spins_up'      : spins_up,
            'spins_down'    : spins_down,
            'summary':summary
           }

