import numpy as np
import itertools

from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase import Atoms



def boxfilter(x,thr):
    return np.asarray([1 if i<thr else 0 for i in x])
def get_types(frame,thr): ## Piero Gasparotto
    # classify the atmos in:
    # 0=molecule
    # 1=surface atoms
    # 2=adatoms
    # 3=hydrogens on the surf
    #frame=ase frame
    #thr=threashold in the histogram for being considered a surface layer
    nat=frame.get_number_of_atoms()
    atype=np.zeros(nat,dtype=np.int16)
    surftypes=('Au','N','B','O','Cu','Ag','Pg','Ga','Pd','Ga') #types of surface atoms
    lbls=frame.get_chemical_symbols()
    isurf=np.asarray([i for i in range(nat) if lbls[i] in surftypes])
    irest=np.setdiff1d(range(nat), isurf)
    # set surf to standard surf
    atype[isurf]=1
    hist, bin_edges = np.histogram(frame.positions[isurf,2], density=True,bins=100)
    # find the positions of the layers
    layers=bin_edges[np.where(hist>thr)[0]]
    # compute the CV: simply the z_coord_of_the_atoms - the_closest_surface_layer
    #relz=np.asarray([i-layers[np.argsort(np.abs(i-layers))[0]] for i in frame.positions[:,2]])
    relz=np.asarray([np.sort(np.abs(i-layers))[0] for i in frame.positions[:,2]])
    # the surf atoms in an ordered layer have relz=0, this is enough to classify all the different atoms
    # since I know the atomic labes I use a switching function on relz to get 1 for the standard surface atoms
    # and 0 for the adatoms
    for j in [isurf[i] for i,v in enumerate(boxfilter(relz[isurf],0.5)) if v==0]:
        atype[j]=2
      
    # assign the H types
    # get the position of the H at the surf
    ihsurf=[j for j in irest if(relz[j]<1.0 and lbls[j]=='H')]
    if len(ihsurf)>1 :
        # set the H at the surf as type 3
        atype[ihsurf]=3
        # get the histo in z of these hydrogens
        hist, bin_edges = np.histogram(frame.positions[ihsurf,2], density=True,bins=100)
        # check if there are layers
        layersh=bin_edges[np.where(hist>thr)[0]]
        # check if an hydrogen is part or not of the layers
        relzh=np.asarray([np.sort(np.abs(i-layersh))[0] for i in frame.positions[ihsurf,2]])
        # hydrogens not in the layers should be adatoms
        for j in [ihsurf[i] for i,v in enumerate(boxfilter(relzh,0.5)) if v==0]:
            atype[j]=2
        
    # assign the other types
    moltypes=('N','B','O')
    #for j in irest:
    #    if(relz[j]<1.0 and lbls[j]=='H'):
    #        atype[j]=3 # hydrogens of the surf
    for j in np.asarray([i for i in isurf if lbls[i] in moltypes]):
        if(relz[j]>0.5):
            atype[j]=0
        #    atype[j]=4 # oxygens of the surf
        #elif(zz[]):
    return atype

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
    vacuum_x=np.max(atoms.positions[:,0]) - np.min(atoms.positions[:,0]) +3 < atoms.cell[0][0]
    vacuum_y=np.max(atoms.positions[:,1]) - np.min(atoms.positions[:,1]) +3 < atoms.cell[1][1]
    vacuum_z=np.max(atoms.positions[:,2]) - np.min(atoms.positions[:,2]) +3 < atoms.cell[2][2]
    all_elements=list(set(atoms.get_chemical_symbols()))
    cov_radii = [covalent_radii[a.number] for a in atoms]
    
    nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
    nl.update(atoms)
    
    #metalating_atoms=['Ag','Au','Cu','Co','Ni','Fe']
    #possible_slab_atoms=['Au','Ag','Cu','Pd','Ga','Ni']
    
    summary=''
    #sys_type='Strange'
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
        tipii=get_types(atoms,0.1)
        # 0=molecule
        # 1=surface atoms
        # 2=adatoms
        # 3=hydrogens on the surf
        sys_type='Slab'
        mol_atoms=np.where(tipii==0)[0].tolist()
        #mol_atoms=extract_mol_indexes_from_slab(atoms)
        all_molecules=molecules(mol_atoms,atoms)


        ## bottom_H  
        bottom_H=np.where(tipii==3)[0].tolist()
        #zmin=np.min(atoms.positions[:,2])
        #listh=[x[0] for x in np.argwhere(atoms.numbers == 1)]
        #for ih in listh:
        #    if atoms[ih].position[2]<zmin+0.8:
        #        bottom_H.append(ih)



        slabatoms=np.where(tipii==1)[0].tolist()
        adatoms=np.where(tipii==2)[0].tolist()
        
        ##slab layers
        nbins=np.max(atoms[slabatoms].positions[:,2]) - np.min(atoms[slabatoms].positions[:,2])
        nbins=int(np.ceil(nbins/0.15))
        hist, bin_edges = np.histogram(atoms[slabatoms].positions[:,2], density=True,bins=nbins)

        thr=np.max(hist)/4.0
        layers=bin_edges[np.where(hist>thr)[0]]
        slab_layers=[[]for i in range(len(layers))]
        for ia in slabatoms:
            idx = (np.abs(layers - atoms.positions[ia,2])).argmin()
            slab_layers[idx].append(ia)
        
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
            'system_type'   : sys_type,
            'the_cell'      : atoms.cell,
            'slab_layers'   : slab_layers,
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

