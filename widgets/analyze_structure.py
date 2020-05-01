import numpy as np
from numpy.linalg import norm
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
import ase.neighborlist
import scipy.stats
from scipy.constants import physical_constants
import itertools
from IPython.display import display, clear_output, HTML
import nglview
import ipywidgets as ipw
from collections import Counter
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
    
def gaussian(x, sig):
    return 1.0/(sig*np.sqrt(2.0*np.pi))*np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

def boxfilter(x,thr):
    return np.asarray([1 if i<thr else 0 for i in x])
def get_types(frame,thr): ## Piero Gasparotto
    # classify the atmos in:
    # 0=molecule
    # 1=slab atoms
    # 2=adatoms
    # 3=hydrogens on the surf
    # 5=unknown
    # 6=metalating atoms
    #frame=ase frame
    #thr=threashold in the histogram for being considered a surface layer
    nat=frame.get_number_of_atoms()
    atype=np.zeros(nat,dtype=np.int16)+5
    area=(frame.cell[0][0]*frame.cell[1][1])
    minz=np.min(frame.positions[:,2])
    maxz=np.max(frame.positions[:,2])
    
    if maxz - minz < 1.0:
        maxz += (1.0 - (maxz - minz))/2
        minz -= (1.0 - (maxz - minz))/2
    
    ##WHICH VALUES SHOULD WE USE BELOW??????
    sigma = 0.2 #thr
    peak_rel_height = 0.5
    layer_tol=1.0*sigma 
    # quack estimate number atoms in a layer:
    nbins=int(np.ceil((maxz-minz)/0.15))
    hist, bin_edges = np.histogram(frame.positions[:,2], density=False,bins=nbins)
    max_atoms_in_a_layer=max(hist)
    
    lbls=frame.get_chemical_symbols()
    n_intervals=int(np.ceil((maxz-minz+3*sigma)/(0.1*sigma)))
    z_values = np.linspace(minz-3*sigma, maxz+3*sigma, n_intervals) #1000
    atoms_z_pos = frame.positions[:,2]
    
    # OPTION 1: generate 2d array to apply the gaussian on
    z_v_exp, at_z_exp = np.meshgrid(z_values, atoms_z_pos)
    arr_2d = z_v_exp - at_z_exp
    atomic_density = np.sum(gaussian(arr_2d, sigma), axis=0)
    
    # OPTION 2: loop through atoms
    # atomic_density = np.zeros(z_values.shape)
    #for ia in range(len(atoms)):
    #    atomic_density += gaussian(z_values - atoms.positions[ia,2], sigma) 

    peaks=find_peaks(atomic_density, height=None,threshold=None,distance=None,
                     prominence=None,width=None,wlen=None,rel_height=peak_rel_height)
    layersg=z_values[peaks[0].tolist()]
    n_tot_layers=len(layersg)
    last_layer=layersg[-1]

    ##check top and bottom layers
    
    found_top_surf = False
    while not found_top_surf:
        iz = layersg[-1]
        twoD_atoms = [frame.positions[i,0:2] for i in range(nat) if np.abs(frame.positions[i,2]-iz) <layer_tol ]
        coverage=0
        if len(twoD_atoms) > max_atoms_in_a_layer/4:
            hull = ConvexHull(twoD_atoms) ##  
            coverage = hull.volume/area
        if coverage > 0.3:
            found_top_surf=True
        else:
            layersg=layersg[0:-1]
                    
    found_bottom_surf = False
    while not found_bottom_surf:
        iz = layersg[0]
        twoD_atoms = [frame.positions[i,0:2] for i in range(nat) if np.abs(frame.positions[i,2]-iz) <layer_tol]
        coverage=0
        if len(twoD_atoms) > max_atoms_in_a_layer/4:        
            hull = ConvexHull(twoD_atoms) ## 
            coverage = hull.volume/area
        if coverage > 0.3 and len(twoD_atoms) > max_atoms_in_a_layer/4 :
            found_bottom_surf=True
        else:
            layersg=layersg[1:]   
    
    bottom_z = layersg[0]
    top_z = layersg[-1]
    
    #check if there is a bottom layer of H
    found_layer_of_H=True
    for i in range(nat):
        iz = frame.positions[i,2]
        if iz > bottom_z - layer_tol and iz < bottom_z + layer_tol:
            if lbls[i]=='H':
                atype[i]=3
            else:
                found_layer_of_H=False
                break
    if found_layer_of_H:
        layersg=layersg[1:]
        #bottom_z=layersg[0]
        
    layers_dist = []
    iprev = layersg[0]
    for inext in layersg[1:]:
        layers_dist.append(abs(iprev - inext))
        iprev = inext 
    
    for i in range(nat):
        iz = frame.positions[i,2]
        if iz > bottom_z - layer_tol and iz < top_z + layer_tol:
            if not (atype[i]==3 and found_layer_of_H):
                atype[i]=1
        else:
            if np.min([np.abs(iz- top_z),np.abs(iz- bottom_z)]) < np.max(layers_dist):   
                if not (atype[i]==3 and found_layer_of_H): 
                    atype[i]=2
    
    # assign the other types
    metalatingtypes=('Au','Ag','Cu','Ni','Co','Zn','Mg')
    moltypes=('H','N','B','O','C','F','S','Br','I','Cl')
    possible_mol_atoms=[i for i in range(nat) if atype[i]==2 and lbls[i] in moltypes]
    possible_mol_atoms+=[i for i in range(nat) if atype[i]==5]
    
    if len(possible_mol_atoms) > 0:
        cov_radii = [covalent_radii[a.number] for a in frame[possible_mol_atoms]]  #adatoms that have a neigh adatom are in a mol
        nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
        nl.update(frame[possible_mol_atoms])
        for ia in range(len(possible_mol_atoms)):
            indices, offsets = nl.get_neighbors(ia)

            if len(indices) > 0:
                if lbls[possible_mol_atoms[ia]] in metalatingtypes:
                    atype[possible_mol_atoms[ia]]=6
                else:
                    atype[possible_mol_atoms[ia]]=0
    return atype,layersg

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

def string_range_to_list(a):
    singles=[int(s) -1 for s in a.split() if s.isdigit()]
    ranges = [r for r in a.split() if '..' in r]
    for r in ranges:
        t=r.split('..')
        to_add=[i-1 for i in range(int(t[0]),int(t[1])+1)]
        singles+=to_add
    return sorted(singles)

def analyze(atoms):
    no_cell=atoms.cell[0][0] <0.1 or atoms.cell[1][1] <0.1 or atoms.cell[2][2] <0.1 
    if no_cell:
        # set bounding box as cell
        cx =(np.amax(atoms.positions[:,0]) - np.amin(atoms.positions[:,0])) + 10
        cy =(np.amax(atoms.positions[:,1]) - np.amin(atoms.positions[:,1])) + 10
        cz =(np.amax(atoms.positions[:,2]) - np.amin(atoms.positions[:,2])) + 10
        atoms.cell = (cx, cy, cz)
    
    atoms.set_pbc([True,True,True])

    total_charge=np.sum(atoms.get_atomic_numbers())
    bottom_H=[]
    adatoms=[]
    remaining=[]
    metalatings=[]
    unclassified=[]
    slabatoms=[]
    slab_layers=[]    
    all_molecules=None
    is_a_bulk=False
    is_a_molecule=False
    is_a_wire=False

    spins_up   = set(str(the_a.symbol)+str(the_a.tag) for the_a in atoms if the_a.tag == 1)
    spins_down = set(str(the_a.symbol)+str(the_a.tag) for the_a in atoms if the_a.tag == 2)
    #### check if there is vacuum otherwise classify as bulk and skip
    
    vacuum_x=np.max(atoms.positions[:,0]) - np.min(atoms.positions[:,0]) +4 < atoms.cell[0][0]
    vacuum_y=np.max(atoms.positions[:,1]) - np.min(atoms.positions[:,1]) +4 < atoms.cell[1][1]
    vacuum_z=np.max(atoms.positions[:,2]) - np.min(atoms.positions[:,2]) +4 < atoms.cell[2][2]
    all_elements= atoms.get_chemical_symbols() # list(set(atoms.get_chemical_symbols()))
    cov_radii = [covalent_radii[a.number] for a in atoms]
    
    nl = NeighborList(cov_radii, bothways = True, self_interaction = False)
    nl.update(atoms)
    
    #metalating_atoms=['Ag','Au','Cu','Co','Ni','Fe']
    
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
    if vacuum_x and vacuum_y and (not vacuum_z):
        is_a_wire=True
        sys_type='Wire'
        summary='Wire along z contains: \n'
        slabatoms=[ia for ia in range(len(atoms))]   
    if vacuum_y and vacuum_z and (not vacuum_x):
        is_a_wire=True
        sys_type='Wire'
        summary='Wire along x contains: \n'
        slabatoms=[ia for ia in range(len(atoms))]
    if vacuum_x and vacuum_z and (not vacuum_y):
        is_a_wire=True
        sys_type='Wire'
        summary='Wire along y contains: \n'
        slabatoms=[ia for ia in range(len(atoms))]        
    ####END check
    if not (is_a_bulk or is_a_molecule or is_a_wire):
        tipii,layersg=get_types(atoms,0.1)
        if vacuum_x:
            slabtype='YZ'
        elif vacuum_y:
            slabtype='XZ'
        else:
            slabtype='XY'

        sys_type='Slab' + slabtype
        mol_atoms=np.where(tipii==0)[0].tolist()
        #mol_atoms=extract_mol_indexes_from_slab(atoms)
        metalatings=np.where(tipii==6)[0].tolist()
        mol_atoms+=metalatings
        
        #identify separate molecules
        all_molecules=molecules(mol_atoms,atoms)


        ## bottom_H  
        bottom_H=np.where(tipii==3)[0].tolist()
        
        ## unclassified  
        unclassified=np.where(tipii==5)[0].tolist()        



        slabatoms=np.where(tipii==1)[0].tolist()
        adatoms=np.where(tipii==2)[0].tolist()
        
        ##slab layers
        slab_layers=[[]for i in range(len(layersg))]
        for ia in slabatoms:
            idx = (np.abs(layersg - atoms.positions[ia,2])).argmin()
            slab_layers[idx].append(ia)
        
        ##end slab layers
        summary='Slab '+slabtype+' contains: \n'
       
    if len(slabatoms) == 0:
        slab_elements = set([])
    else:
        slab_elements=set(atoms[slabatoms].get_chemical_symbols())

    if len(bottom_H) >0:
        summary+='bottom H: ' + mol_ids_range(bottom_H)   + '\n'
    if len(slabatoms) > 0:    
        summary+='slab atoms: '   + mol_ids_range(slabatoms)  + '\n' 
    for nlayer in range(len(slab_layers)):
        summary+='slab layer '+str(nlayer+1)+': '+mol_ids_range(slab_layers[nlayer])+'\n'    
    if len(adatoms)>0:
        
        summary+='adatoms: '  + mol_ids_range(adatoms)    + '\n'  
    if all_molecules:
        
        summary+='#'+str(len(all_molecules))   + ' molecules: '
        for nmols in range(len(all_molecules)):
            summary+=str(nmols)+') '+mol_ids_range(all_molecules[nmols])
        
    summary+=' \n' 
    if len(mol_ids_range(metalatings))>0:
        summary+='metal atoms inside molecules (already counted): '+ mol_ids_range(metalatings) + '\n'
    if len(mol_ids_range(unclassified))>0:
        summary+='unclassified: ' + mol_ids_range(unclassified)

    return {'total_charge'  : total_charge,
            'system_type'   : sys_type,
            'cell'          : " ".join([str(i) for i in itertools.chain(*atoms.cell.tolist())]),
            'slab_layers'   : slab_layers,
            'bottom_H'      : sorted(bottom_H),
            'slabatoms'     : sorted(slabatoms),
            'adatoms'       : sorted(adatoms),
            'all_molecules' : all_molecules,
            'metalatings'   : sorted(metalatings),
            'unclassified'  : sorted(unclassified),
            'numatoms'      : len(atoms),
            'all_elements'  : all_elements,
            'slab_elements' : slab_elements,
            'spins_up'      : spins_up,
            'spins_down'    : spins_down,
            'summary':summary
           }