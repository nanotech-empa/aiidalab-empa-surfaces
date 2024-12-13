import copy
import itertools
import aiidalab_widgets_base as awb

import numpy as np
from ase import Atoms, neighborlist
from scipy import sparse
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from .lowdimfinder import LowDimFinder
import traitlets as tr
def gaussian(x, sig):
    return (
        1.0
        / (sig * np.sqrt(2.0 * np.pi))
        * np.exp(-np.power(x, 2.0) / (2 * np.power(sig, 2.0)))
    )

def boxfilter(x, thr):
    return np.asarray([1 if i < thr else 0 for i in x])
    # Piero Gasparotto
def get_types( frame):  # classify the atmos in:
    # 0=molecule
    # 1=slab atoms
    # 2=adatoms
    # 3=hydrogens on the surf
    # 5=unknown
    # 6=metalating atoms
    # frame=ase frame
    # thr=threashold in the histogram for being considered a surface layer
    nat = frame.get_global_number_of_atoms()

    # all atom types set to 5 (unknown)
    atype = np.zeros(nat, dtype=np.int16) + 5
    area = frame.cell[0][0] * frame.cell[1][1]
    minz = np.min(frame.positions[:, 2])
    maxz = np.max(frame.positions[:, 2])

    if maxz - minz < 1.0:
        maxz += (1.0 - (maxz - minz)) / 2
        minz -= (1.0 - (maxz - minz)) / 2

    # Which values should we use below?
    sigma = 0.2  # thr
    peak_rel_height = 0.5
    layer_tol = 1.0 * sigma

    # Quick estimate number atoms in a layer:
    nbins = int(np.ceil((maxz - minz) / 0.15))
    hist, _ = np.histogram(frame.positions[:, 2], density=False, bins=nbins)
    max_atoms_in_a_layer = max(hist)

    lbls = frame.get_chemical_symbols()
    n_intervals = int(np.ceil((maxz - minz + 3 * sigma) / (0.1 * sigma)))
    z_values = np.linspace(minz - 3 * sigma, maxz + 3 * sigma, n_intervals)  # 1000
    atoms_z_pos = frame.positions[:, 2]

    # OPTION 1: generate 2d array to apply the gaussian on
    z_v_exp, at_z_exp = np.meshgrid(z_values, atoms_z_pos)
    arr_2d = z_v_exp - at_z_exp
    atomic_density = np.sum(gaussian(arr_2d, sigma), axis=0)

    # OPTION 2: loop through atoms
    # atomic_density = np.zeros(z_values.shape)
    # for ia in range(len(atoms)):
    #    atomic_density += gaussian(z_values - atoms.positions[ia,2], sigma)

    peaks = find_peaks(
        atomic_density,
        height=None,
        threshold=None,
        distance=None,
        prominence=None,
        width=None,
        wlen=None,
        rel_height=peak_rel_height,
    )
    layersg = z_values[peaks[0].tolist()]

    # Check top and bottom layers should be documented better

    found_top_surf = False
    while not found_top_surf:
        iz = layersg[-1]
        two_d_atoms = [
            frame.positions[i, 0:2]
            for i in range(nat)
            if np.abs(frame.positions[i, 2] - iz) < layer_tol
        ]
        coverage = 0
        if len(two_d_atoms) > max_atoms_in_a_layer / 4:
            hull = ConvexHull(two_d_atoms)
            coverage = hull.volume / area
        if coverage > 0.3:
            found_top_surf = True
        else:
            layersg = layersg[0:-1]

    found_bottom_surf = False
    while not found_bottom_surf:
        iz = layersg[0]
        two_d_atoms = [
            frame.positions[i, 0:2]
            for i in range(nat)
            if np.abs(frame.positions[i, 2] - iz) < layer_tol
        ]
        coverage = 0
        if len(two_d_atoms) > max_atoms_in_a_layer / 4:
            hull = ConvexHull(two_d_atoms)
            coverage = hull.volume / area
        if coverage > 0.3 and len(two_d_atoms) > max_atoms_in_a_layer / 4:
            found_bottom_surf = True
        else:
            layersg = layersg[1:]

    bottom_z = layersg[0]
    top_z = layersg[-1]

    # check if there is a bottom layer of H
    found_layer_of_h = True
    for i in range(nat):
        iz = frame.positions[i, 2]
        if (layer_tol + iz) > bottom_z and iz < (bottom_z + layer_tol):
            if lbls[i] == "H":
                atype[i] = 3
            else:
                found_layer_of_h = False
                break
    if found_layer_of_h:
        layersg = layersg[1:]
        # bottom_z=layersg[0]

    layers_dist = []
    iprev = layersg[0]
    for inext in layersg[1:]:
        layers_dist.append(abs(iprev - inext))
        iprev = inext

    for i in range(nat):
        iz = frame.positions[i, 2]
        if (layer_tol + iz) > bottom_z and iz < (top_z + layer_tol):
            if not (atype[i] == 3 and found_layer_of_h):
                atype[i] = 1
        else:
            if np.min([np.abs(iz - top_z), np.abs(iz - bottom_z)]) < np.max(
                layers_dist
            ):
                if not (atype[i] == 3 and found_layer_of_h):
                    atype[i] = 2

    # assign the other types
    #metalatingtypes = ("Au", "Ag", "Cu", "Ni", "Co", "Zn", "Mg")
    #moltypes = ("H", "N", "B", "O", "C", "F", "S", "Br", "I", "Cl")
    #possible_mol_atoms = [
    #    i for i in range(nat) if atype[i] == 2 and lbls[i] in moltypes
    #]
    #possible_mol_atoms += [i for i in range(nat) if atype[i] == 5]
    # identify separate molecules
    # all_molecules=self.molecules(mol_atoms,atoms)
    #all_molecules = []

    return atype, layersg
    

def transform_vector(pos1, pos2, v1):
    """
    Transform vector v1 using the transformation that maps pos2 to pos1.
    
    Args:
        pos1 (np.ndarray): Target positions, shape (n, 3).
        pos2 (np.ndarray): Source positions, shape (n, 3).
        v1 (np.ndarray): Vector to transform, shape (3,).
        
    Returns:
        np.ndarray: Transformed vector, shape (3,).
    """
    # Ensure inputs are numpy arrays
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    v1 = np.array(v1)

    # Compute centroids of both point sets
    centroid1 = np.mean(pos1, axis=0)
    centroid2 = np.mean(pos2, axis=0)

    # Center the points
    pos1_centered = pos1 - centroid1
    pos2_centered = pos2 - centroid2

    # Compute the covariance matrix
    H = np.dot(pos2_centered.T, pos1_centered)

    # Singular Value Decomposition (SVD) for rotation
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Translation vector
    t = centroid1 - np.dot(R, centroid2)

    # Transform the vector
    transformed_v1 = -np.dot(R, v1) #+ t
    transformed_v1 = transformed_v1 / np.linalg.norm(transformed_v1)
    

    return transformed_v1

class StructureAnalyzer(tr.HasTraits):
    structure = tr.Instance(Atoms, allow_none=True)
    details = tr.Dict()

    def __init__(self, only_sys_type=False, who_called_me="Boh"):
        self.only_sys_type = only_sys_type
        self.who_called_me = who_called_me
        super().__init__()

    @tr.observe("structure")
    def _observe_structure(self, _=None):
        with self.hold_trait_notifications():
            self.details = self.analyze()

    def analyze(
        self,
    ):
        if self.structure is None:
            return {}

        atoms = self.structure
        
 
        
        sys_size = np.ptp(atoms.positions, axis=0)
        no_cell = (
            atoms.cell[0][0] < 0.1 or atoms.cell[1][1] < 0.1 or atoms.cell[2][2] < 0.1
        )
        if no_cell:
            # set bounding box as cell
            atoms.cell = sys_size + 10

        atoms.set_pbc([True, True, True])
        
       # LowDimFinder section
        low_dim_finder = LowDimFinder(
        # This is called aiida_structure but it's wrong, it's an ASE structure
        #aiida_structure=conventional_asecell,
        aiida_structure=atoms,
        vacuum_space=40.0,
        radii_offset=-0.75, # [-0.75 -0.7, -0.65, -0.6, -0.55]
        bond_margin=0.0,
        max_supercell=3,
        min_supercell=3,
        rotation=True,
        full_periodicity=False,
        radii_source="alvarez",
        orthogonal_axis_2D=True,
    )
        res = low_dim_finder.get_group_data()
        # End LowDimFinder section
        is_a_bulk = np.amax(res['dimensionality']) == 3
        is_a_molecule = np.amax(res['dimensionality']) == 0
        is_a_wire = np.amax(res['dimensionality']) == 1
        is_a_slab = np.amax(res['dimensionality']) == 2
        max_dim_portion = res['dimensionality'].index(max(res['dimensionality']))
        # orientation
        if not is_a_bulk and not is_a_molecule:
            direction = transform_vector(atoms.positions[res['unit_cell_ids'][max_dim_portion]],np.array(res['positions'][max_dim_portion]),np.array([0,0,1]))
            dir_short = [f"{x:.2f}" for x in direction]
        #
        

        total_charge = np.sum(atoms.get_atomic_numbers())
        bottom_h = []
        adatoms = []
        bulkatoms = []
        wireatoms = []
        metalatings = []
        unclassified = []
        slabatoms = []
        slab_layers = []
        all_molecules = [res['unit_cell_ids'][i] for i,j in enumerate(res['dimensionality']) if j == 0] 

        spins_up = [the_a.index for the_a in atoms if the_a.tag == 1]
        spins_down = [the_a.index for the_a in atoms if the_a.tag == 2]
        other_tags = [the_a.index for the_a in atoms if the_a.tag > 2]



        # Do not use a set in the following line list(set(atoms.get_chemical_symbols()))
        # need ALL atoms and elements for spin guess and for cost calculation
        all_elements = atoms.get_chemical_symbols()
        summary = ""
        cases = []
        if len(spins_up) > 0:
            summary += "spins_up: " + awb.utils.llist_to_string_range(spins_up) + "\n"
        if len(spins_down) > 0:
            summary += "spins_down: " + awb.utils.llist_to_string_range(spins_down) + "\n"
        if len(other_tags) > 0:
            summary += "other_tags: " + awb.utils.llist_to_string_range(other_tags) + "\n"
            
        if is_a_bulk :
            
            sys_type = "Bulk"
            cases = ["b"]
            summary += "Bulk contains: \n"
            slabatoms = list(range(len(atoms)))
            bulkatoms = slabatoms

        if is_a_molecule :
            
            sys_type = "Molecule"
            summary += "Molecule: \n"
            #all_molecules = molecules(list(range(len(atoms))), atoms)
            com = np.average(atoms.positions, axis=0)
            summary += (
                "COM: "
                + str(com)
                + ", min z: "
                + str(np.min(atoms.positions[:, 2]))
                + "\n"
            )
        if is_a_wire :
            print("Wire")
            sys_type = "Wire"
            cases = ["w"]
            summary += f"Wire along {dir_short}  \n"
            slabatoms = list(range(len(atoms)))
            wireatoms = slabatoms
        # END check

        if is_a_slab:
            cases = ["s"]
            tipii, layersg = get_types(atoms)
            if np.abs(np.dot(direction, [1, 0, 0])) > 0.9:
                slabtype = "YZ"
            elif np.abs(np.dot(direction, [0, 1, 0])) > 0.9:
                slabtype = "XZ"
            else:
                slabtype = "XY"

            sys_type = "Slab" + slabtype
            mol_atoms = [idatom for mol in all_molecules for idatom in mol] #np.where(tipii == 0)[0].tolist()
            # mol_atoms=extract_mol_indexes_from_slab(atoms)
            #metalatings = np.where(tipii == 6)[0].tolist()
            #mol_atoms += metalatings

            bottom_h = np.where(tipii == 3)[0].tolist()
            slabatoms = np.where(tipii == 1)[0].tolist()
            adatoms = np.where(tipii == 2)[0].tolist()
            unclassified = [idatom for idatom in list(range(atoms.get_global_number_of_atoms())) if idatom not in  set(bottom_h+slabatoms+mol_atoms)]  # np.where(tipii == 5)[0].tolist()
            
            
            #MOVE this up to get_types
            slab_layers = [[] for i in range(len(layersg))]
            for ia in slabatoms:
                idx = (np.abs(layersg - atoms.positions[ia, 2])).argmin()
                slab_layers[idx].append(ia)

            # End slab layers
            summary += "Slab " + slabtype + " contains: \n"
        summary += (
            "Cell: " + " ".join([str(i) for i in atoms.cell.diagonal().tolist()]) + "\n"
        )
        if len(slabatoms) == 0:
            slab_elements = set()
        else:
            slab_elements = set(atoms[slabatoms].get_chemical_symbols())

        if len(bottom_h) > 0:
            summary += "bottom H: " + awb.utils.list_to_string_range(bottom_h) + "\n"
        if len(slabatoms) > 0:
            summary += "slab atoms: " + awb.utils.list_to_string_range(slabatoms) + "\n"
        for nlayer, the_layer in enumerate(slab_layers):
            summary += (
                "slab layer "
                + str(nlayer + 1)
                + ": "
                + awb.utils.list_to_string_range(the_layer)
                + "\n"
            )
        if len(adatoms) > 0:
            cases.append("a")
            summary += "adatoms: " + awb.utils.list_to_string_range(adatoms) + "\n"
        if all_molecules:
            cases.append("m")
            summary += "#" + str(len(all_molecules)) + " molecules: "
            for nmols, the_mol in enumerate(all_molecules):
                summary += str(nmols + 1) + ") " + awb.utils.list_to_string_range(the_mol)

        summary += " \n"
        if len(metalatings) > 0:
            metalating_str = awb.utils.list_to_string_range(metalatings)
            summary += (
                f"metal atoms inside molecules (already counted): {metalating_str}\n"
            )
        if len(unclassified) > 0:
            cases.append("u")
            summary += "unclassified: " + awb.utils.list_to_string_range(unclassified)

        # Indexes from 0 if mol_ids_range is not called.
        cell_str = " ".join([str(i) for i in itertools.chain(*atoms.cell.tolist())])

        return {
            "total_charge": total_charge,
            "system_type": sys_type,
            "cell": cell_str,
            "slab_layers": slab_layers,
            "bottom_H": sorted(bottom_h),
            "bulkatoms": sorted(bulkatoms),
            "wireatoms": sorted(wireatoms),
            "slabatoms": sorted(slabatoms),
            "adatoms": sorted(adatoms),
            "all_molecules": all_molecules,
            "metalatings": sorted(metalatings),
            "unclassified": sorted(unclassified),
            "numatoms": len(atoms),
            "all_elements": all_elements,
            "slab_elements": slab_elements,
            "spins_up": spins_up,
            "spins_down": spins_down,
            "other_tags": other_tags,
            "sys_size": sys_size,
            "cases": cases,
            "summary": summary,
        }
