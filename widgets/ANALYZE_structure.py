import itertools
from copy import deepcopy

import ase
import nglview
import numpy as np
import scipy.stats
from aiidalab_widgets_base.utils import list_to_string_range
from ase.data import covalent_radii
from ase.geometry.analysis import Analysis
from scipy import sparse
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull
from traitlets import Dict, HasTraits, Instance, observe


def to_ranges(iterable):
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]


def mol_ids_range(ismol):
    # shifts the list by +1
    range_string = ""
    shifted_list = [i + 1 for i in ismol]
    ranges = list(to_ranges(shifted_list))
    for i in range(len(ranges)):
        if ranges[i][1] > ranges[i][0]:
            range_string += f"{ranges[i][0]}..{ranges[i][1]}"
        else:
            range_string += f"{ranges[i][0]} "
    return range_string


def conne_matrix(atoms):
    cutoff = ase.neighborlist.natural_cutoffs(atoms)
    neighbor_list = ase.neighborlist.NeighborList(
        cutoff, self_interaction=False, bothways=False
    )
    neighbor_list.update(atoms)

    return neighbor_list.get_connectivity_matrix()


def clusters(matrix):
    nclusters, idlist = sparse.csgraph.connected_components(matrix)
    return nclusters, [np.where(idlist == i)[0].tolist() for i in range(nclusters)]


def molecules(ismol, atoms):
    if ismol:
        nmols, ids = clusters(conne_matrix(atoms[ismol]))
        return [[ismol[i] for i in ids[j]] for j in range(nmols)]
    return []


class StructureAnalyzer(HasTraits):
    structure = Instance(ase.Atoms, allow_none=True)
    details = Dict()

    def __init__(self, only_sys_type=False, who_called_me="Boh"):
        self.only_sys_type = only_sys_type
        self.who_called_me = who_called_me
        super().__init__()

    def gaussian(self, x, sig):
        return (
            1.0
            / (sig * np.sqrt(2.0 * np.pi))
            * np.exp(-np.power(x, 2.0) / (2 * np.power(sig, 2.0)))
        )

    def boxfilter(self, x, thr):
        return np.asarray([1 if i < thr else 0 for i in x])

    def get_types(self, frame, thr):  # Piero Gasparotto
        # classify the atmos in:
        # 0=molecule
        # 1=slab atoms
        # 2=adatoms
        # 3=hydrogens on the surf
        # 5=unknown
        # 6=metalating atoms
        # frame=ase frame
        # thr=threashold in the histogram for being considered a surface layer
        nat = frame.get_number_of_atoms()

        # all atom types set to 5 (unknown)
        atype = np.zeros(nat, dtype=np.int16) + 5
        area = frame.cell[0][0] * frame.cell[1][1]
        minz = np.min(frame.positions[:, 2])
        maxz = np.max(frame.positions[:, 2])

        if maxz - minz < 1.0:
            maxz += (1.0 - (maxz - minz)) / 2
            minz -= (1.0 - (maxz - minz)) / 2

        # Which values should we use?
        sigma = 0.2
        peak_rel_height = 0.5
        layer_tol = 1.0 * sigma

        # Quick estimate number atoms in a layer.
        nbins = int(np.ceil((maxz - minz) / 0.15))
        hist, bin_edges = np.histogram(frame.positions[:, 2], density=False, bins=nbins)
        max_atoms_in_a_layer = max(hist)

        lbls = frame.get_chemical_symbols()
        n_intervals = int(np.ceil((maxz - minz + 3 * sigma) / (0.1 * sigma)))
        z_values = np.linspace(minz - 3 * sigma, maxz + 3 * sigma, n_intervals)  # 1000
        atoms_z_pos = frame.positions[:, 2]

        # Generate 2d array to apply the gaussian on.
        z_v_exp, at_z_exp = np.meshgrid(z_values, atoms_z_pos)
        arr_2d = z_v_exp - at_z_exp
        atomic_density = np.sum(self.gaussian(arr_2d, sigma), axis=0)

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
        len(layersg)
        layersg[-1]

        # Check top and bottom layers should be documented better.
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

        # Check if there is a bottom layer of H.
        found_layer_of_h = True
        for i in range(nat):
            iz = frame.positions[i, 2]
            if iz > bottom_z - layer_tol and iz < bottom_z + layer_tol:
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
            if iz > bottom_z - layer_tol and iz < top_z + layer_tol:
                if not (atype[i] == 3 and found_layer_of_h):
                    atype[i] = 1
            else:
                if np.min([np.abs(iz - top_z), np.abs(iz - bottom_z)]) < np.max(
                    layers_dist
                ):
                    if not (atype[i] == 3 and found_layer_of_h):
                        atype[i] = 2

        # assign the other types
        metalatingtypes = ("Au", "Ag", "Cu", "Ni", "Co", "Zn", "Mg")
        moltypes = ("H", "N", "B", "O", "C", "F", "S", "Br", "I", "Cl")
        possible_mol_atoms = [
            i for i in range(nat) if atype[i] == 2 and lbls[i] in moltypes
        ]
        possible_mol_atoms += [i for i in range(nat) if atype[i] == 5]
        # identify separate molecules
        # all_molecules=self.molecules(mol_atoms,atoms)
        all_molecules = []
        if len(possible_mol_atoms) > 0:
            # conne = conne_matrix(frame[possible_mol_atoms])
            fragments = molecules(possible_mol_atoms, frame)
            all_molecules = deepcopy(fragments)
            # remove isolated atoms
            for frag in fragments:
                if len(frag) == 1:
                    all_molecules.remove(frag)
                else:
                    for atom in frag:
                        if lbls[atom] in metalatingtypes:
                            atype[atom] = 6
                        else:
                            atype[atom] = 0

        return atype, layersg, all_molecules

    def all_connected_to(self, id_atom, atoms, exclude):
        cov_radii = [covalent_radii[a.number] for a in atoms]

        atoms.set_pbc([False, False, False])
        nl_no_pbc = ase.neighborlist.NeighborList(
            cov_radii, bothways=True, self_interaction=False
        )
        nl_no_pbc.update(atoms)
        atoms.set_pbc([True, True, True])

        tofollow = []
        followed = []
        isconnected = []
        tofollow.append(id_atom)
        isconnected.append(id_atom)
        while len(tofollow) > 0:
            indices, offsets = nl_no_pbc.get_neighbors(tofollow[0])
            indices = list(indices)
            followed.append(tofollow[0])
            for i in indices:
                if (i not in isconnected) and (atoms[i].symbol not in exclude):
                    tofollow.append(i)
                    isconnected.append(i)
            for i in followed:
                if i in tofollow:  # do not remove this check
                    tofollow.remove(i)

        return isconnected

    def string_range_to_list(self, a):
        singles = [int(s) - 1 for s in a.split() if s.isdigit()]
        ranges = [r for r in a.split() if ".." in r]
        for r in ranges:
            t = r.split("..")
            to_add = [i - 1 for i in range(int(t[0]), int(t[1]) + 1)]
            singles += to_add
        return sorted(singles)

    @observe("structure")
    def _observe_structure(self, _=None):
        with self.hold_trait_notifications():
            self.details = self.analyze()

    def analyze(self):
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

        total_charge = np.sum(atoms.get_atomic_numbers())
        bottom_h = []
        adatoms = []
        bulkatoms = []
        wireatoms = []
        metalatings = []
        unclassified = []
        slabatoms = []
        slab_layers = []
        all_molecules = None
        is_a_bulk = False
        is_a_molecule = False
        is_a_wire = False

        spins_up = [the_a.index for the_a in atoms if the_a.tag == 1]
        spins_down = [the_a.index for the_a in atoms if the_a.tag == 2]
        other_tags = [the_a.index for the_a in atoms if the_a.tag > 2]

        # Check if there is vacuum otherwise classify as bulk and skip.
        vacuum_x = sys_size[0] + 4 < atoms.cell[0][0]
        vacuum_y = sys_size[1] + 4 < atoms.cell[1][1]
        vacuum_z = sys_size[2] + 4 < atoms.cell[2][2]

        # Do not use a set in the following line list(set(atoms.get_chemical_symbols()))
        # Need ALL atoms and elements for spin guess and for cost calculation
        all_elements = atoms.get_chemical_symbols()

        summary = ""
        cases = []
        if len(spins_up) > 0:
            summary += "spins_up: " + list_to_string_range(spins_up) + "\n"
        if len(spins_down) > 0:
            summary += "spins_down: " + list_to_string_range(spins_down) + "\n"
        if len(other_tags) > 0:
            summary += "other_tags: " + list_to_string_range(other_tags) + "\n"
        if (not vacuum_z) and (not vacuum_x) and (not vacuum_y):
            is_a_bulk = True
            sys_type = "Bulk"
            cases = ["b"]
            summary += "Bulk contains: \n"
            slabatoms = list(range(len(atoms)))
            bulkatoms = slabatoms

        if vacuum_x and vacuum_y and vacuum_z:
            is_a_molecule = True
            sys_type = "Molecule"
            if not self.only_sys_type:
                summary += "Molecule: \n"
                all_molecules = molecules(list(range(len(atoms))), atoms)
                com = np.average(atoms.positions, axis=0)
                summary += (
                    "COM: "
                    + str(com)
                    + ", min z: "
                    + str(np.min(atoms.positions[:, 2]))
                    + "\n"
                )
        if vacuum_x and vacuum_y and (not vacuum_z):
            is_a_wire = True
            sys_type = "Wire"
            cases = ["w"]
            if not self.only_sys_type:
                summary += "Wire along z contains: \n"
                slabatoms = list(range(len(atoms)))
        if vacuum_y and vacuum_z and (not vacuum_x):
            is_a_wire = True
            sys_type = "Wire"
            cases = ["w"]
            if not self.only_sys_type:
                summary += "Wire along x contains: \n"
                slabatoms = list(range(len(atoms)))
        if vacuum_x and vacuum_z and (not vacuum_y):
            is_a_wire = True
            sys_type = "Wire"
            cases = ["w"]
            if not self.only_sys_type:
                summary += "Wire along y contains: \n"
                slabatoms = list(range(len(atoms)))
                wireatoms = slabatoms
        is_a_slab = not (is_a_bulk or is_a_molecule or is_a_wire)
        if self.only_sys_type:
            if is_a_slab:
                return {"system_type": "Slab"}
            else:
                return {"system_type": sys_type}

        elif is_a_slab:
            cases = ["s"]
            tipii, layersg, all_molecules = self.get_types(atoms, 0.1)
            if vacuum_x:
                slabtype = "YZ"
            elif vacuum_y:
                slabtype = "XZ"
            else:
                slabtype = "XY"

            sys_type = "Slab" + slabtype
            mol_atoms = np.where(tipii == 0)[0].tolist()
            metalatings = np.where(tipii == 6)[0].tolist()
            mol_atoms += metalatings

            bottom_h = np.where(tipii == 3)[0].tolist()

            # Unclassified
            unclassified = np.where(tipii == 5)[0].tolist()

            slabatoms = np.where(tipii == 1)[0].tolist()
            adatoms = np.where(tipii == 2)[0].tolist()

            # Slab layers.
            slab_layers = [[] for i in range(len(layersg))]
            for ia in slabatoms:
                idx = (np.abs(layersg - atoms.positions[ia, 2])).argmin()
                slab_layers[idx].append(ia)

            summary += "Slab " + slabtype + " contains: \n"
        summary += (
            "Cell: " + " ".join([str(i) for i in atoms.cell.diagonal().tolist()]) + "\n"
        )
        if len(slabatoms) == 0:
            slab_elements = set()
        else:
            slab_elements = set(atoms[slabatoms].get_chemical_symbols())

        if len(bottom_h) > 0:
            summary += "bottom H: " + mol_ids_range(bottom_h) + "\n"
        if len(slabatoms) > 0:
            summary += "slab atoms: " + mol_ids_range(slabatoms) + "\n"
        for nlayer in range(len(slab_layers)):
            summary += (
                "slab layer "
                + str(nlayer + 1)
                + ": "
                + mol_ids_range(slab_layers[nlayer])
                + "\n"
            )
        if len(adatoms) > 0:
            cases.append("a")
            summary += "adatoms: " + mol_ids_range(adatoms) + "\n"
        if all_molecules:
            cases.append("m")
            summary += "#" + str(len(all_molecules)) + " molecules: "
            for nmols in range(len(all_molecules)):
                summary += str(nmols + 1) + ") " + mol_ids_range(all_molecules[nmols])

        summary += " \n"
        if len(metalatings) > 0:
            summary += (
                "metal atoms inside molecules (already counted): "
                + mol_ids_range(metalatings)
                + "\n"
            )
        if len(unclassified) > 0:
            cases.append("u")
            summary += "unclassified: " + mol_ids_range(unclassified)

        # Indexes from 0 if mol_ids_range is not called.

        return {
            "total_charge": total_charge,
            "system_type": sys_type,
            "cell": " ".join([str(i) for i in itertools.chain(*atoms.cell.tolist())]),
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
