import ase
import ipywidgets as ipw
import numpy as np
import rdkit
import scipy
import sklearn.decomposition
import traitlets as tr
from ase import neighborlist


class CdxmlUpload2GnrWidget(ipw.VBox):
    """Class that allows to upload structures from user's computer."""

    structure = tr.Instance(ase.Atoms, allow_none=True)

    def __init__(self, title="CDXML to GNR", description="Upload Structure"):
        self.title = title
        self.mols = None
        self.original_structure = None
        self.selection = set()
        self.file_upload = ipw.FileUpload(
            description=description, multiple=False, layout={"width": "initial"}
        )
        supported_formats = ipw.HTML(
            """<a href="https://pubs.acs.org/doi/10.1021/ja0697875" target="_blank">
        Supported structure formats: ".cdxml"
        </a>"""
        )

        # self.file_upload.observe(self._on_file_upload, names="value")
        self.file_upload.observe(self._on_file_upload_rdkit_version, names="value")

        self.allmols = ipw.Dropdown(
            options=[None], description="Select mol", value=None, disabled=True
        )
        self.allmols.observe(self._on_sketch_selected, names="value")

        super().__init__(children=[self.file_upload, supported_formats, self.allmols])

    @staticmethod
    def guess_scaling_factor(atoms):
        """Scaling factor to correct the bond length."""

        # Set bounding box as cell.
        atoms.cell = np.ptp(atoms.positions, axis=0) + 15
        atoms.pbc = (True, True, True)

        # Calculate all atom-atom distances.
        c_atoms = [a for a in atoms if a.symbol[0] == "C"]
        n_atoms = len(c_atoms)
        dists = np.zeros([n_atoms, n_atoms])
        for i, atom_a in enumerate(c_atoms):
            for j, atom_b in enumerate(c_atoms):
                dists[i, j] = np.linalg.norm(atom_a.position - atom_b.position)

        # Find bond distances to closest neighbor.
        dists += np.diag([np.inf] * n_atoms)  # Don't consider diagonal.
        bonds = np.amin(dists, axis=1)

        # Average bond distance.
        avg_bond = float(scipy.stats.mode(bonds)[0])

        # Scale box to match equilibrium carbon-carbon bond distance.
        cc_eq = 1.4313333333
        return cc_eq / avg_bond

    @staticmethod
    def scale(atoms, s):
        """Scale atomic positions by the `factor`."""
        c_x, c_y, c_z = atoms.cell
        atoms.set_cell((s * c_x, s * c_y, c_z), scale_atoms=True)
        atoms.cell = np.ptp(atoms.positions, axis=0) + 15
        atoms.center()
        return atoms

    @staticmethod
    def rdkit2ase(mol):
        """Converts rdkit molecule into ase Atoms"""
        species = [
            ase.data.chemical_symbols[atm.GetAtomicNum()] for atm in mol.GetAtoms()
        ]
        pos = np.asarray(list(mol.GetConformer().GetPositions()))
        pca = sklearn.decomposition.PCA(n_components=3)
        posnew = pca.fit_transform(pos)
        atoms = ase.Atoms(species, positions=posnew)
        sys_size = np.ptp(atoms.positions, axis=0)
        atoms.rotate(-90, "z")  # cdxml are rotated
        atoms.pbc = True
        atoms.cell = sys_size + 10
        atoms.center()

        return atoms

    @staticmethod
    def add_h(atoms):
        """Add missing hydrogen atoms."""

        n_l = neighborlist.NeighborList(
            [ase.data.covalent_radii[a.number] for a in atoms],
            bothways=True,
            self_interaction=False,
        )
        n_l.update(atoms)

        need_hydrogen = []
        for atm in atoms:
            if len(n_l.get_neighbors(atm.index)[0]) < 3:
                if atm.symbol == "C" or atm.symbol == "N":
                    need_hydrogen.append(atm.index)

        print("Added missing Hydrogen atoms: ", need_hydrogen)

        for atm in need_hydrogen:
            vec = np.zeros(3)
            indices, offsets = n_l.get_neighbors(atoms[atm].index)
            for i, offset in zip(indices, offsets):
                vec += -atoms[atm].position + (
                    atoms.positions[i] + np.dot(offset, atoms.get_cell())
                )
            vec = -vec / np.linalg.norm(vec) * 1.1 + atoms[atm].position
            atoms.append(ase.Atom("H", vec))

        return atoms

    def _on_file_upload_rdkit_version(self, change=None):
        """When file upload button is pressed."""
        self.mols = {}
        listmols = []
        molid = 0
        for fname, _item in change["new"].items():
            frmt = fname.split(".")[-1]
            if frmt == "cdxml":
                cdxml_file_string = self.file_upload.value[fname]["content"].decode(
                    "ascii"
                )

                mol_supplier = rdkit.Chem.MolsFromCDXML(cdxml_file_string)

                for mol in mol_supplier:
                    atoms = self.rdkit2ase(mol)
                    formula = atoms.get_chemical_formula()
                    self.mols[molid] = atoms
                    listmols.append((str(molid) + ": " + formula, molid))
                    molid += 1

                self.allmols.options = listmols

                self.allmols.disabled = False

            break

    def _on_sketch_selected(self, change=None):
        self.structure = None  # needed to empty view in second viewer
        if self.mols is None or self.allmols.value is None:
            return
        atoms = self.mols[self.allmols.value]
        factor = self.guess_scaling_factor(atoms)
        atoms = self.scale(atoms, factor)
        atoms = self.add_h(atoms)
        self.structure = atoms
        self.file_upload.value.clear()
