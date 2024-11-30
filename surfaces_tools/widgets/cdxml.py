import xml.etree.ElementTree as ET
import numpy as np
import rdkit
import scipy
import sklearn.decomposition
import traitlets as tr
import ase
import ipywidgets as ipw

class CdxmlUpload2GnrWidget(ipw.VBox):
    """Class that allows to upload structures from user's computer."""

    structure = tr.Instance(ase.Atoms, allow_none=True)

    def __init__(self, title="CDXML to GNR", description="Upload Structure"):
        self.title = title
        self.file_upload = ipw.FileUpload(
            description=description,
            multiple=False,
            layout={"width": "initial"},
            accept=".cdxml",
        )
        supported_formats = ipw.HTML(
            """<a href="https://pubs.acs.org/doi/10.1021/ja0697875" target="_blank">
        Supported structure formats: ".cdxml"
        </a>"""
        )

        self.file_upload.observe(self._on_file_upload, names="value")
        self.nunits = ipw.Text(description="N units",value='Infinite',disabled=True)
        self.create = ipw.Button(description="Create model",button_style="success",)
        self.create.on_click(self._on_button_click)

        #self._structure_selector = ipw.Dropdown(
        #    options=[None], description="Select mol", value=None, disabled=True
        #)
        #self._structure_selector.observe(self._on_sketch_selected, names="value")
        self.output_message = ipw.HTML(value="")

        super().__init__(
            children=[
                self.file_upload,
                self.nunits,
                supported_formats,
                self.create,
                self.output_message,
            ]
        )

    @staticmethod
    def guess_scaling_factor(atoms):
        """Scaling factor to correct the bond length."""

        # Set bounding box as cell.
        #atoms.cell = np.ptp(atoms.positions, axis=0) + 15
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
        #atoms.cell = np.ptp(atoms.positions, axis=0) + 15
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
        message = ""

        n_l = ase.neighborlist.NeighborList(
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

        message = f"Added missing Hydrogen atoms: {need_hydrogen}."

        for atm in need_hydrogen:
            vec = np.zeros(3)
            indices, offsets = n_l.get_neighbors(atoms[atm].index)
            for i, offset in zip(indices, offsets):
                vec += -atoms[atm].position + (
                    atoms.positions[i] + np.dot(offset, atoms.get_cell())
                )
            vec = -vec / np.linalg.norm(vec) * 1.1 + atoms[atm].position
            atoms.append(ase.Atom("H", vec))

        return message, atoms
    
    @staticmethod
    def transform_points(set1, set2, points):
        """
        Transform a set of points based on the scaling and rotation that aligns set1 to set2.

        Args:
            set1 (list of tuples): Reference set of points (e.g., [(x1, y1), ...]).
            set2 (list of tuples): Transformed set of points (e.g., [(x2, y2), ...]).
            points (list of tuples): Points to transform (e.g., [(px1, py1), ...]).

        Returns:
            list of tuples: Transformed points.
        """
        #set1 = np.array(set1)
        #set2 = np.array(set2)
        #points = np.array(points)

        # Compute centroids of set1 and set2
        centroid1 = np.mean(set1, axis=0)
        centroid2 = np.mean(set2, axis=0)

        # Center the sets around their centroids
        centered_set1 = set1 - centroid1
        centered_set2 = set2 - centroid2

        # Compute the scaling factor
        norm1 = np.linalg.norm(centered_set1, axis=1).mean()
        norm2 = np.linalg.norm(centered_set2, axis=1).mean()
        scale = norm2 / norm1

        # Compute the rotation matrix using Singular Value Decomposition (SVD)
        H = np.dot(centered_set1.T, centered_set2)  # Cross-covariance matrix
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)  # Rotation matrix

        # Apply the scaling and rotation to the points
        transformed_points = scale * np.dot(points - centroid1, R.T) + centroid2

        return transformed_points.tolist()
    
    @staticmethod
    def NOextract_crossing_and_atom_positions(cdxml_content):
        """
        Extract atom positions and the first two crossing points derived from square parentheses.
        The unit vector is the normalized vector connecting the midpoints of the square parentheses bounding boxes,
        with positive x and y.

        Args:
            cdxml_content (str): The content of the CDXML file as a string.

        Returns:
            tuple: Three values:
                - crossing_points_pair: A 2x3 NumPy array containing the midpoints of the two square parentheses.
                - atom_positions: Atom positions as a numpy array of shape (M, 3).
                - unit_vector: A unit vector derived from the translation vector between the parentheses.
        """
        root = ET.fromstring(cdxml_content)

        # Parse all atom positions
        atom_positions = []
        atom_id_map = {}
        for node in root.findall(".//n"):
            atom_id = node.get("id")
            if atom_id and "p" in node.attrib:
                position = tuple(map(float, node.attrib["p"].split()))
                atom_positions.append((position[0], position[1], 0.0))  # Add z=0
                atom_id_map[atom_id] = len(atom_positions) - 1  # Map atom ID to index

        atom_positions = np.array(atom_positions)

        # Parse square parentheses
        brackets = []
        for graphic in root.findall(".//graphic[@BracketType='Square']"):
            if "BoundingBox" in graphic.attrib:
                bb = list(map(float, graphic.attrib["BoundingBox"].split()))
                x_min, y_min, x_max, y_max = bb

                # Compute the midpoint of the bounding box
                midpoint = ((x_min + x_max) / 2, (y_min + y_max) / 2, 0.0)
                brackets.append(midpoint)

        if len(brackets) != 2:
            raise ValueError("Expected exactly 2 square parentheses, found: {}".format(len(brackets)))

        # Calculate the unit vector from the two brackets
        brackets = np.array(brackets)
        vector = brackets[1] - brackets[0]
        unit_vector = vector[:2] / np.linalg.norm(vector[:2])

        # Ensure positive x and y for the unit vector
        if unit_vector[0] < 0 or unit_vector[1] < 0:
            unit_vector = -unit_vector

        return brackets, atom_positions, unit_vector
    
    @staticmethod
    def extract_crossing_and_atom_positions(cdxml_content):
        """
        Extract the first two crossing points such that the vector connecting them is aligned with the unit vector.

        Args:
            cdxml_content (str): The content of the CDXML file as a string.

        Returns:
            tuple: Three values:
                - crossing_points_pair: A tuple of two crossing points that are aligned with the unit vector.
                  The unit_vector is  vector with positive x and y, parallel to the vector connecting the square brackets
                - atom_positions: Atom positions as a numpy array of shape (M, 3).
        """
        root = ET.fromstring(cdxml_content)

        # Parse all atom positions
        atom_positions = []
        atom_id_map = {}
        for node in root.findall(".//n"):
            atom_id = node.get("id")
            if atom_id and "p" in node.attrib:
                position = tuple(map(float, node.attrib["p"].split()))
                atom_positions.append((position[0], position[1], 0.0))  # Add z=0
                atom_id_map[atom_id] = len(atom_positions) - 1  # Map atom ID to index

        atom_positions = np.array(atom_positions)

        # Parse crossing bonds and compute crossing points
        crossing_points = []
        unit_vector = None

        for crossing in root.findall(".//crossingbond"):
            bond_id = crossing.get("BondID")

            if bond_id:
                # Find the bond's start and end atom positions
                bond = root.find(f".//b[@id='{bond_id}']")
                if bond is not None:
                    start_id = bond.get("B")
                    end_id = bond.get("E")
                    if start_id in atom_id_map and end_id in atom_id_map:
                        start_pos = atom_positions[atom_id_map[start_id]]
                        end_pos = atom_positions[atom_id_map[end_id]]

                        # Compute the midpoint of the bond
                        midpoint = (
                            (start_pos[0] + end_pos[0]) / 2,
                            (start_pos[1] + end_pos[1]) / 2,
                            0.0,  # Add z=0
                        )
                        crossing_points.append(midpoint)

                        ## Compute the unit vector for the first bond
                        #if unit_vector is None:
                        #    vector = np.array(end_pos) - np.array(start_pos)
                        #    unit_vector = vector[:2] / np.linalg.norm(vector[:2])
                        #    # Ensure positive x and y
                        #    if unit_vector[0] < 0 or unit_vector[1] < 0:
                        #        unit_vector = -unit_vector

        crossing_points = np.array(crossing_points)

        # Parse square parentheses
        brackets = []
        for graphic in root.findall(".//graphic[@BracketType='Square']"):
            if "BoundingBox" in graphic.attrib:
                bb = list(map(float, graphic.attrib["BoundingBox"].split()))
                x_min, y_min, x_max, y_max = bb

                # Compute the midpoint of the bounding box
                midpoint = ((x_min + x_max) / 2, (y_min + y_max) / 2, 0.0)
                brackets.append(midpoint)

        if len(brackets) != 2 and len(brackets) != 0:
            #raise ValueError("Expected exactly 2 square parentheses, found: {}".format(len(brackets)))
            print(f"found {len(brackets)} square parentheses")

        if len(brackets) == 2:
            # Calculate the unit vector from the two brackets
            brackets = np.array(brackets)
            vector = brackets[1] - brackets[0]
            unit_vector = vector[:2] / np.linalg.norm(vector[:2])

            # Ensure positive x and y for the unit vector
            if unit_vector[0] < 0 or unit_vector[1] < 0:
                unit_vector = -unit_vector


            # Find the first two crossing points aligned with the unit vector
            for i in range(len(crossing_points)):
                for j in range(i + 1, len(crossing_points)):
                    vector = crossing_points[j][:2] - crossing_points[i][:2]
                    unit_test_vector = vector / np.linalg.norm(vector)
                    # Check alignment (dot product close to 1)
                    if np.dot(unit_test_vector, unit_vector) > 0.99:
                        isnotperiodic = False
                        return crossing_points[[i,j]], atom_positions,isnotperiodic

        return None, atom_positions, True
 
    
    @staticmethod
    def align_and_trim_atoms(atoms, crossing_points, units=None):
        """
        Aligns an ASE Atoms object with the x-axis based on two crossing points,
        trims or replicates atoms based on specified x-bounds, sets a new unit cell, and centers the structure.

        Args:
            atoms (ASE.Atoms): The ASE Atoms object to transform.
            crossing_points (numpy.ndarray): A 2x3 NumPy array containing two crossing points.
            Nunits (int, optional): Number of units to replicate along the x-axis. If None, trims atoms.

        Returns:
            ASE.Atoms: The transformed ASE Atoms object.
        """
        # Ensure crossing_points is a 2x3 array
        assert crossing_points.shape == (2, 3), "crossing_points must be a 2x3 NumPy array."

        # Calculate the vector connecting the two crossing points
        vector = crossing_points[1] - crossing_points[0]
        norm_vector = np.linalg.norm(vector)

        # Calculate rotation angle to align the vector with the x-axis
        angle = np.arctan2(vector[1], vector[0])

        # Rotate the atoms and crossing points
        atoms.rotate(-np.degrees(angle), "z", center=(0, 0, 0))
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle), 0],
            [np.sin(-angle),  np.cos(-angle), 0],
            [0,               0,              1]
        ])
        rotated_crossing_points = np.dot(crossing_points, rotation_matrix.T)

        # Define x-bounds based on crossing points
        x_min = min(rotated_crossing_points[:, 0])
        x_max = max(rotated_crossing_points[:, 0]) + 0.1

        # Extract positions and define the mask for atoms within bounds
        positions = atoms.get_positions()
        mask = (positions[:, 0] > x_min) & (positions[:, 0] <= x_max)
        bounded_atoms = atoms[mask].copy()
        mask = (positions[:, 0] <= x_min) 
        tail_atoms = atoms[mask].copy()
        mask = (positions[:, 0] > x_max)
        head_atoms = atoms[mask].copy()
        try:
            Nunits = int(units)
        except ValueError:
            Nunits = None
        
        if Nunits is None or Nunits < 1:
            # Trim atoms based on x-bounds
            atoms = bounded_atoms
        else:
            # Replicate atoms for Nunits
            replicated_atoms = bounded_atoms.copy()
            for ni in range(1,Nunits):
                shifted_positions = bounded_atoms.get_positions() + np.array([ni * norm_vector, 0, 0])
                replicated_atoms += ase.Atoms(bounded_atoms.get_chemical_symbols(), positions=shifted_positions)

            # Add atoms shifted beyond xmax
            shifted_positions = head_atoms.get_positions() + np.array([(Nunits-1) * norm_vector, 0, 0])
            replicated_atoms += ase.Atoms(head_atoms.get_chemical_symbols(), positions=shifted_positions)
            replicated_atoms += tail_atoms
            atoms = replicated_atoms

        # Set the new unit cell
        if Nunits is None or Nunits < 1:
            L1 = norm_vector
        else:
            L1 = np.ptp(atoms.get_positions()[:, 0]) + 10.0  # Size in x-direction + 10 Å
        L2 = 10.0 + np.ptp(atoms.get_positions()[:, 1])  # Size in y-direction + 10 Å
        L3 = 15.0  # Fixed value
        atoms.set_cell([L1, L2, L3])
        atoms.center()

        return atoms
    

    def _on_file_upload(self, change=None):
        """When file upload button is pressed."""
        #self._structure_selector.options = [None]
        #self._structure_selector.disabled = True
        self.nunits.value = 'Infinite'
        self.nunits.disabled = True
        fname, item = next(iter(change["new"].items()))

        # Decode the uploaded file content
        cdxml_content = item["content"].decode("ascii")

        try:
            # Extract parentheses midpoints from the CDXML content
            #open_midpoints, closed_midpoints = self.extract_parentheses_boxes_from_content(cdxml_content)
            self.crossing_points, self.cdxml_atoms, self.nunits.disabled = self.extract_crossing_and_atom_positions(cdxml_content)


            # Convert CDXML content to RDKit molecules and ASE Atoms objects
            options = [
                self.rdkit2ase(mol)
                for mol in rdkit.Chem.MolsFromCDXML(cdxml_content)
            ]
            self.atoms = options[0]
            # Populate the structure selector
            #self._structure_selector.options = [
            #    (f"{i}: " + mol.get_chemical_formula(), mol)
            #    for i, mol in enumerate(options)
            #]
            #self._structure_selector.disabled = False

        except Exception as exc:
            self.output_message.value = f"Error reading file: {exc}"

        # Clear the file upload widget
        self.file_upload.value.clear()        

        
        
    #def _on_sketch_selected(self, change=None):
    def _on_button_click(self,b):
        atoms = self.atoms
        if self.crossing_points is not None:
            crossing_points = self.transform_points(self.cdxml_atoms, atoms.positions, self.crossing_points)
            atoms = self.align_and_trim_atoms(atoms,np.array(crossing_points),units=self.nunits.value)
        factor = self.guess_scaling_factor(atoms)
        atoms = self.scale(atoms, factor)
        self.output_message.value, atoms = self.add_h(atoms)
        self.structure = atoms
