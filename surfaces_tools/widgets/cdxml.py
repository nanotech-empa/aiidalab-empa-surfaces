"""Widget to convert CDXML to planar structures"""

import xml.etree.ElementTree as ET
import nglview as nv
import ase
import ipywidgets as ipw
import numpy as np
import traitlets as tr
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.neighborlist import NeighborList
from scipy.spatial.distance import pdist


class CdxmlUpload2GnrWidget(ipw.VBox):
    """Widget for uploading CDXML files and converting them to ASE Atoms structures."""

    structure = tr.Instance(ase.Atoms, allow_none=True)

    def __init__(self, title="CDXML to GNR", description="Upload Structure"):
        self.title = title

        # File upload widget for .cdxml files
        self.file_upload = ipw.FileUpload(
            description=description,
            multiple=False,
            layout={"width": "initial"},
            accept=".cdxml",
        )
        self.file_upload.observe(self._on_file_upload, names="value")

        # Additional widgets
        self.nunits = ipw.Text(description="N units", value="Infinite", disabled=True)
        self.create_button = ipw.Button(
            description="Create model",
            button_style="success",
        )
        self.create_button.on_click(self._on_button_click)

        supported_formats = ipw.HTML(
            """
            <a href="https://pubs.acs.org/doi/10.1021/ja0697875" target="_blank">
            Supported structure formats: ".cdxml"
            </a>
            """
        )

        # Output message widget
        self.output_message = ipw.HTML(value="")

        # Initialize the widget layout
        super().__init__(
            children=[
                self.file_upload,
                self.nunits,
                supported_formats,
                self.create_button,
                self.output_message,
            ]
        )

        # Internal state
        self.structure = None
        self.crossing_points = None
        self.cdxml_atoms = None
        self.atoms = None

    @staticmethod
    def add_hydrogen_atoms(atoms: Atoms) -> tuple[str, Atoms]:
        """Add missing hydrogen atoms to the Atoms object based on covalent radii."""
        message = ""

        neighbor_list = NeighborList(
            [covalent_radii[atom.number] for atom in atoms],
            bothways=True,
            self_interaction=False,
        )
        neighbor_list.update(atoms)

        need_hydrogen = [
            atom.index
            for atom in atoms
            if len(neighbor_list.get_neighbors(atom.index)[0]) < 3
            and atom.symbol in {"C", "N"}
        ]

        message = f"Added missing Hydrogen atoms: {need_hydrogen}."

        for index in need_hydrogen:
            vec = np.zeros(3)
            indices, offsets = neighbor_list.get_neighbors(atoms[index].index)
            for i, offset in zip(indices, offsets):
                vec += -atoms[index].position + (
                    atoms.positions[i] + np.dot(offset, atoms.get_cell())
                )
            vec = -vec / np.linalg.norm(vec) * 1.1 + atoms[index].position
            atoms.append(ase.Atom("H", vec))

        return message, atoms

    def _on_file_upload(self, change=None):
        """Handles the file upload event and converts CDXML to ASE Atoms."""
        self.nunits.value = "Infinite"
        self.nunits.disabled = True

        uploaded_file = list(self.file_upload.value.values())[0]
        cdxml_content = uploaded_file["content"].decode("utf-8")
        try:
            self.atoms = self.cdxml_to_ase_from_string(cdxml_content)
            (
                self.crossing_points,
                self.cdxml_atoms,
                self.nunits.disabled,
            ) = self.extract_crossing_and_atom_positions(cdxml_content)
        except ValueError as exc:
            self.output_message.value = f"Error: {exc}"
        except Exception as exc:
            self.output_message.value = f"Unexpected error: {exc}"

        # Clear the file upload widget
        self.file_upload.value.clear()

    def _on_button_click(self, _=None):
        """Handles the creation of the ASE model when 'Create model' button is clicked."""
        if not self.atoms:
            self.output_message.value = "Error: No atoms available to process."
            return

        atoms = self.atoms.copy()
        

        if self.crossing_points is not None:
            crossing_points = self.transform_points(
                self.cdxml_atoms, atoms.positions, self.crossing_points
            )
            atoms = self.align_and_trim_atoms(
                atoms, np.array(crossing_points), units=self.nunits.value
            )
        else:
            self.output_message.value = "Error: No 'crossing points' found."
            return

        if self.nunits.disabled:
            extra_cell = 15.0
            atoms.cell = (np.ptp(atoms.positions, axis=0)) + extra_cell
            atoms.center()

        if self.nunits.value == "Infinite":
            atoms.pbc = True

        self.output_message.value, self.structure = self.add_hydrogen_atoms(atoms)
    
    @staticmethod
    def cdxml_to_ase_from_string(cdxml_content: str, target_cc_distance: float = 1.43) -> ase.Atoms:
        """
        Converts CDXML content provided as a string into an ASE Atoms object,
        scaling coordinates so that the smallest C-C distance is target_cc_distance (default: 1.43 Å).
        Atoms without an 'Element' attribute are considered Carbon ('C').
        Atoms with an 'Element' attribute use the periodic table symbol.

        Args:
            cdxml_content (str): The content of the CDXML file as a string.
            target_cc_distance (float): Desired minimum C-C distance (default: 1.43 Å).

        Returns:
            Atoms: An ASE Atoms object with scaled coordinates.
        """
        # Parse the CDXML content from the string
        root = ET.fromstring(cdxml_content)

        # Extract atom data from 'n' elements
        symbols = []
        positions = []

        for atom in root.findall('.//n'):
            # Determine the element symbol
            if 'Element' in atom.attrib:
                # Convert atomic number to element symbol using ASE's chemical_symbols
                element_number = int(atom.get('Element'))
                if element_number < len(chemical_symbols):
                    element = chemical_symbols[element_number]
                else:
                    raise ValueError(f"Unknown atomic number {element_number} in CDXML content.")
            else:
                # Default to Carbon ('C') if no Element attribute is present
                element = 'C'

            symbols.append(element)

            # Get 2D coordinates from 'p' attribute and assume z=0
            p = atom.get('p', '0 0').split()
            x, y = float(p[0]), float(p[1])
            positions.append([x, y, 0.0])

        if not symbols or not positions:
            raise ValueError("No valid atoms found in the CDXML content.")

        # Convert positions to a numpy array
        positions = np.array(positions)

        # Find the smallest C-C distance
        carbon_indices = [i for i, sym in enumerate(symbols) if sym == 'C']
        if len(carbon_indices) < 2:
            raise ValueError("Not enough Carbon atoms to calculate C-C distance.")

        # Calculate pairwise distances between all Carbon atoms
        carbon_positions = positions[carbon_indices]
        cc_distances = pdist(carbon_positions)

        # Find the minimum C-C distance
        min_cc_distance = np.min(cc_distances)

        # Scale coordinates to set the minimum C-C distance to target_cc_distance
        scale_factor = target_cc_distance / min_cc_distance
        positions *= scale_factor

        # Create an ASE Atoms object with the scaled positions
        ase_atoms = ase.Atoms(symbols=symbols, positions=positions)

        return ase_atoms    
    
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
        # set1 = np.array(set1)
        # set2 = np.array(set2)
        # points = np.array(points)

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
        cross_covariance = np.dot(centered_set1.T, centered_set2)
        u_matrix, _, vt_matrix = np.linalg.svd(cross_covariance)
        rotation_m = np.dot(vt_matrix.T, u_matrix.T)  # Rotation matrix

        # Apply the scaling and rotation to the points
        transformed_points = (
            scale * np.dot(points - centroid1, rotation_m.T) + centroid2
        )

        return transformed_points.tolist()
    
    @staticmethod
    def max_extension_points(points):
        """
        Given a list of points, checks whether the maximum extension is along the x-axis or y-axis,
        and returns two points accordingly.

        Args:
            points (list of tuple): List of points as (x, y, z) coordinates.

        Returns:
            tuple: Two points as ((x1, y1, z1), (x2, y2, z2))
        """
        # Unpack x, y, and z coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        # Calculate the range along x and y
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        # Determine the points based on the largest range
        if x_range >= y_range:
            minx, maxx = min(x_coords), max(x_coords)
            return np.array([[minx - 7.5, 0, 0], [maxx + 7.5, 0, 0]])
        else:
            miny, maxy = min(y_coords), max(y_coords)
            return np.array([[0, miny - 7.5, 0], [0, maxy + 7.5, 0]])



    def extract_crossing_and_atom_positions(self, cdxml_content: str):
        """
        Extract the first two crossing points such that the vector connecting them is aligned with the unit vector.

        Args:
            cdxml_content (str): The content of the CDXML file as a string.

        Returns:
            tuple: Three values:
                - crossing_points_pair: A tuple of two crossing points that are aligned with the unit vector.
                  The unit_vector is a vector with positive x and y, parallel to the vector connecting the square brackets
                - atom_positions: Atom positions as a numpy array of shape (M, 3).
                - isnotperiodic (bool): Indicates whether the structure is non-periodic.
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

        for crossing in root.findall(".//crossingbond"):
            bond_id = crossing.get("BondID")

            if bond_id:
                bond = root.find(f".//b[@id='{bond_id}']")
                if bond is not None:
                    start_id = bond.get("B")
                    end_id = bond.get("E")
                    if start_id in atom_id_map and end_id in atom_id_map:
                        start_pos = atom_positions[atom_id_map[start_id]]
                        end_pos = atom_positions[atom_id_map[end_id]]

                        midpoint = (
                            (start_pos[0] + end_pos[0]) / 2,
                            (start_pos[1] + end_pos[1]) / 2,
                            0.0,  # Add z=0
                        )
                        crossing_points.append(midpoint)

        crossing_points = np.array(crossing_points)

        # Parse square parentheses
        brackets = []
        for graphic in root.findall(".//graphic[@BracketType='Square']"):
            if "BoundingBox" in graphic.attrib:
                bb = list(map(float, graphic.attrib["BoundingBox"].split()))
                x_min, y_min, x_max, y_max = bb

                midpoint = ((x_min + x_max) / 2, (y_min + y_max) / 2, 0.0)
                brackets.append(midpoint)

        isnotperiodic = True
        if len(brackets) == 0:
            twopoints = self.max_extension_points(atom_positions)
            return twopoints, atom_positions, isnotperiodic

        if len(brackets) == 2:
            brackets = np.array(brackets)
            vector = brackets[1] - brackets[0]
            unit_vector = vector[:2] / np.linalg.norm(vector[:2])

            if unit_vector[0] < 0 or unit_vector[1] < 0:
                unit_vector = -unit_vector

            for i in range(len(crossing_points)):
                for j in range(i + 1, len(crossing_points)):
                    vector = crossing_points[j][:2] - crossing_points[i][:2]
                    unit_test_vector = vector / np.linalg.norm(vector)
                    if np.dot(unit_test_vector, unit_vector) > 0.99:
                        isnotperiodic = False
                        return crossing_points[[i, j]], atom_positions, isnotperiodic

        return None, atom_positions, True


    @staticmethod
    def align_and_trim_atoms(atoms, crossing_points, units=None):
        """
        Aligns an ASE Atoms object with the x-axis based on two crossing points,
        trims or replicates atoms based on specified x-bounds, sets a new unit cell, and centers the structure.

        Args:
            atoms (ASE.Atoms): The ASE Atoms object to transform.
            crossing_points (numpy.ndarray): A 2x3 NumPy array containing two crossing points.
            n_units (int, optional): Number of units to replicate along the x-axis. If None, trims atoms.

        Returns:
            ASE.Atoms: The transformed ASE Atoms object.
        """
        # Ensure crossing_points is a 2x3 array
        assert crossing_points.shape == (
            2,
            3,
        ), "crossing_points must be a 2x3 NumPy array."

        # Calculate the vector connecting the two crossing points
        vector = crossing_points[1] - crossing_points[0]
        norm_vector = np.linalg.norm(vector)

        # Calculate rotation angle to align the vector with the x-axis
        angle = np.arctan2(vector[1], vector[0])

        # Rotate the atoms and crossing points
        atoms.rotate(-np.degrees(angle), "z", center=(0, 0, 0))
        rotation_matrix = np.array(
            [
                [np.cos(-angle), -np.sin(-angle), 0],
                [np.sin(-angle), np.cos(-angle), 0],
                [0, 0, 1],
            ]
        )
        rotated_crossing_points = np.dot(crossing_points, rotation_matrix.T)

        # Define x-bounds based on crossing points
        x_min = min(rotated_crossing_points[:, 0])
        x_max = max(rotated_crossing_points[:, 0]) + 0.1

        # Extract positions and define the mask for atoms within bounds
        positions = atoms.get_positions()
        mask = (positions[:, 0] > x_min) & (positions[:, 0] <= x_max)
        bounded_atoms = atoms[mask].copy()
        mask = positions[:, 0] <= x_min
        tail_atoms = atoms[mask].copy()
        mask = positions[:, 0] > x_max
        head_atoms = atoms[mask].copy()
        try:
            n_units = int(units)
        except ValueError:
            n_units = None

        if n_units is None or n_units < 1:
            # Trim atoms based on x-bounds
            atoms = bounded_atoms
        else:
            # Replicate atoms for n_units
            replicated_atoms = bounded_atoms.copy()
            for ni in range(1, n_units):
                shifted_positions = bounded_atoms.get_positions() + np.array(
                    [ni * norm_vector, 0, 0]
                )
                replicated_atoms += ase.Atoms(
                    bounded_atoms.get_chemical_symbols(), positions=shifted_positions
                )

            # Add atoms shifted beyond xmax
            shifted_positions = head_atoms.get_positions() + np.array(
                [(n_units - 1) * norm_vector, 0, 0]
            )
            replicated_atoms += ase.Atoms(
                head_atoms.get_chemical_symbols(), positions=shifted_positions
            )
            replicated_atoms += tail_atoms
            atoms = replicated_atoms

        # Set the new unit cell
        if n_units is None or n_units < 1:
            l1 = norm_vector
            atoms.set_periodic=True
        else:
            l1 = (
                np.ptp(atoms.get_positions()[:, 0]) + 15.0
            )  # Size in x-direction + 10 Å
        l2 = 15.0 + np.ptp(atoms.get_positions()[:, 1])  # Size in y-direction + 15 Å
        l3 = 15.0  # Fixed value
        atoms.set_cell([l1, l2, l3])
        atoms.center()

        return atoms
