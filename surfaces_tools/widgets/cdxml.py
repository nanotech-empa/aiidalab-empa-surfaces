"""Widget to convert CDXML to planar structures"""

import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import DefaultDict, Optional

import ase
import ipywidgets as ipw
import nglview as nv
import numpy as np
import pandas as pd
import traitlets as tr
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.neighborlist import NeighborList
from scipy.spatial.distance import pdist

DEFAULT_VALENCES = {
    "C": 4,
    "N": 3,
    "O": 2,
    "S": 2,
}

HYDROGEN_BOND_LENGTHS = {
    "C": 1.09,
    "N": 1.01,
    "O": 0.96,
    "S": 1.34,
}

CDXML_BOND_ORDER_ARRAY = "cdxml_bond_order"
CDXML_MAX_BOND_ORDER_ARRAY = "cdxml_max_bond_order"
FLOAT_RE = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return a normalized 3-vector, or zeros for a near-zero vector."""
    vector = np.array(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else np.zeros(3)


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the rotation matrix that rotates source into target."""
    source = normalize(source)
    target = normalize(target)
    cross = np.cross(source, target)
    dot = np.dot(source, target)
    if np.linalg.norm(cross) < 1e-8:
        return np.eye(3)
    skew = np.array(
        [
            [0, -cross[2], cross[1]],
            [cross[2], 0, -cross[0]],
            [-cross[1], cross[0], 0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1 - dot) / (np.linalg.norm(cross) ** 2))


def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector around axis by angle radians."""
    axis = normalize(axis)
    vector = np.array(vector, dtype=float)
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * np.dot(axis, vector) * (1 - math.cos(angle))
    )


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
        self.use_clever_hydrogenation = ipw.Checkbox(
            description="Use clever hydrogenation",
            value=True,
            indent=False,
        )
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
                self.use_clever_hydrogenation,
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
    def _hydrogen_count(symbol: str, bond_order: float) -> int:
        target_valence = DEFAULT_VALENCES.get(symbol)
        if target_valence is None:
            return 0
        return max(0, int(round(target_valence - bond_order)))

    @staticmethod
    def _expected_neighbor_count(
        n_hydrogen: int, bond_order: float, max_bond_order: float
    ) -> int:
        if n_hydrogen == 3:
            return 1
        if n_hydrogen == 2 and max_bond_order >= 1.5:
            return 1
        if n_hydrogen == 1 and max_bond_order >= 1.5:
            return 2
        return max(1, int(round(bond_order)))

    @staticmethod
    def _plane_normal(neighbor_vectors: list[np.ndarray]) -> np.ndarray:
        for i, first in enumerate(neighbor_vectors):
            for second in neighbor_vectors[i + 1 :]:
                normal = normalize(np.cross(first, second))
                if np.linalg.norm(normal) > 1e-8:
                    return normal
        return np.array([0.0, 0.0, 1.0])

    @classmethod
    def _hydrogen_directions(
        cls,
        symbol: str,
        n_hydrogen: int,
        neighbor_vectors: list[np.ndarray],
        max_bond_order: float,
    ) -> tuple[list[np.ndarray], float]:
        if n_hydrogen <= 0:
            return [], HYDROGEN_BOND_LENGTHS.get(symbol, 1.1)

        has_multiple_bond = max_bond_order >= 1.5
        if symbol in {"O", "N"}:
            v_sum = (
                np.sum(neighbor_vectors, axis=0) if neighbor_vectors else np.zeros(3)
            )
            base_dir = (
                normalize(-v_sum)
                if np.linalg.norm(v_sum) > 1e-6
                else np.array([0.0, 0.0, 1.0])
            )
            bond_length = HYDROGEN_BOND_LENGTHS.get(symbol, 1.0)

            if n_hydrogen == 1:
                if symbol == "O" and len(neighbor_vectors) == 1:
                    return [
                        rotate_vector(
                            -neighbor_vectors[0],
                            np.array([0.0, 0.0, 1.0]),
                            math.radians(35),
                        )
                    ], bond_length
                return [base_dir], bond_length

            if n_hydrogen == 2:
                theta = math.radians(104.5 if symbol == "O" else 107.0)
                rot_axis = cls._plane_normal(neighbor_vectors)
                return [
                    rotate_vector(base_dir, rot_axis, angle)
                    for angle in (-theta / 2, theta / 2)
                ], bond_length

            return [base_dir], bond_length

        if symbol != "C":
            v_sum = (
                np.sum(neighbor_vectors, axis=0) if neighbor_vectors else np.zeros(3)
            )
            avg = (
                normalize(-v_sum)
                if np.linalg.norm(v_sum) > 1e-6
                else np.array([0.0, 0.0, 1.0])
            )
            return [avg], HYDROGEN_BOND_LENGTHS.get(symbol, 1.01)

        # CH3: tetrahedral cone around the opposite heavy-atom bond.
        if n_hydrogen == 3 and len(neighbor_vectors) == 1:
            v = neighbor_vectors[0]
            theta = math.radians(109.47)
            local_dirs = [
                np.array(
                    [
                        math.sin(theta) * math.cos(phi),
                        math.sin(theta) * math.sin(phi),
                        math.cos(theta),
                    ]
                )
                for phi in (0, 2 * math.pi / 3, 4 * math.pi / 3)
            ]
            rotation = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), -v)
            return [-rotation @ direction for direction in local_dirs], 1.10

        # CH2: planar if attached by a multiple/aromatic bond, tetrahedral otherwise.
        if n_hydrogen == 2:
            bond_length = 1.09 if has_multiple_bond else 1.10
            if len(neighbor_vectors) == 1:
                v = neighbor_vectors[0]
                if has_multiple_bond:
                    bis = -v
                    return [
                        rotate_vector(bis, np.array([0.0, 0.0, 1.0]), angle)
                        for angle in (math.radians(60), math.radians(-60))
                    ], bond_length

                theta = math.radians(109.47)
                local_dirs = [
                    np.array(
                        [
                            math.sin(theta) * math.cos(phi),
                            math.sin(theta) * math.sin(phi),
                            math.cos(theta),
                        ]
                    )
                    for phi in (0, 2 * math.pi / 3)
                ]
                rotation = rotation_matrix_from_vectors(np.array([0.0, 0.0, 1.0]), -v)
                return [rotation @ direction for direction in local_dirs], bond_length

            if len(neighbor_vectors) >= 2:
                v1, v2 = neighbor_vectors[:2]
                bis = normalize(-(v1 + v2))
                plane_normal = cls._plane_normal([v1, v2])
                if has_multiple_bond:
                    return [
                        rotate_vector(bis, plane_normal, angle)
                        for angle in (math.radians(60), math.radians(-60))
                    ], bond_length

                angle = math.radians(54.75)
                return [
                    math.cos(angle) * bis + math.sin(angle) * plane_normal,
                    math.cos(angle) * bis - math.sin(angle) * plane_normal,
                ], bond_length

        # CH: sp3 centers with three planar single bonds get the H out of plane.
        if n_hydrogen == 1:
            if len(neighbor_vectors) >= 3 and not has_multiple_bond:
                return [cls._plane_normal(neighbor_vectors)], 1.09
            v_sum = (
                np.sum(neighbor_vectors, axis=0) if neighbor_vectors else np.zeros(3)
            )
            avg = (
                normalize(-v_sum)
                if np.linalg.norm(v_sum) > 1e-6
                else cls._plane_normal(neighbor_vectors)
            )
            return [avg], 1.09

        return [], HYDROGEN_BOND_LENGTHS.get(symbol, 1.1)

    @classmethod
    def add_hydrogen_atoms(
        cls,
        atoms: Atoms,
        bond_orders: Optional[list[float]] = None,
        use_clever_hydrogenation: bool = True,
    ) -> tuple[str, Atoms]:
        """Add missing H atoms while preserving the planar CDXML geometry."""
        neighbor_list = NeighborList(
            [covalent_radii[atom.number] for atom in atoms],
            bothways=True,
            self_interaction=False,
        )
        neighbor_list.update(atoms)

        if not use_clever_hydrogenation:
            need_hydrogen = [
                atom.index
                for atom in atoms
                if len(neighbor_list.get_neighbors(atom.index)[0]) < 3
                and atom.symbol == "C"
            ]

            for index in need_hydrogen:
                vec = np.zeros(3)
                indices, offsets = neighbor_list.get_neighbors(atoms[index].index)
                for i, offset in zip(indices, offsets):
                    vec += -atoms[index].position + (
                        atoms.positions[i] + np.dot(offset, atoms.get_cell())
                    )
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-12:
                    position = -vec / vec_norm * 1.1 + atoms[index].position
                else:
                    position = atoms[index].position + np.array([0.0, 0.0, 1.1])
                atoms.append(ase.Atom("H", position))

            message = (
                f"Added missing Hydrogen atoms (safe hydrogenation): {need_hydrogen}."
            )
            return message, atoms

        if bond_orders is None and CDXML_BOND_ORDER_ARRAY in atoms.arrays:
            bond_orders = atoms.arrays[CDXML_BOND_ORDER_ARRAY]
        if bond_orders is None:
            bond_orders = [
                len(neighbor_list.get_neighbors(atom.index)[0]) for atom in atoms
            ]
        if CDXML_MAX_BOND_ORDER_ARRAY in atoms.arrays:
            max_bond_orders = atoms.arrays[CDXML_MAX_BOND_ORDER_ARRAY]
        else:
            max_bond_orders = np.ones(len(atoms))

        added_hydrogens = []

        for index, atom in enumerate(atoms):
            if atom.symbol == "H":
                continue
            n_hydrogen = cls._hydrogen_count(atom.symbol, float(bond_orders[index]))
            if n_hydrogen == 0:
                continue

            neighbor_candidates = []
            indices, offsets = neighbor_list.get_neighbors(atoms[index].index)
            for i, offset in zip(indices, offsets):
                if atoms[i].symbol == "H":
                    continue
                vec = (
                    atoms.positions[i]
                    + np.dot(offset, atoms.get_cell())
                    - atoms[index].position
                )
                vec[2] = 0.0
                distance = np.linalg.norm(vec[:2])
                if distance > 1e-8:
                    neighbor_candidates.append((distance, normalize(vec)))

            expected_neighbors = cls._expected_neighbor_count(
                n_hydrogen,
                float(bond_orders[index]),
                float(max_bond_orders[index]),
            )
            neighbor_vectors = [
                vector
                for _, vector in sorted(neighbor_candidates, key=lambda item: item[0])[
                    :expected_neighbors
                ]
            ]

            directions, bond_length = cls._hydrogen_directions(
                atom.symbol,
                n_hydrogen,
                neighbor_vectors,
                float(max_bond_orders[index]),
            )
            for direction in directions:
                atoms.append(ase.Atom("H", atom.position + bond_length * direction))
                added_hydrogens.append(index)

        message = (
            f"Added missing Hydrogen atoms (clever hydrogenation): {added_hydrogens}."
        )

        return message, atoms

    def _on_file_upload(self, change=None):
        """Handles the file upload event and converts CDXML to ASE Atoms."""
        self.nunits.value = "Infinite"
        self.nunits.disabled = True

        upload_value = self.file_upload.value
        if not upload_value:
            return
        if isinstance(upload_value, dict):
            uploaded_file = list(upload_value.values())[0]
        else:
            uploaded_file = upload_value[0]
        content = uploaded_file["content"]
        if isinstance(content, str):
            cdxml_content = content
        else:
            cdxml_content = bytes(content).decode("utf-8")
        try:
            bond_orders, max_bond_orders = self.get_bond_order_arrays_from_cdxml_string(
                cdxml_content
            )
            self.atoms = self.cdxml_to_ase_from_string(
                cdxml_content,
                bond_orders=bond_orders,
                max_bond_orders=max_bond_orders,
            )
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
        if isinstance(self.file_upload.value, dict):
            self.file_upload.value.clear()
        else:
            self.file_upload.value = ()

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

        self.output_message.value, self.structure = self.add_hydrogen_atoms(
            atoms,
            use_clever_hydrogenation=self.use_clever_hydrogenation.value,
        )
        # self.structure = struc

    @staticmethod
    def _parse_float_list(text: str) -> list[float]:
        return [float(value) for value in re.findall(FLOAT_RE, text)]

    @staticmethod
    def _parse_bond_order(value) -> float:
        if value is None:
            return 1.0
        try:
            return float(value)
        except ValueError:
            order_names = {
                "single": 1.0,
                "double": 2.0,
                "triple": 3.0,
                "quadruple": 4.0,
                "aromatic": 1.5,
                "dative": 1.0,
            }
            return order_names.get(value.strip().lower(), 1.0)

    @staticmethod
    def get_bond_order_arrays_from_cdxml_string(
        cdxml_content: str,
    ) -> tuple[list[float], list[float]]:
        """
        Parses CDXML content provided as a string and computes the total bond order
        for each atom, including implicit single bonds from graphical electrons (radicals).

        Lone electrons are detected via SymbolType="Electron", specific representations
        (like '•', '.', 'radical'), or by small bounding boxes. Each such electron is
        counted as a single bond to the nearest atom.

        Args:
            cdxml_content (str): The content of the CDXML file as a string.

        Returns:
            Two lists ordered like the CDXML atoms: total bond order and maximum
            explicit bond order per atom.
        """
        # Parse XML content from string
        # tree = ET.parse(io.StringIO(cdxml_content))
        root = ET.fromstring(cdxml_content)
        # Extract atoms and their 2D positions from 'p' attribute
        atoms = []
        for n in root.iter("n"):
            pos_str = n.attrib.get("p")
            if pos_str:
                x_str, y_str = pos_str.strip().split()
                atoms.append(
                    {
                        "id": n.attrib.get("id"),
                        "element": n.attrib.get("Element"),
                        "x": float(x_str),
                        "y": float(y_str),
                    }
                )
        atoms_df = pd.DataFrame(atoms)
        if atoms_df.empty:
            return [], []

        # Extract bonds between atoms
        bonds = []
        for b in root.iter("b"):
            bonds.append(
                {
                    "from": b.attrib.get("B"),
                    "to": b.attrib.get("E"),
                    "order": CdxmlUpload2GnrWidget._parse_bond_order(
                        b.attrib.get("Order")
                    ),
                }
            )

        # Helper function to extract center of a bounding box string
        def parse_bbox_center(bbox_str):
            coords = CdxmlUpload2GnrWidget._parse_float_list(bbox_str)
            if len(coords) == 4:
                return (coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2
            return None, None

        # Find lone electrons represented as graphics and assign them to nearest atom
        electron_bonds = []
        for graphic in root.iter("graphic"):
            bbox_str = graphic.attrib.get("BoundingBox")
            if not bbox_str:
                continue

            is_electron = False

            # Method 1: Check for SymbolType="Electron"
            if graphic.attrib.get("SymbolType") == "Electron":
                is_electron = True

            # Method 2: Check represent text content
            if not is_electron:
                for r in graphic.iter("represent"):
                    if r.text and r.text.strip() in {
                        "•",
                        ".",
                        "radical",
                        "unpaired electron",
                    }:
                        is_electron = True

            # Method 3: Fallback – small bounding box dimensions
            if not is_electron:
                coords = CdxmlUpload2GnrWidget._parse_float_list(bbox_str)
                if len(coords) == 4:
                    width = abs(coords[2] - coords[0])
                    height = abs(coords[3] - coords[1])
                    if width < 5 and height < 5:
                        is_electron = True

            if not is_electron:
                continue  # Skip non-electron graphics

            # Assign the electron to the closest atom (if within a reasonable distance)
            x, y = parse_bbox_center(bbox_str)
            atoms_df["distance"] = (
                (atoms_df["x"] - x) ** 2 + (atoms_df["y"] - y) ** 2
            ) ** 0.5
            closest = atoms_df.loc[atoms_df["distance"].idxmin()]

            if closest["distance"] < 15:
                electron_bonds.append(
                    {"from": closest["id"], "to": "electron", "order": 1.0}
                )

        # Combine all bonds and accumulate bond orders per atom
        all_bonds = bonds + electron_bonds
        connectivity: DefaultDict[str, float] = defaultdict(float)
        max_connectivity: DefaultDict[str, float] = defaultdict(float)
        for bond in all_bonds:
            if bond["from"] is None:
                continue
            connectivity[bond["from"]] += bond["order"]
            max_connectivity[bond["from"]] = max(
                max_connectivity[bond["from"]], bond["order"]
            )
            if bond["to"] not in {None, "electron"}:
                connectivity[bond["to"]] += bond["order"]
                max_connectivity[bond["to"]] = max(
                    max_connectivity[bond["to"]], bond["order"]
                )

        atom_ids = atoms_df["id"].tolist()
        return (
            [connectivity.get(atom_id, 0.0) for atom_id in atom_ids],
            [max_connectivity.get(atom_id, 0.0) for atom_id in atom_ids],
        )

    @staticmethod
    def get_total_bond_orders_from_cdxml_string(cdxml_content: str) -> list[float]:
        return CdxmlUpload2GnrWidget.get_bond_order_arrays_from_cdxml_string(
            cdxml_content
        )[0]

    @staticmethod
    def cdxml_to_ase_from_string(
        cdxml_content: str,
        target_cc_distance: float = 1.43,
        bond_orders: Optional[list[float]] = None,
        max_bond_orders: Optional[list[float]] = None,
    ) -> ase.Atoms:
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

        for atom in root.findall(".//n"):
            # Determine the element symbol
            if "Element" in atom.attrib:
                # Convert atomic number to element symbol using ASE's chemical_symbols
                element_number = int(atom.attrib["Element"])
                if element_number < len(chemical_symbols):
                    element = chemical_symbols[element_number]
                else:
                    raise ValueError(
                        f"Unknown atomic number {element_number} in CDXML content."
                    )
            else:
                # Default to Carbon ('C') if no Element attribute is present
                element = "C"

            symbols.append(element)

            # Get 2D coordinates from 'p' attribute and assume z=0
            p = atom.get("p", "0 0").split()
            x, y = float(p[0]), float(p[1])
            positions.append([x, y, 0.0])

        if not symbols or not positions:
            raise ValueError("No valid atoms found in the CDXML content.")

        # Convert positions to a numpy array
        positions_array = np.array(positions)

        # Find the smallest C-C distance
        carbon_indices = [i for i, sym in enumerate(symbols) if sym == "C"]
        if len(carbon_indices) < 2:
            raise ValueError("Not enough Carbon atoms to calculate C-C distance.")

        # Calculate pairwise distances between all Carbon atoms
        carbon_positions = positions_array[carbon_indices]
        cc_distances = pdist(carbon_positions)

        # Find the minimum C-C distance
        min_cc_distance = np.min(cc_distances)

        # Scale coordinates to set the minimum C-C distance to target_cc_distance
        scale_factor = target_cc_distance / min_cc_distance
        positions_array *= scale_factor

        # Create an ASE Atoms object with the scaled positions
        ase_atoms = ase.Atoms(symbols=symbols, positions=positions_array)
        if bond_orders is not None:
            if len(bond_orders) != len(ase_atoms):
                raise ValueError(
                    "Number of CDXML bond-order entries does not match atoms."
                )
            ase_atoms.new_array(
                CDXML_BOND_ORDER_ARRAY,
                np.array(bond_orders, dtype=float),
            )
        if max_bond_orders is not None:
            if len(max_bond_orders) != len(ase_atoms):
                raise ValueError(
                    "Number of CDXML max-bond-order entries does not match atoms."
                )
            ase_atoms.new_array(
                CDXML_MAX_BOND_ORDER_ARRAY,
                np.array(max_bond_orders, dtype=float),
            )

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

        crossing_points_array = np.array(crossing_points)

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
            brackets_array = np.array(brackets)
            vector = brackets_array[1] - brackets_array[0]
            unit_vector = vector[:2] / np.linalg.norm(vector[:2])

            if unit_vector[0] < 0 or unit_vector[1] < 0:
                unit_vector = -unit_vector

            for i in range(len(crossing_points_array)):
                for j in range(i + 1, len(crossing_points_array)):
                    vector = crossing_points_array[j][:2] - crossing_points_array[i][:2]
                    unit_test_vector = vector / np.linalg.norm(vector)
                    if np.dot(unit_test_vector, unit_vector) > 0.99:
                        isnotperiodic = False
                        return (
                            np.array(
                                [crossing_points_array[i], crossing_points_array[j]]
                            ),
                            atom_positions,
                            isnotperiodic,
                        )

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
                shifted_atoms = bounded_atoms.copy()
                shifted_atoms.positions = bounded_atoms.get_positions() + np.array(
                    [ni * norm_vector, 0, 0]
                )
                replicated_atoms += shifted_atoms

            # Add atoms shifted beyond xmax
            shifted_head_atoms = head_atoms.copy()
            shifted_head_atoms.positions = head_atoms.get_positions() + np.array(
                [(n_units - 1) * norm_vector, 0, 0]
            )
            replicated_atoms += shifted_head_atoms
            replicated_atoms += tail_atoms
            atoms = replicated_atoms

        # Set the new unit cell
        if n_units is None or n_units < 1:
            l1 = norm_vector
            atoms.set_periodic = True
        else:
            l1 = (
                np.ptp(atoms.get_positions()[:, 0]) + 15.0
            )  # Size in x-direction + 10 Å
        l2 = 15.0 + np.ptp(atoms.get_positions()[:, 1])  # Size in y-direction + 15 Å
        l3 = 15.0  # Fixed value
        atoms.set_cell([l1, l2, l3])
        atoms.center()

        return atoms
