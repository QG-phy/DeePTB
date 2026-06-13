from typing import Optional, Tuple

import numpy as np
from ase import Atoms

from dptb.postprocess.unified.eph.utils import (
    as_array,
    assemble_directed_hk_from_blocks,
    normalize_integer_array,
    normalize_kpoints,
    normalize_orbital_slices,
    orbital_slices_from_system,
    strip_single_k_matrix,
    validate_finite_positive_scalar,
)


class FDProvider:
    """Compute dH/dR and dS/dR by central finite differences."""

    def __init__(
        self,
        system,
        displacement: float = 1e-3,
        use_scc: bool = False,
        derivative_mode: str = "row",
    ):
        if use_scc:
            raise NotImplementedError("SCC-corrected electron-phonon coupling is not supported in v1.")
        if derivative_mode not in {"row", "full"}:
            raise ValueError("derivative_mode must be either 'row' or 'full'.")
        displacement = validate_finite_positive_scalar(displacement, "displacement")
        self.system = system
        self.displacement = float(displacement)
        self.use_scc = use_scc
        self.derivative_mode = derivative_mode

    def compute(self, kpoints: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.derivative_mode == "row":
            return self._compute_row_derivatives(kpoints)
        return self._compute_full_derivatives(kpoints)

    def _compute_full_derivatives(self, kpoints: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        kpoints = normalize_kpoints(kpoints)
        atoms0 = self.system.atoms.copy()
        num_atoms = len(atoms0)
        hk0, sk0 = self.system.get_hk(k_points=kpoints, use_scc=self.use_scc)
        num_orbitals = _strip_batch(as_array(hk0)).shape[-1]
        h_derivatives = np.zeros((kpoints.shape[0], num_atoms, 3, num_orbitals, num_orbitals), dtype=complex)
        overlap_derivatives = None if sk0 is None else np.zeros_like(h_derivatives)

        try:
            for atom_index in range(num_atoms):
                for cartesian_direction in range(3):
                    plus = atoms0.copy()
                    minus = atoms0.copy()
                    plus.positions[atom_index, cartesian_direction] += self.displacement
                    minus.positions[atom_index, cartesian_direction] -= self.displacement

                    self.system.set_atoms(plus)
                    h_plus, s_plus = self.system.get_hk(k_points=kpoints, use_scc=self.use_scc)
                    self.system.set_atoms(minus)
                    h_minus, s_minus = self.system.get_hk(k_points=kpoints, use_scc=self.use_scc)

                    h_derivatives[:, atom_index, cartesian_direction] = (
                        _strip_batch(as_array(h_plus)) - _strip_batch(as_array(h_minus))
                    ) / (2.0 * self.displacement)
                    if overlap_derivatives is not None:
                        if s_plus is None or s_minus is None:
                            raise RuntimeError("Overlap disappeared during finite-difference calculation.")
                        overlap_derivatives[:, atom_index, cartesian_direction] = (
                            _strip_batch(as_array(s_plus)) - _strip_batch(as_array(s_minus))
                        ) / (2.0 * self.displacement)
        finally:
            self.system.set_atoms(atoms0)

        return h_derivatives, overlap_derivatives

    def _compute_row_derivatives(self, kpoints: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        kpoints = normalize_kpoints(kpoints)
        atoms0 = self.system.atoms.copy()
        orbital_slices = normalize_orbital_slices(orbital_slices_from_system(self.system))
        num_atoms = len(atoms0)
        if len(orbital_slices) != num_atoms:
            raise RuntimeError("Cannot infer atom-resolved orbital slices for finite-difference EPC.")
        num_orbitals = orbital_slices[-1].stop

        _, s0_blocks = self.system.calculator.get_hr(self.system.data)
        has_overlap = s0_blocks is not None
        h_derivatives = np.zeros((kpoints.shape[0], 3, num_orbitals, num_orbitals), dtype=complex)
        overlap_derivatives = np.zeros_like(h_derivatives) if has_overlap else None

        try:
            for atom_index, orbital_slice in enumerate(orbital_slices):
                for cartesian_direction in range(3):
                    plus = atoms0.copy()
                    minus = atoms0.copy()
                    plus.positions[atom_index, cartesian_direction] += self.displacement
                    minus.positions[atom_index, cartesian_direction] -= self.displacement

                    self.system.set_atoms(plus)
                    h_plus_blocks, s_plus_blocks = self.system.calculator.get_hr(self.system.data)
                    h_plus = assemble_directed_hk_from_blocks(h_plus_blocks, kpoints, orbital_slices, num_orbitals)

                    self.system.set_atoms(minus)
                    h_minus_blocks, s_minus_blocks = self.system.calculator.get_hr(self.system.data)
                    h_minus = assemble_directed_hk_from_blocks(h_minus_blocks, kpoints, orbital_slices, num_orbitals)

                    h_delta = (h_plus - h_minus) / (2.0 * self.displacement)
                    h_derivatives[:, cartesian_direction, orbital_slice, :] = h_delta[:, orbital_slice, :]

                    if overlap_derivatives is not None:
                        if s_plus_blocks is None or s_minus_blocks is None:
                            raise RuntimeError("Overlap disappeared during finite-difference calculation.")
                        s_plus = assemble_directed_hk_from_blocks(s_plus_blocks, kpoints, orbital_slices, num_orbitals)
                        s_minus = assemble_directed_hk_from_blocks(s_minus_blocks, kpoints, orbital_slices, num_orbitals)
                        overlap_delta = (s_plus - s_minus) / (2.0 * self.displacement)
                        overlap_derivatives[:, cartesian_direction, orbital_slice, :] = overlap_delta[:, orbital_slice, :]
        finally:
            self.system.set_atoms(atoms0)

        return h_derivatives, overlap_derivatives


class SupercellFD:
    """Finite-difference row derivatives from a phonon supercell."""

    def __init__(
        self,
        system,
        supercell_atoms,
        primitive_to_supercell_atom: np.ndarray,
        supercell_to_primitive_atom: np.ndarray,
        supercell_atom_to_cell: np.ndarray,
        primitive_orbital_offsets: np.ndarray,
        supercell_orbital_offsets: np.ndarray,
        shortest_vectors: np.ndarray,
        vector_multiplicity: np.ndarray,
        displacement: float = 1e-3,
        use_scc: bool = False,
    ):
        if use_scc:
            raise NotImplementedError("SCC-corrected electron-phonon coupling is not supported in v1.")
        displacement = validate_finite_positive_scalar(displacement, "displacement")
        self.system = system
        self.supercell_atoms = supercell_atoms.copy()
        self.primitive_to_supercell_atom = normalize_integer_array(
            primitive_to_supercell_atom,
            "primitive_to_supercell_atom",
        )
        self.supercell_to_primitive_atom = normalize_integer_array(
            supercell_to_primitive_atom,
            "supercell_to_primitive_atom",
        )
        self.supercell_atom_to_cell = normalize_integer_array(supercell_atom_to_cell, "supercell_atom_to_cell")
        self.primitive_orbital_offsets = normalize_integer_array(primitive_orbital_offsets, "primitive_orbital_offsets")
        self.supercell_orbital_offsets = normalize_integer_array(supercell_orbital_offsets, "supercell_orbital_offsets")
        self.shortest_vectors = np.asarray(shortest_vectors, dtype=float)
        self.vector_multiplicity = normalize_integer_array(vector_multiplicity, "vector_multiplicity")
        _validate_supercell_mapping_inputs(
            num_supercell_atoms=len(self.supercell_atoms),
            primitive_to_supercell_atom=self.primitive_to_supercell_atom,
            supercell_to_primitive_atom=self.supercell_to_primitive_atom,
            supercell_atom_to_cell=self.supercell_atom_to_cell,
            primitive_orbital_offsets=self.primitive_orbital_offsets,
            supercell_orbital_offsets=self.supercell_orbital_offsets,
            shortest_vectors=self.shortest_vectors,
            vector_multiplicity=self.vector_multiplicity,
        )
        self.displacement = float(displacement)
        self.use_scc = use_scc
        self._supercell_h_derivatives = None
        self._supercell_overlap_derivatives = None

    @classmethod
    def from_phonopy(
        cls,
        system,
        phonopy_obj,
        displacement: float = 1e-3,
        length_unit: str = "angstrom",
        use_scc: bool = False,
    ) -> "SupercellFD":
        supercell = phonopy_obj.supercell
        length_scale = _length_unit_scale_to_angstrom(length_unit)

        supercell_atoms = Atoms(
            symbols=list(supercell.symbols),
            scaled_positions=np.asarray(supercell.scaled_positions, dtype=float),
            cell=np.asarray(supercell.cell, dtype=float) * length_scale,
            pbc=True,
        )

        (
            primitive_to_supercell_atom,
            supercell_to_primitive_atom,
            supercell_atom_to_cell,
            shortest_vectors,
            vector_multiplicity,
        ) = _phonopy_supercell_maps(phonopy_obj)
        primitive_symbols = [supercell.symbols[int(atom)] for atom in primitive_to_supercell_atom]
        primitive_orbital_counts = _orbital_counts_from_symbols(system, primitive_symbols)
        supercell_orbital_counts = _orbital_counts_from_symbols(system, list(supercell.symbols))
        primitive_orbital_offsets = np.insert(np.cumsum(primitive_orbital_counts), 0, 0)
        supercell_orbital_offsets = np.insert(np.cumsum(supercell_orbital_counts), 0, 0)

        return cls(
            system=system,
            supercell_atoms=supercell_atoms,
            primitive_to_supercell_atom=primitive_to_supercell_atom,
            supercell_to_primitive_atom=supercell_to_primitive_atom,
            supercell_atom_to_cell=supercell_atom_to_cell,
            primitive_orbital_offsets=primitive_orbital_offsets,
            supercell_orbital_offsets=supercell_orbital_offsets,
            shortest_vectors=shortest_vectors,
            vector_multiplicity=vector_multiplicity,
            displacement=displacement,
            use_scc=use_scc,
        )

    def compute(self, kpoints: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        kpoints = normalize_kpoints(kpoints)
        supercell_h_derivatives, supercell_overlap_derivatives = self._compute_supercell_derivatives()
        atoms0 = self.system.atoms.copy()
        num_orbitals = int(self.primitive_orbital_offsets[len(self.primitive_to_supercell_atom)])
        h_derivatives = np.zeros((kpoints.shape[0], 3, num_orbitals, num_orbitals), dtype=complex)
        overlap_derivatives = None if supercell_overlap_derivatives is None else np.zeros_like(h_derivatives)

        try:
            self.system.set_atoms(self.supercell_atoms)
            for primitive_atom_index, _ in enumerate(self.primitive_to_supercell_atom):
                row_slice = slice(
                    self.primitive_orbital_offsets[primitive_atom_index],
                    self.primitive_orbital_offsets[primitive_atom_index + 1],
                )
                for cartesian_direction in range(3):
                    h_rows = self._fourier_row_derivative(
                        supercell_h_derivatives[primitive_atom_index, cartesian_direction],
                        primitive_atom_index,
                        kpoints,
                    )
                    h_derivatives[:, cartesian_direction, row_slice, :] = h_rows[:, row_slice, :]

                    if overlap_derivatives is not None:
                        overlap_rows = self._fourier_row_derivative(
                            supercell_overlap_derivatives[primitive_atom_index, cartesian_direction],
                            primitive_atom_index,
                            kpoints,
                        )
                        overlap_derivatives[:, cartesian_direction, row_slice, :] = overlap_rows[:, row_slice, :]
        finally:
            self.system.set_atoms(atoms0)

        return h_derivatives, overlap_derivatives

    def _compute_supercell_derivatives(self):
        if self._supercell_h_derivatives is not None:
            return self._supercell_h_derivatives, self._supercell_overlap_derivatives

        atoms0 = self.system.atoms.copy()
        supercell_norb = int(self.supercell_orbital_offsets[-1])
        supercell_h_derivatives = np.zeros(
            (len(self.primitive_to_supercell_atom), 3, supercell_norb, supercell_norb),
            dtype=complex,
        )
        supercell_overlap_derivatives = None

        try:
            self.system.set_atoms(self.supercell_atoms)
            for primitive_atom_index, supercell_atom_index in enumerate(self.primitive_to_supercell_atom):
                for cartesian_direction in range(3):
                    plus = self.supercell_atoms.copy()
                    minus = self.supercell_atoms.copy()
                    plus.positions[supercell_atom_index, cartesian_direction] += self.displacement
                    minus.positions[supercell_atom_index, cartesian_direction] -= self.displacement

                    self.system.set_atoms(plus)
                    h_plus, s_plus = self.system.get_hk(
                        k_points=np.array([[0.0, 0.0, 0.0]]),
                        use_scc=self.use_scc,
                    )
                    self.system.set_atoms(minus)
                    h_minus, s_minus = self.system.get_hk(
                        k_points=np.array([[0.0, 0.0, 0.0]]),
                        use_scc=self.use_scc,
                    )

                    supercell_h_derivatives[primitive_atom_index, cartesian_direction] = (
                        strip_single_k_matrix(as_array(h_plus)) - strip_single_k_matrix(as_array(h_minus))
                    ) / (2.0 * self.displacement)
                    if s_plus is not None and s_minus is not None:
                        if supercell_overlap_derivatives is None:
                            supercell_overlap_derivatives = np.zeros_like(supercell_h_derivatives)
                        supercell_overlap_derivatives[primitive_atom_index, cartesian_direction] = (
                            strip_single_k_matrix(as_array(s_plus)) - strip_single_k_matrix(as_array(s_minus))
                        ) / (2.0 * self.displacement)
        finally:
            self.system.set_atoms(atoms0)

        self._supercell_h_derivatives = supercell_h_derivatives
        self._supercell_overlap_derivatives = supercell_overlap_derivatives
        return self._supercell_h_derivatives, self._supercell_overlap_derivatives

    def _fourier_row_derivative(
        self,
        supercell_derivative: np.ndarray,
        primitive_atom_index: int,
        kpoints: np.ndarray,
    ) -> np.ndarray:
        num_orbitals = int(self.primitive_orbital_offsets[len(self.primitive_to_supercell_atom)])
        out = np.zeros((kpoints.shape[0], num_orbitals, num_orbitals), dtype=complex)
        source_supercell_atom = int(self.primitive_to_supercell_atom[primitive_atom_index])
        source_atom = int(self.supercell_to_primitive_atom[source_supercell_atom])
        source_cell = int(self.supercell_atom_to_cell[source_supercell_atom])
        row_super = slice(
            self.supercell_orbital_offsets[source_supercell_atom],
            self.supercell_orbital_offsets[source_supercell_atom + 1],
        )
        row_primitive = slice(
            self.primitive_orbital_offsets[source_atom],
            self.primitive_orbital_offsets[source_atom + 1],
        )

        for target_supercell_atom in range(len(self.supercell_to_primitive_atom)):
            target_atom = int(self.supercell_to_primitive_atom[target_supercell_atom])
            target_cell = int(self.supercell_atom_to_cell[target_supercell_atom])
            multiplicity = int(self.vector_multiplicity[target_cell, source_atom])
            vectors = self.shortest_vectors[
                target_cell,
                source_atom,
                :multiplicity,
            ]
            phase = np.exp(2j * np.pi * (kpoints @ vectors.T)).sum(axis=1) / multiplicity
            col_super = slice(
                self.supercell_orbital_offsets[target_supercell_atom],
                self.supercell_orbital_offsets[target_supercell_atom + 1],
            )
            col_primitive = slice(
                self.primitive_orbital_offsets[target_atom],
                self.primitive_orbital_offsets[target_atom + 1],
            )
            out[:, row_primitive, col_primitive] += (
                supercell_derivative[row_super, col_super][None, :, :] * phase[:, None, None]
            )
        return out


def _strip_batch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _length_unit_scale_to_angstrom(length_unit: str) -> float:
    if not isinstance(length_unit, str):
        raise ValueError("length_unit must be 'angstrom' or 'bohr'.")
    normalized = length_unit.lower()
    if normalized in {"angstrom", "ang", "a"}:
        return 1.0
    if normalized in {"bohr", "au", "a.u.", "atomic_unit"}:
        return 0.529177249
    raise ValueError("length_unit must be 'angstrom' or 'bohr'.")


def _validate_supercell_mapping_inputs(
    num_supercell_atoms: int,
    primitive_to_supercell_atom: np.ndarray,
    supercell_to_primitive_atom: np.ndarray,
    supercell_atom_to_cell: np.ndarray,
    primitive_orbital_offsets: np.ndarray,
    supercell_orbital_offsets: np.ndarray,
    shortest_vectors: np.ndarray,
    vector_multiplicity: np.ndarray,
) -> None:
    if primitive_to_supercell_atom.ndim != 1 or primitive_to_supercell_atom.size == 0:
        raise ValueError("primitive_to_supercell_atom must be a one-dimensional non-empty array.")
    if supercell_to_primitive_atom.shape != (num_supercell_atoms,):
        raise ValueError("supercell_to_primitive_atom must have one entry per supercell atom.")
    if supercell_atom_to_cell.shape != (num_supercell_atoms,):
        raise ValueError("supercell_atom_to_cell must have one entry per supercell atom.")
    if np.any(primitive_to_supercell_atom < 0) or np.any(primitive_to_supercell_atom >= num_supercell_atoms):
        raise ValueError("primitive_to_supercell_atom contains an index outside the supercell atom range.")

    num_primitive_atoms = primitive_to_supercell_atom.shape[0]
    if np.any(supercell_to_primitive_atom < 0) or np.any(supercell_to_primitive_atom >= num_primitive_atoms):
        raise ValueError("supercell_to_primitive_atom contains an index outside the primitive atom range.")
    if primitive_orbital_offsets.shape != (num_primitive_atoms + 1,):
        raise ValueError("primitive_orbital_offsets must have length nprimitive + 1.")
    if supercell_orbital_offsets.shape != (num_supercell_atoms + 1,):
        raise ValueError("supercell_orbital_offsets must have length nsupercell + 1.")
    _validate_orbital_offsets(primitive_orbital_offsets, "primitive_orbital_offsets")
    _validate_orbital_offsets(supercell_orbital_offsets, "supercell_orbital_offsets")

    if vector_multiplicity.ndim != 2:
        raise ValueError("vector_multiplicity must have shape (ncells, nprimitive).")
    if shortest_vectors.ndim != 4 or shortest_vectors.shape[-1] != 3:
        raise ValueError("shortest_vectors must have shape (ncells, nprimitive, max_multiplicity, 3).")
    if shortest_vectors.shape[:2] != vector_multiplicity.shape:
        raise ValueError("shortest_vectors and vector_multiplicity must agree on (ncells, nprimitive).")
    if not np.all(np.isfinite(shortest_vectors)):
        raise ValueError("shortest_vectors must contain finite values.")
    if np.any(vector_multiplicity <= 0) or np.any(vector_multiplicity > shortest_vectors.shape[2]):
        raise ValueError("vector_multiplicity must be positive and no larger than shortest_vectors max multiplicity.")
    if np.any(supercell_atom_to_cell < 0) or np.any(supercell_atom_to_cell >= vector_multiplicity.shape[0]):
        raise ValueError("supercell_atom_to_cell contains an index outside the cell range.")
    if vector_multiplicity.shape[1] != num_primitive_atoms:
        raise ValueError("vector_multiplicity primitive axis must match primitive_to_supercell_atom.")


def _validate_orbital_offsets(offsets: np.ndarray, name: str) -> None:
    if offsets.ndim != 1 or offsets[0] != 0:
        raise ValueError(f"{name} must be a one-dimensional array starting at 0.")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")


def _phonopy_supercell_maps(phonopy_obj):
    try:
        from phonopy.structure.cells import get_smallest_vectors
    except ImportError as exc:
        raise ImportError("phonopy is required to build SupercellFD from a phonopy object.") from exc

    primitive = phonopy_obj.primitive
    supercell = phonopy_obj.supercell
    n_supercells = int(round(abs(np.linalg.det(supercell.supercell_matrix))))
    n_primitive_atoms = len(primitive)

    supercell_atom_to_cell = np.tile(np.arange(n_supercells), n_primitive_atoms)
    primitive_map = supercell.u2u_map
    supercell_to_primitive_atom = np.array(
        [primitive_map[atom] for atom in supercell.s2u_map],
        dtype=int,
    )

    primitive_to_supercell_atom = np.zeros((n_primitive_atoms,), dtype=int)
    primitive_to_supercell_atom[0] = (n_supercells - 1) // 2
    for atom_index in range(1, n_primitive_atoms):
        primitive_to_supercell_atom[atom_index] = primitive_to_supercell_atom[atom_index - 1] + n_supercells

    shortest_vectors, vector_multiplicity = get_smallest_vectors(
        supercell.cell,
        supercell.scaled_positions,
        supercell.scaled_positions[primitive_to_supercell_atom],
    )
    trans_mat_float = np.dot(supercell.cell, np.linalg.inv(primitive.cell))
    trans_mat = np.rint(trans_mat_float).astype(int)
    if not (np.abs(trans_mat_float - trans_mat) < 1e-8).all():
        raise ValueError("Could not infer integer supercell-to-primitive transform from phonopy cells.")
    shortest_vectors = np.asarray(np.dot(shortest_vectors, trans_mat), dtype=float, order="C")

    return (
        primitive_to_supercell_atom,
        supercell_to_primitive_atom,
        supercell_atom_to_cell,
        shortest_vectors,
        np.asarray(vector_multiplicity, dtype=int),
    )


def _orbital_counts_from_symbols(system, symbols) -> np.ndarray:
    orbital_info = system.calculator.get_orbital_info()
    try:
        return np.asarray([len(orbital_info[symbol]) for symbol in symbols], dtype=int)
    except KeyError as exc:
        raise KeyError(f"Element {exc} found in phonopy object but missing from DeePTB basis.") from exc
