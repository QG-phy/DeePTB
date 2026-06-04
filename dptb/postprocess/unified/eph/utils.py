from typing import Sequence, Tuple

import numpy as np
import torch


def as_array(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def normalize_integer_indices(indices, name: str) -> np.ndarray:
    arr = np.atleast_1d(as_array(indices))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a one-dimensional non-empty array.")
    return normalize_integer_array(arr, name)


def normalize_integer_array(indices, name: str) -> np.ndarray:
    arr = as_array(indices)
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices.")
    return arr.astype(int, copy=False)


def normalize_integer_scalar(value, name: str) -> int:
    arr = as_array(value)
    if arr.shape != () or np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be an integer.")
    return int(arr)


def validate_finite_positive_scalar(value, name: str) -> float:
    out = validate_finite_scalar(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return out


def validate_finite_nonnegative_scalar(value, name: str) -> float:
    out = validate_finite_scalar(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return out


def validate_finite_scalar(value, name: str) -> float:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be finite.") from None
    if arr.shape != ():
        raise ValueError(f"{name} must be finite.")
    if np.issubdtype(arr.dtype, np.bool_) or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be finite.")
    out = float(arr)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite.")
    return out


def normalize_kpoints(kpoints) -> np.ndarray:
    arr = as_array(kpoints, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("A single k/q point must have shape (3,).")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("k/q points must have shape (npoints, 3).")
    if arr.shape[0] == 0:
        raise ValueError("k/q points must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("k/q points must be finite.")
    return arr


def reshape_phonopy_eigenvectors(eigenvectors: np.ndarray, natoms: int) -> np.ndarray:
    ev = np.asarray(eigenvectors, dtype=complex)
    if ev.ndim == 4 and ev.shape[2:] == (natoms, 3):
        return ev
    if ev.ndim == 3 and ev.shape[1] == natoms * 3:
        return ev.transpose(0, 2, 1).reshape(ev.shape[0], ev.shape[2], natoms, 3)
    if ev.ndim == 3 and ev.shape[2] == natoms * 3:
        return ev.reshape(ev.shape[0], ev.shape[1], natoms, 3)
    raise ValueError("Unsupported phonopy eigenvector shape.")


def normalize_orbital_slices(orbital_slices):
    normalized = []
    for item in orbital_slices:
        if isinstance(item, slice):
            start = item.start
            stop = item.stop
        else:
            try:
                start, stop = item
            except (TypeError, ValueError):
                raise ValueError("orbital_slices must contain slice objects or (start, stop) pairs.") from None
        if start is None or stop is None:
            raise ValueError("orbital_slices must have explicit start and stop values.")
        start = normalize_integer_scalar(start, "orbital_slices start")
        stop = normalize_integer_scalar(stop, "orbital_slices stop")
        normalized.append(slice(start, stop))
    if not normalized:
        raise ValueError("orbital_slices must be non-empty.")
    expected_start = 0
    for item in normalized:
        if item.start != expected_start or item.stop <= item.start:
            raise ValueError("orbital_slices must be contiguous non-empty ranges starting at 0.")
        expected_start = item.stop
    return normalized


def orbital_slices_from_atom_orbs(atom_orbs: Sequence[str]) -> Sequence[Tuple[int, int]]:
    slices = []
    start = 0
    current_atom = None
    for iorb, label in enumerate(atom_orbs):
        atom_idx = label.split("-", 1)[0]
        if current_atom is None:
            current_atom = atom_idx
        elif atom_idx != current_atom:
            slices.append((start, iorb))
            start = iorb
            current_atom = atom_idx
    if current_atom is not None:
        slices.append((start, len(atom_orbs)))
    return slices


def orbital_slices_from_system(system) -> Sequence[Tuple[int, int]]:
    """Return atom-resolved orbital slices from structured system metadata."""
    if hasattr(system, "atomic_symbols") and hasattr(system, "calculator"):
        orbital_info = system.calculator.get_orbital_info()
        slices = []
        start = 0
        for symbol in system.atomic_symbols:
            try:
                count = len(orbital_info[symbol])
            except KeyError as exc:
                raise KeyError(f"Element {exc} found in system but missing from orbital metadata.") from exc
            slices.append((start, start + count))
            start += count
        return slices

    if hasattr(system, "atom_orbs"):
        return orbital_slices_from_atom_orbs(system.atom_orbs)

    raise RuntimeError("Cannot infer atom-resolved orbital slices for EPC.")


def strip_batch(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def strip_single_k_matrix(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 4 and arr.shape[:2] == (1, 1):
        return arr[0, 0]
    return arr


def assemble_directed_hk_from_blocks(
    blocks,
    kpoints: np.ndarray,
    orbital_slices: Sequence[slice],
    norb: int,
) -> np.ndarray:
    kpoints = normalize_kpoints(kpoints)
    norb = normalize_integer_scalar(norb, "norb")
    if norb <= 0:
        raise ValueError("norb must be a positive integer.")
    orbital_slices = normalize_orbital_slices(orbital_slices)
    if orbital_slices[-1].stop != norb:
        raise ValueError("orbital_slices must cover exactly norb orbitals.")
    hk = np.zeros((kpoints.shape[0], norb, norb), dtype=complex)
    for key, block in blocks.items():
        iatom, jatom, shift = parse_block_key(key)
        if iatom < 0 or jatom < 0:
            raise ValueError(f"Unexpected negative atom index in real-space block key: {key}")
        if iatom >= len(orbital_slices) or jatom >= len(orbital_slices):
            continue
        block_arr = as_array(block, dtype=complex)
        expected_shape = (
            orbital_slices[iatom].stop - orbital_slices[iatom].start,
            orbital_slices[jatom].stop - orbital_slices[jatom].start,
        )
        if block_arr.shape != expected_shape:
            raise ValueError(
                f"Real-space block {key} has shape {block_arr.shape}; expected {expected_shape} from orbital_slices."
            )
        phase = np.exp(-2j * np.pi * (kpoints @ shift)).reshape(-1, 1, 1)
        hk[:, orbital_slices[iatom], orbital_slices[jatom]] += block_arr[None, :, :] * phase
    return hk


def parse_block_key(key: str) -> Tuple[int, int, np.ndarray]:
    parts = key.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected real-space block key: {key}")
    return int(parts[0]), int(parts[1]), np.asarray([int(x) for x in parts[2:]], dtype=float)
