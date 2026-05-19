from typing import Sequence, Tuple

import numpy as np
import torch


def as_array(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def normalize_kpoints(kpoints) -> np.ndarray:
    arr = as_array(kpoints, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError("A single k/q point must have shape (3,).")
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("k/q points must have shape (npoints, 3).")
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
            normalized.append(item)
        else:
            start, stop = item
            normalized.append(slice(int(start), int(stop)))
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
    hk = np.zeros((kpoints.shape[0], norb, norb), dtype=complex)
    for key, block in blocks.items():
        iatom, jatom, shift = parse_block_key(key)
        if iatom >= len(orbital_slices) or jatom >= len(orbital_slices):
            continue
        block_arr = as_array(block, dtype=complex)
        phase = np.exp(-2j * np.pi * (kpoints @ shift)).reshape(-1, 1, 1)
        hk[:, orbital_slices[iatom], orbital_slices[jatom]] += block_arr[None, :, :] * phase
    return hk


def parse_block_key(key: str) -> Tuple[int, int, np.ndarray]:
    parts = key.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected real-space block key: {key}")
    return int(parts[0]), int(parts[1]), np.asarray([int(x) for x in parts[2:]], dtype=float)
