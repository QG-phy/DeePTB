import os
from pathlib import Path

import numpy as np

from dptb.postprocess.unified.eph import Phonons


def external_path(env_name, description):
    path = os.environ.get(env_name)
    if not path:
        raise FileNotFoundError(f"Set {env_name} to {description}.")
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"{env_name} does not exist: {resolved}")
    return resolved


def require_file(path, message):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. {message}")
    return path


def regularize_tiny_negative_frequencies(phonons, tol=1e-3):
    """Clip tiny negative phonopy frequencies caused by acoustic numerical noise."""
    frequencies = np.array(phonons.frequencies, copy=True)
    min_frequency = float(np.min(frequencies))
    if min_frequency < -tol:
        raise ValueError(
            f"Found phonon frequency {min_frequency} THz below tolerance {-tol} THz; "
            "this looks like a real imaginary mode, not acoustic numerical noise."
        )
    clipped = int(np.count_nonzero(frequencies < 0.0))
    frequencies[frequencies < 0.0] = 0.0
    if clipped:
        phonons = Phonons(
            qpoints=phonons.qpoints,
            frequencies=frequencies,
            eigenvectors=phonons.eigenvectors,
            masses=phonons.masses,
            cell=phonons.cell,
            scaled_positions=phonons.scaled_positions,
            metadata={
                **phonons.metadata,
                "negative_frequency_regularization": "clipped_to_zero",
                "negative_frequency_tolerance_thz": tol,
                "negative_frequency_min_before_clip_thz": min_frequency,
                "negative_frequency_clipped_count": clipped,
            },
        )
    return phonons
