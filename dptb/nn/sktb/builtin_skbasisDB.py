from dptb.nn.sktb.onsiteDB import onsite_energy_database
from dptb.nn.sktb.electronic_configDB import electronic_config_dict

skbasisDB={
    "H": [
        "1s"
    ],
    "He": [
        "1s"
    ],
    "Li": [
        "2s",
        "2p"
    ],
    "Be": [
        "2s",
        "2p"
    ],
    "B": [
        "2s",
        "2p"
    ],
    "C": [
        "2s",
        "2p"
    ],
    "N": [
        "2s",
        "2p"
    ],
    "O": [
        "2s",
        "2p"
    ],
    "F": [
        "2s",
        "2p"
    ],
    "Ne": [
        "2s",
        "2p"
    ],
    "Na": [
        "3s",
        "3p"
    ],
    "Mg": [
        "3s",
        "3p",
        "3d"
    ],
    "Al": [
        "3s",
        "3p",
        "3d"
    ],
    "Si": [
        "3s",
        "3p",
        "3d"
    ],
    "P": [
        "3s",
        "3p",
        "3d"
    ],
    "S": [
        "3s",
        "3p",
        "3d"
    ],
    "Cl": [
        "3s",
        "3p",
        "3d"
    ],
    "Ar": [
        "3s",
        "3p",
        "3d"
    ],
    "K": [
        "3d",
        "4s",
        "4p"
    ],
    "Ca": [
        "3d",
        "4s",
        "4p"
    ],
    "Sc": [
        "3d",
        "4s",
        "4p"
    ],
    "Ti": [
        "3d",
        "4s",
        "4p"
    ],
    "V": [
        "3d",
        "4s",
        "4p"
    ],
    "Cr": [
        "3d",
        "4s",
        "4p"
    ],
    "Mn": [
        "3d",
        "4s",
        "4p"
    ],
    "Fe": [
        "3d",
        "4s",
        "4p"
    ],
    "Co": [
        "3d",
        "4s",
        "4p"
    ],
    "Ni": [
        "3d",
        "4s",
        "4p"
    ],
    "Cu": [
        "3d",
        "4s",
        "4p"
    ],
    "Zn": [
        "3d",
        "4s",
        "4p"
    ],
    "Ga": [
        "4s",
        "4p",
        "4d"
    ],
    "Ge": [
        "4s",
        "4p",
        "4d"
    ],
    "As": [
        "4s",
        "4p",
        "4d"
    ],
    "Se": [
        "4s",
        "4p",
        "4d"
    ],
    "Br": [
        "4s",
        "4p",
        "4d"
    ],
    "Kr": [
        "4s",
        "4p",
        "4d"
    ],
    "Rb": [
        "4d",
        "5s",
        "5p"
    ],
    "Sr": [
        "5s",
        "5p",
        "4d"
    ],
    "Y": [
        "4d",
        "5s",
        "5p"
    ],
    "Zr": [
        "4d",
        "5s",
        "5p"
    ],
    "Nb": [
        "4d",
        "5s",
        "5p"
    ],
    "Mo": [
        "4d",
        "5s",
        "5p"
    ],
    "Tc": [
        "4d",
        "5s",
        "5p"
    ],
    "Ru": [
        "4d",
        "5s",
        "5p"
    ],
    "Rh": [
        "4d",
        "5s",
        "5p"
    ],
    "Pd": [
        "4d",
        "5s",
        "5p"
    ],
    "Ag": [
        "4d",
        "5s",
        "5p"
    ],
    "Cd": [
        "4d",
        "5s",
        "5p"
    ],
    "In": [
        "5s",
        "5p",
        "5d"
    ],
    "Sn": [
        "5s",
        "5p",
        "5d"
    ],
    "Sb": [
        "5s",
        "5p",
        "5d"
    ],
    "Te": [
        "5s",
        "5p",
        "5d"
    ],
    "I": [
        "5s",
        "5p",
        "5d"
    ],
    "Xe": [
        "5s",
        "5p",
        "5d"
    ],
    "Cs": [
        "5d",
        "6s",
        "6p"
    ],
    "Ba": [
        "6s",
        "6p",
        "5d"
    ],
    "La": [
        "5d",
        "6s",
        "6p"
    ],
    "Hf": [
        "5d",
        "6s",
        "6p"
    ],
    "Ta": [
        "5d",
        "6s",
        "6p"
    ],
    "W": [
        "5d",
        "6s",
        "6p"
    ],
    "Re": [
        "5d",
        "6s",
        "6p"
    ],
    "Os": [
        "5d",
        "6s",
        "6p"
    ],
    "Ir": [
        "5d",
        "6s",
        "6p"
    ],
    "Pt": [
        "5d",
        "6s",
        "6p"
    ],
    "Au": [
        "5d",
        "6s",
        "6p"
    ],
    "Hg": [
        "6s",
        "6p"
    ],
    "Tl": [
        "6s",
        "6p"
    ],
    "Pb": [
        "6s",
        "6p"
    ],
    "Bi": [
        "6s",
        "6p"
    ],
    "Po": [
        "6s",
        "6p"
    ],
    "At": [
        "6s",
        "6p"
    ],
    "Rn": [
        "6s",
        "6p",
        "6d"
    ]
}

"""
    "Lu": [
        "5d",
        "6s",
        "6p"
    ],
    "Ra": [
        "7s",
        "7p"
    ],
    "Th": [
        "6d",
        "7s",
        "7p"
    ]
"""

def occupations_builtin_basis(atom:str, basis:list):
    occupations = {atom:{}}
    for ib in basis:
        occupations[atom][ib] = electronic_config_dict[atom]['valence'][ib]
    
    return occupations

def onsite_e_builtin_basis(atom:str, basis:list):
    ''' The function `onsite_e_builtin_basis` retrieves onsite energies for a given atom and basis set from
    a database, handling cases where basis orbitals are not directly found.
    Parameters:
    ----------
        atom (str): The chemical symbol of the atom (e.g., 'Si').
        basis (list): List of basis orbitals (e.g., ['3s', '3p']).
    Returns:
    --------
        dict: A nested dictionary mapping the atom to its basis orbitals and their corresponding onsite energies.
              Example: {'Si': {'3s': -10.877726996967032, '3p': -4.161972093023409}}
    Raises:
    -------
        AssertionError: If the atom is not found in the onsite energy database.
        ValueError: If a basis orbital is not found in the onsite energy database or if the basis set is invalid for the atom.
    Notes:
    -----
        - If a basis orbital is not directly found but its valence configuration is zero, the function attempts to find a corresponding
          orbital with a wildcard (e.g., 's*').
    '''
    assert atom in onsite_energy_database, f"{atom} not found in onsite energy database."
    val_config = electronic_config_dict[atom]['valence']
    
    onsite_e = {atom:{}}
    for ib in basis:
        if ib in onsite_energy_database[atom]:
            onsite_e[atom][ib] =  onsite_energy_database[atom][ib]
        elif val_config[ib] == 0:
            orb = ib[1] + '*'
            if orb in onsite_energy_database[atom]:
                onsite_e[atom][ib] =  onsite_energy_database[atom][orb]
            else:
                raise ValueError(f"{atom}-{ib} not found in onsite energy database.")
        else:
            raise ValueError(f"Wrong basis set for {atom}.")
    return onsite_e


def get_onsiteE_all_builtin_basis():
    onsite_e = {}
    for atom, basis in skbasisDB.items():
        onsite_e_atom = onsite_e_builtin_basis(atom, basis)
        onsite_e.update(onsite_e_atom)
    
    return onsite_e

onsiteE_builtin_basis_dict = get_onsiteE_all_builtin_basis()
