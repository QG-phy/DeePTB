from dptb.nn.sktb.onsiteDB import onsite_energy_database
from dptb.nn.sktb.electronic_configDB import electronic_config_dict
from dptb.nn.sktb.HubbardUDB import Hubbard_U_dict
from dptb.utils.constants import Harte2eV, Ryd2eV

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

def Hubbard_U_builtin_basis(atom:str, basis:list, unit:str='Ha'):
    '''The function `Hubbard_U_builtin_basis` retrieves Hubbard U values for a given atom and basis set in
    different energy units.
    
    Parameters
    ----------
    atom : str
        The `atom` parameter in the `Hubbard_U_builtin_basis` function is a string representing the atomic
    symbol for which you want to retrieve Hubbard U values.
    basis : list
        The `basis` parameter in the `Hubbard_U_builtin_basis` function is a list that contains the basis
    functions for which you want to retrieve Hubbard U values for a specific atom. You can provide a
    list of basis functions as input to the function to get the corresponding Hubbard U values for those
    basis
    unit : str, optional
        The `unit` parameter in the `Hubbard_U_builtin_basis` function specifies the unit in which the
    Hubbard U values will be returned. The function supports three units: 'Ha' (Hartree), 'eV' (electron
    volts), and 'Ry' (Rydberg).
    
    Returns
    -------
        The function `Hubbard_U_builtin_basis` returns a dictionary containing Hubbard U values for the
    specified atom and basis set, converted to the specified energy unit (Ha, eV, or Ry). The dictionary
    has the following structure: `{atom: {basis_set: Hubbard_U_value}}`.
    
    '''
    assert atom in Hubbard_U_dict, f"{atom} not found in Hubbard_U database."
    if 'ha' in unit.lower():
        factor = 1.0
    elif 'ev' in unit.lower():
        factor = Harte2eV
    elif 'ry' in unit.lower():
        factor = Harte2eV / Ryd2eV
    else:
        raise ValueError(f"Unknown unit {unit}.")
    Hubbard_U = {atom:{}}
    for ib in basis:
        if ib in Hubbard_U_dict[atom]:
            Hubbard_U[atom][ib] = Hubbard_U_dict[atom][ib] * factor
        else:
            print(f"{atom} {ib} not found in Hubbard_U database. and set to 0.0")
            Hubbard_U[atom][ib] = 0.0
    return Hubbard_U

def occupations_builtin_basis(atom:str, basis:list):
    '''This function creates a dictionary of occupations for a given atom based on a provided basis.
    
    Parameters
    ----------
    atom : str
        The `atom` parameter in the `occupations_builtin_basis` function is a string that represents the
    atomic symbol of an element. It is used to specify which element's electronic configuration should
    be retrieved.
    basis : list
        It seems like you have not provided the `basis` parameter for the function
    `occupations_builtin_basis(atom:str, basis:list)`. Could you please provide the list of basis
    elements that you want to use in the function?
    
    Returns
    -------
        The function `occupations_builtin_basis` is returning a dictionary containing the occupations of
    each basis function for the given atom. The dictionary is structured as follows: {atom:
    {basis_function: occupation}}.
    
    '''
    occupations = {atom:{}}
    for ib in basis:
        occupations[atom][ib] = electronic_config_dict[atom]['valence'][ib]
    
    return occupations

def onsite_e_builtin_basis(atom:str, basis:list, unit:str='Ha'):
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
    
    if 'ev' in unit.lower():
        factor = 1.0
    elif 'ha' in unit.lower():
        factor = 1.0/Harte2eV
    elif 'ry' in unit.lower():
        factor = 1.0/Ryd2eV
    else:
        raise ValueError(f"Unknown unit {unit}.")

    onsite_e = {atom:{}}
    for ib in basis:
        if ib in onsite_energy_database[atom]:
            onsite_e[atom][ib] =  onsite_energy_database[atom][ib] * factor
        elif val_config[ib] == 0:
            orb = ib[1] + '*'
            if orb in onsite_energy_database[atom]:
                onsite_e[atom][ib] =  onsite_energy_database[atom][orb] * factor
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
