# this file is for atomic dft calculation
# given atomic configuration, return sk tables.
# using hotcent api.

from ase.units import Bohr, Ha
from hotcent.atomic_dft import AtomicDFT
from hotcent.confinement import PowerConfinement
from hotcent.offsite_twocenter import Offsite2cTable
from dptb.utils.tools import atomic_num_dict
from dptb.nn.sktb.cov_radiiDB import Covalent_radii
from typing import Union,Dict, List
from dptb.nn.sktb.electronic_configDB import electronic_config_dict
from dptb.nn.sktb.builtin_skbasisDB import skbasisDB
from dptb.nn.sktb.builtin_skbasisDB import onsite_e_builtin_basis, occupations_builtin_basis, Hubbard_U_builtin_basis
import logging
log = logging.getLogger(__name__)


class DFT2SKTable(object):
    def __init__(self, xc: str, superposition: str = 'density', scalarrel=True, **kwargs):
        self.xc = xc
        self.superposition = superposition
        self.scalarrel = scalarrel
        self.kwargs = kwargs
        self.atomic_objs = {}
            
    def update_config(self, basis:Dict[List], rw: dict, pw:dict,  rd:dict=None, pd:dict=None):
        atomic_symbols = list(basis.keys())
        # check basis
        for ia in atomic_symbols:
            assert isinstance(basis[ia], list)
            for ib in basis[ia]:
                assert ib in electronic_config_dict[ia]['valence'].keys(), f'the {ia}-{ib} not in electronic_config_dict, check dptb.nn.sktb.electronic_configDB'
        
        if rd is None:
            rd = rw
        if pd is None:
            pd = pw
        for ia in atomic_symbols:
            if ia in self.atomic_objs:
                # run_completed 判断是否运行atomdft.run()
                assert 'run_completed' in self.atomic_objs[ia]
                if rd[ia] == self.atomic_objs[ia]['rd'] and pd[ia] == self.atomic_objs[ia]['pd'] and \
                    rw[ia] == self.atomic_objs[ia]['rw'] and pw[ia] == self.atomic_objs[ia]['pw']:
                    continue
            else:
                self.atomic_objs[ia] = {}

            atomic_config = electronic_config_dict[ia]
            electronic_config = atomic_config['atomic_core'] + " " + " ".join(list(atomic_config['valence'].keys()))
            valences = skbasisDB[ia]
            
            confinement_density = PowerConfinement(r0=rd[ia], s=pd[ia]) 
            confinement_wavefunc = {}
            for val in valences:  # set confinement for each valence
                confinement_wavefunc[val] = PowerConfinement(r0=(rw[ia]), s=pw[ia])
            
            atomdft = AtomicDFT(ia,
                            xc=self.xc,
                            configuration=electronic_config,
                            valence=valences,
                            scalarrel=True,
                            confinement=confinement_density,
                            wf_confinement=confinement_wavefunc,
                            txt='-',
                            )
            
            # constuct atomic_objs
            onsite_e = onsite_e_builtin_basis(ia, basis[ia],unit='Ha')
            occupations = occupations_builtin_basis(ia, basis[ia])
            hubbardvalues = Hubbard_U_builtin_basis(ia, basis[ia],unit='Ha')

            self.atomic_objs[ia].update({'basis': basis[ia], 
                                            'rd': rd[ia],
                                            'pd': pd[ia],
                                            'rw': rw[ia],
                                            'pw': pw[ia],                                            
                                            'atomdft': atomdft, 
                                            'run_completed': False,
                                            'eigenvalues': onsite_e[ia],
                                            'occupations': occupations[ia],
                                            'hubbardvalues':hubbardvalues[ia]}
                                        )
            # U values need 
    def run_atomic_dft(self):
        for ia in self.atomic_objs:
            if self.atomic_objs[ia]['run_completed']:
                continue
            self.atomic_objs[ia]['atomdft'].run()
            self.atomic_objs[ia]['run_completed'] = True
            #self.atomic_objs[ia]['atomdft'].info = {'hubbardvalues': {}}
            #self.atomic_objs[ia]['atomdft'].info['hubbardvalues'] = self.atomic_objs[ia]['hubbardvalues']
            #self.atomic_objs[ia]['atomdft'].info['occupations'] = self.atomic_objs[ia]['occupations']
            #self.atomic_objs[ia]['atomdft'].info['eigenvalues'] = self.atomic_objs[ia]['eigenvalues']

    def get_skf_pair(self, atom_a:str, atom_b:str=None, rmin=0.4, dr=0.02, N = 800, stride=1):
        if atom_b is None:
            atom_b = atom_a
        log.info(f'getting {atom_a}-{atom_b} sk table')
        assert atom_a in self.atomic_objs and atom_b in self.atomic_objs
        
        for ia in [atom_a, atom_b]:
            if not self.atomic_objs[ia]['run_completed']:
                self.atomic_objs[ia]['atomdft'].run()
                self.atomic_objs[ia]['run_completed'] = True
        log.info(f'finished runing the atomicdfts ....')

        off2c = Offsite2cTable(self.atomic_objs[atom_a]['atomdft'], self.atomic_objs[atom_b]['atomdft'])
        off2c.run(rmin, dr, N, superposition=self.superposition,
                         xc=self.xc,stride=stride, smoothen_tails=True)
        log.info(f'finished runing sk tables calculations on r-grids....')
        # Write the SK tables without repulsion (only electronic part)
        # for homo-case, the atomic eigenvalues, Hubbardvalues, occupations are also saved 
        # but the spe as well as the (spin-polarization error) is set to 0.0.
    
        if atom_a == atom_b:    
            sk.write(eigenvalues=self.atomic_objs[atom_a]['eigenvalues'],
                     hubbardvalues=self.atomic_objs[atom_a]['hubbardvalues'],
                     occupations=self.atomic_objs[atom_a]['occupations'], 
                     spe=0.
                    )
        else:
            sk.write()
        log.info(f'finished writing sk tables....')
    
    def get_skfs(self, basis:Dict[List], rw: dict, pw:dict,  rd:dict=None, pd:dict=None):
        self.update_config(basis, rd, pw, rw, pd)
        self.run_atomic_dft()
        atom_symbols = list(self.atomic_objs.keys())
        for ia in range(len(atom_symbols)):
            atoms_a = atom_symbols[ia]
            for atoms_b in atom_symbols[ia:]:
                self.get_skf_pair(atoms_a, atoms_b)
        
        