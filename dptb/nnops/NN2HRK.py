import torch
import numpy as np
from dptb.structure.structure import BaseStruct
from dptb.dataprocess.processor import Processor
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from ase import Atoms
from dptb.utils.tools import  nnsk_correction

class NN2HRK(object):
    def __init__(self, apihost, mode):
        assert mode in ['nnsk', 'dptb']
        self.apihost = apihost
        self.mode = mode
        self.hamileig = HamilEig(dtype=torch.float32)
        
        self.if_nn_HR_ready = False
        self.if_dp_HR_ready = False
    
        ## parameters.
        self.device = apihost.model_config['device']
        self.dtype =  apihost.model_config['dtype']
            
        self.sorted_onsite="st"
        self.sorted_bond="st"
        self.sorted_env="itype-jtype"

    def update_struct(self,structure):
        # update status is the structure is update.
        if isinstance(structure, BaseStruct):
            self.structure = structure
        elif isinstance(structure,Atoms):
            struct = BaseStruct(atom=structure, format='ase', cutoff=self.apihost.model_config['bond_cutoff'], proj_atom_anglr_m=self.apihost.model_config['proj_atom_anglr_m'], proj_atom_neles=self.apihost.model_config['proj_atom_neles'], onsitemode=self.apihost.model_config['onsitemode'], time_symm=self.apihost.model_config['time_symm'])
            self.structure = struct
        else:
            raise ValueError("Invalid structure type: %s" % type(structure))
        
        self.time_symm = self.structure.time_symm
        self.if_dp_HR_ready = False
        self.if_nn_HR_ready = False

    def get_HR(self):
        if self.mode == 'nnsk' and not self.if_nn_HR_ready:
            self._get_nnsk_HR()
            
        if self.mode == 'dptb' and not self.if_dp_HR_ready:
            self._get_dptb_HR()

        return self.allbonds, self.hamil_blocks, self.overlap_blocks 
    
    
    def get_HK(self, kpoints):
        assert self.if_nn_HR_ready or self.if_dp_HR_ready, "The HR shoule be calcualted before call for HK." 

        if not self.use_orthogonal_basis:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm)
            skmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='S', time_symm=self.time_symm)
        else:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm)
            skmat = torch.eye(hkmat.shape[1], dtype=torch.complex64).unsqueeze(0).repeat(hkmat.shape[0], 1, 1)
        return hkmat, skmat
    
    def get_eigenvalues(self,kpoints,spindeg=2):
        assert self.if_nn_HR_ready or self.if_dp_HR_ready, "The HR shoule be calcualted before call for HK." 
        eigenvalues,_ = self.hamileig.Eigenvalues(kpoints, time_symm=self.time_symm)
        eigks = eigenvalues.detach().numpy()

        num_el = np.sum(self.structure.proj_atom_neles_per)
        nk = len(kpoints)
        numek = num_el * nk // spindeg
        sorteigs =  np.sort(np.reshape(eigks,[-1]))
        EF=(sorteigs[numek] + sorteigs[numek-1])/2
        return eigks, EF

    def _get_nnsk_HR(self):
        assert isinstance(self.structure, BaseStruct)
        assert self.structure.onsitemode == self.apihost.model_config['onsitemode']
        # TODO: 注意检查 processor 关于 env_cutoff 和 onsite_cutoff.
        predict_process = Processor(structure_list=self.structure, batchsize=1, kpoint=None, eigen_list=None, device=self.device, dtype=self.dtype, 
                                        env_cutoff=self.apihost.model_config['env_cutoff'], onsitemode=self.apihost.model_config['onsitemode'], onsite_cutoff=self.apihost.model_config['onsite_cutoff'], sorted_onsite="st", sorted_bond="st", sorted_env="st")

        batch_bonds, batch_bond_onsites = predict_process.get_bond(sorted=self.sorted_bond)
        coeffdict = self.apihost.model(mode='hopping')
        batch_hoppings = self.apihost.hops_fun.get_skhops(batch_bonds=batch_bonds, coeff_paras=coeffdict, rcut=self.apihost.model_config['skfunction']['sk_cutoff'], w=self.apihost.model_config['skfunction']['sk_decay_w'])
        nn_onsiteE, onsite_coeffdict = self.apihost.model(mode='onsite')
        batch_onsiteEs = self.apihost.onsite_fun(batch_bonds_onsite=batch_bond_onsites, onsite_db=self.apihost.onsite_db, nn_onsiteE=nn_onsiteE)
        
        if self.apihost.model_config['onsitemode'] == 'strain':
            batch_onsite_envs = predict_process.get_onsitenv(cutoff=self.apihost.model_config['onsite_cutoff'], sorted=self.sorted_onsite)
            batch_onsiteVs = self.apihost.onsitestrain_fun.get_skhops(batch_bonds=batch_onsite_envs, coeff_paras=onsite_coeffdict)
            onsiteEs, hoppings, onsiteVs = batch_onsiteEs[0], batch_hoppings[0], batch_onsiteVs[0]
            onsitenvs = np.asarray(batch_onsite_envs[0][:,1:])
        else:
            onsiteEs, hoppings, onsiteVs = batch_onsiteEs[0], batch_hoppings[0],  None
            onsitenvs = None

        self.hamileig.update_hs_list(struct=self.structure, hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs)
        self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[0][:,1:]), bonds_hoppings=np.asarray(batch_bonds[0][:,1:]), 
                                    onsite_envs=onsitenvs)
        
        # 同一个类实例, 只能计算一种TB hamiltonian. 
        self.if_nn_HR_ready = True
        self.if_dp_HR_ready = False
        self.use_orthogonal_basis = self.hamileig.use_orthogonal_basis
        self.allbonds, self.hamil_blocks = self.hamileig.all_bonds, self.hamileig.hamil_blocks
        
        if not self.hamileig.use_orthogonal_basis:
            self.overlap_blocks = None
        else:
            self.overlap_blocks = self.hamileig.overlap_blocks
    
    def _get_dptb_HR(self):
        predict_process = Processor(structure_list=self.structure, batchsize=1, kpoint=None, eigen_list=None, device=self.device, dtype=self.dtype, 
                                        env_cutoff=self.apihost.model_config['env_cutoff'], onsitemode=self.apihost.model_config['onsitemode'], onsite_cutoff=self.apihost.model_config['onsite_cutoff'], sorted_onsite="st", sorted_bond="st", sorted_env="st")
        
        batch_bonds, batch_bond_onsites = predict_process.get_bond(sorted=self.sorted_bond)
        batch_env = predict_process.get_env(cutoff=self.apihost.model_config['env_cutoff'], sorted=self.sorted_env)
        batch_bond_hoppings, batch_hoppings, batch_bond_onsites, batch_onsiteEs = self.apihost.nntb.calc(batch_bonds, batch_env)

        if  self.apihost.model_config['use_correction']:
            coeffdict = self.apihost.sknet(mode='hopping')
            batch_nnsk_hoppings = self.apihost.hops_fun.get_skhops( batch_bond_hoppings, coeffdict, 
                            rcut=self.apihost.model_config["skfunction"]["sk_cutoff"], w=self.apihost.model_config["skfunction"]["sk_decay_w"])
            nnsk_onsiteE, onsite_coeffdict = self.apihost.sknet(mode='onsite')
            batch_nnsk_onsiteEs = self.apihost.onsite_fun(batch_bonds_onsite=batch_bond_onsites, onsite_db=self.apihost.onsite_db, nn_onsiteE=nnsk_onsiteE)

            if self.apihost.model_config['onsitemode'] == "strain":
                batch_onsite_envs = predict_process.get_onsitenv(cutoff=self.apihost.model_config['onsite_cutoff'], sorted=self.sorted_onsite)
                batch_nnsk_onsiteVs = self.apihost.onsitestrain_fun.get_skhops(batch_bonds=batch_onsite_envs, coeff_paras=onsite_coeffdict)
                onsiteVs = batch_nnsk_onsiteVs[0]
                onsitenvs = np.asarray(batch_onsite_envs[0][:,1:])
            else:
                onsiteVs = None
                onsitenvs = None

            onsiteEs, hoppings, _, _ = nnsk_correction(nn_onsiteEs=batch_onsiteEs[0], nn_hoppings=batch_hoppings[0],
                                    sk_onsiteEs=batch_nnsk_onsiteEs[0], sk_hoppings=batch_nnsk_hoppings[0],
                                    sk_onsiteSs=None, sk_overlaps=None)
        else:
             onsiteEs, hoppings = batch_onsiteEs[0], batch_hoppings[0]

        
        self.hamileig.update_hs_list(struct=self.structure, hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs)
        self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[0][:,1:]), bonds_hoppings=np.asarray(batch_bond_hoppings[0][:,1:]),
                                    onsite_envs=onsitenvs)

        # 同一个类实例, 只能计算一种TB hamiltonian. 
        self.if_nn_HR_ready = False
        self.if_dp_HR_ready = True
        self.use_orthogonal_basis = self.hamileig.use_orthogonal_basis
        self.allbonds, self.hamil_blocks = self.hamileig.all_bonds, self.hamileig.hamil_blocks

        if not self.hamileig.use_orthogonal_basis:
            self.overlap_blocks = None
        else:
            self.overlap_blocks = self.hamileig.overlap_blocks