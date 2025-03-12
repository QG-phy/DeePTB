from ase.io import read, write
import numpy as np
from dptb.data import AtomicData, AtomicDataDict
import torch
from dptb.utils.constants import atomic_num_dict, atomic_num_dict_r
import os
from pathlib import Path

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


def test_rmax_float():
    strfile = os.path.join(rootdir, "hBN", "hBN.vasp")
    atoms = read(strfile)
    atomic_options = {}
    atomic_options['pbc'] = True
    atomic_options['r_max'] = 2.6

    data = AtomicData.from_ase(atoms, **atomic_options)
    expected_ind = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    expected_shift = torch.tensor([[-1.,  0.,  0.],
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  1.,  0.],
        [ 1.,  1.,  0.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  0.],
        [ 0., -1.,  0.],
        [-1.,  0.,  0.],
        [ 1., -0., -0.],
        [ 1., -0., -0.],
        [-0., -1., -0.],
        [-0., -1., -0.],
        [-1., -1., -0.],
        [-0., -0., -0.],
        [-1., -1., -0.],
        [-0.,  1., -0.],
        [ 1., -0., -0.]])
        
    exp_Val = torch.cat([expected_ind.T,expected_shift],axis=1)
    tar_val = torch.cat([data.edge_index.T,data.edge_cell_shift],axis=1)
    for ii in tar_val:
        assert ii in exp_Val

def test_rmax_dict_eq():
    strfile = os.path.join(rootdir, "hBN", "hBN.vasp")
    atoms = read(strfile)
    atomic_options = {}
    atomic_options['pbc'] = True
    atomic_options['r_max']  = {'B': 2.6, 'N': 2.6}
    data = AtomicData.from_ase(atoms, **atomic_options)
    expected_ind = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
                                 [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    expected_shift = torch.tensor([[-1.,  0.,  0.],
                                   [-1.,  0.,  0.],
                                   [ 0.,  1.,  0.],
                                   [ 0.,  1.,  0.],
                                   [ 1.,  1.,  0.],
                                   [ 0.,  0.,  0.],
                                   [ 1.,  1.,  0.],
                                   [ 0., -1.,  0.],
                                   [-1.,  0.,  0.],
                                   [ 1., -0., -0.],
                                   [ 1., -0., -0.],
                                   [-0., -1., -0.],
                                   [-0., -1., -0.],
                                   [-1., -1., -0.],
                                   [-0., -0., -0.],
                                   [-1., -1., -0.],
                                   [-0.,  1., -0.],
                                   [ 1., -0., -0.]])
    exp_Val = torch.cat([expected_ind.T,expected_shift],axis=1)
    tar_val = torch.cat([data.edge_index.T,data.edge_cell_shift],axis=1)
    for ii in tar_val:
        assert ii in exp_Val
    
def test_rmax_dict_neq():
    strfile = os.path.join(rootdir, "hBN", "hBN.vasp")
    atoms = read(strfile)
    atomic_options = {}
    atomic_options['pbc'] = True
    atomic_options['r_max']  = {'B':1.5,'N':2.6}
    data = AtomicData.from_ase(atoms, **atomic_options)
    expected_ind = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
                                 [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]])
    expected_shift = torch.tensor([[-1.,  0.,  0.],
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  1.,  0.],
        [ 1.,  1.,  0.],
        [ 0.,  0.,  0.],
        [ 1., -0., -0.],
        [ 1., -0., -0.],
        [-0., -1., -0.],
        [-0., -1., -0.],
        [-1., -1., -0.],
        [-0., -0., -0.]])
    exp_Val = torch.cat([expected_ind.T,expected_shift],axis=1)
    tar_val = torch.cat([data.edge_index.T,data.edge_cell_shift],axis=1)
    for ii in tar_val:
        assert ii in exp_Val