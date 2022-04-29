# Convert from Tensorflow implementation 
# https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/bb94b6e889b9851e36a86380b8b35c680ae5484e/ImageAlignment/Codes/tensorDLT.py
import torch
from torch.nn.modules.utils import _pair
import numpy as np

import numpy as np

#######################################################
# Auxiliary matrices used to solve DLT
Aux_M1  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float32)


Aux_M2  = np.array([
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [ 0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ]], dtype=np.float32)



Aux_M3  = np.array([
          [0],
          [1],
          [0],
          [1],
          [0],
          [1],
          [0],
          [1]], dtype=np.float32)



Aux_M4  = np.array([
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float32)


Aux_M5  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ]], dtype=np.float32)



Aux_M6  = np.array([
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ],
          [-1 ],
          [ 0 ]], dtype=np.float32)


Aux_M71 = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float32)


Aux_M72 = np.array([
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [-1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 ,-1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  ,-1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 ,-1 , 0 ]], dtype=np.float32)



Aux_M8  = np.array([
          [0 , 1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 ,-1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 , 1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ]], dtype=np.float32)


Aux_Mb  = np.array([
          [0 ,-1 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [1 , 0 , 0 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , -1  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 1 , 0  , 0 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 ,-1 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 1 , 0 , 0 , 0 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 0 ,-1 ],
          [0 , 0 , 0 , 0  , 0 , 0 , 1 , 0 ]], dtype=np.float32)


def solve_DLT(pts_shift, patch_size=128):
    """Compute H matrix using direct linear transform.
    Args:
        pts_shift: Four points offset. shape [N, 8, 1]
        patch_size: Patch size. if each image has different shape, it should be a [bs, 2, 1] tensor.
    Returns:
        H_mat: Homography matrix. shape [N, 3, 3]
    """
    batch_size = pts_shift.size(0)
    device = pts_shift.device
    if isinstance(patch_size, torch.Tensor):
        tmp = torch.tensor([0., 0., 1., 0., 0., 1., 1., 1.])[None, :, None].expand(batch_size, -1, -1).to(device)
        pts_coord1 = patch_size.repeat(1, 4, 1)*tmp
    elif isinstance(patch_size, (int, float,tuple, list)):
        h, w = _pair(patch_size)
        pts_coord1 = torch.tensor([0., 0., w, 0, 0, h, w, h])[None, :, None].expand(batch_size, -1, -1).to(device)
    else:
        raise ValueError
    pts_coord2 = pts_coord1 + pts_shift

    M1 = torch.from_numpy(Aux_M1)[None, ...].expand(batch_size, -1, -1).to(device)
    M2 = torch.from_numpy(Aux_M2)[None, ...].expand(batch_size, -1, -1).to(device)
    M3 = torch.from_numpy(Aux_M3)[None, ...].expand(batch_size, -1, -1).to(device)
    M4 = torch.from_numpy(Aux_M4)[None, ...].expand(batch_size, -1, -1).to(device)
    M5 = torch.from_numpy(Aux_M5)[None, ...].expand(batch_size, -1, -1).to(device)
    M6 = torch.from_numpy(Aux_M6)[None, ...].expand(batch_size, -1, -1).to(device)
    M71 = torch.from_numpy(Aux_M71)[None, ...].expand(batch_size, -1, -1).to(device)
    M72 = torch.from_numpy(Aux_M72)[None, ...].expand(batch_size, -1, -1).to(device)
    M8 = torch.from_numpy(Aux_M8)[None, ...].expand(batch_size, -1, -1).to(device)
    Mb = torch.from_numpy(Aux_Mb)[None, ...].expand(batch_size, -1, -1).to(device)

    A1 = torch.bmm(M1, pts_coord2)
    A2 = torch.bmm(M2, pts_coord2)
    A3 = M3
    A4 = torch.bmm(M4, pts_coord2)
    A5 = torch.bmm(M5, pts_coord2)
    A6 = M6
    A7 = torch.bmm(M71, pts_coord1) * torch.bmm(M72, pts_coord2)
    A8 = torch.bmm(M71, pts_coord1) * torch.bmm(M8, pts_coord2)

    A_mat = torch.stack([A1.reshape(-1, 8),
                         A2.reshape(-1, 8),
                         A3.reshape(-1, 8),
                         A4.reshape(-1, 8),
                         A5.reshape(-1, 8),
                         A6.reshape(-1, 8),
                         A7.reshape(-1, 8),
                         A8.reshape(-1, 8)], dim=1).permute(0, 2, 1)
    b_mat = torch.bmm(Mb, pts_coord1)
    
    H_8el = torch.linalg.solve(A_mat, b_mat)
    H_9el = torch.cat([H_8el, torch.ones(batch_size, 1, 1).to(device)], dim=1)
    H_mat = H_9el.reshape(batch_size, 3, 3)
    return H_mat

