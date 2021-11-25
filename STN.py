from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import inverse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def AffineTransform(reference, sensed, affine_matrix):
    sensed_grid = F.affine_grid(affine_matrix, sensed.size())
    sensed_tran = F.grid_sample(sensed, sensed_grid)
    a = torch.tensor([[[0, 0, 1]]], dtype=torch.float).to(device)
    a = a.repeat(sensed.size()[0], 1, 1)
    affine_matrix = torch.cat([affine_matrix, a], dim=1)
    inv_affine_matrix = inverse(affine_matrix)
    inv_affine_matrix_1 = inv_affine_matrix[:, 0:2, :]
    reference_grid = F.affine_grid(inv_affine_matrix_1, reference.size())
    reference_inv_tran = F.grid_sample(reference, reference_grid)
    return sensed_tran, reference_inv_tran, inv_affine_matrix
