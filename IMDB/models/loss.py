import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_diag_sum(inp):
    inp = torch.diagonal(inp)
    return torch.sum(inp)

def con_loss(x1, x2, temp=0.1):

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix /= temp
    # print(sim_matrix.shape)
    row_softmax_matrix = -F.log_softmax(sim_matrix, dim=1)

    colomn_softmax_matrix = -F.log_softmax(sim_matrix, dim=0)

    row_diag_sum = compute_diag_sum(row_softmax_matrix)
    colomn_diag_sum = compute_diag_sum(colomn_softmax_matrix)
    contrastive_loss = (row_diag_sum + colomn_diag_sum) / (2 * len(row_softmax_matrix))

    return contrastive_loss
