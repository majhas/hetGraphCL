import numpy as np
import scipy.sparse as sp

def add_self_loop(adj):
    return adj + sp.eye(adj.shape[0])

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    ''' Add self loop and normalize adj matrix '''
    adj = add_self_loop(adj)
    adj = normalize_adj(adj)
    return adj
