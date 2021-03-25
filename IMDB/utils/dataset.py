import numpy as np
import torch

from torch.utils.data import Dataset

from utils.aug import *

class HetGraphDataset(Dataset):
    def __init__(self, x, adj, node_types, aug_type=None, metapath=None,
                    edge_types=None, aug_ratio=0.2, sparse=False, self_loop=True, device='cpu'):
        self.x = x
        self.adj = adj
        self.node_types = node_types
        self.edge_types = edge_types
        self.aug_type = aug_type
        self.aug_ratio = aug_ratio
        self.metapath = metapath
        self.sparse = sparse
        self.self_loop = self_loop
        self.device = device
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        x = self.x[idx]
        adj = self.adj[idx]
        node_types = self.node_types[idx]

        x, adj = self.apply_aug(x, adj)

        return x, adj

    def apply_aug(self, x, adj):

        # if self.self_loop:
        #     adj = self._add_self_loop(adj)
        # print('before:')
        # print(len(torch.where(torch.sum(adj, dim=1) == 1)[0]))
        # print(len(torch.where(adj == 1)[0]))

        if self.aug_type == None:
            aug_x, aug_adj = x, adj
        elif self.aug_type == 'dropN':
            aug_x, aug_adj = drop_nodes(x, adj, self.aug_ratio)
        elif self.aug_type == 'dropE':
            aug_x, aug_adj = drop_edges(x, adj, self.aug_ratio)
        elif self.aug_type == 'maskN':
            aug_x, aug_adj = mask_nodes(x, adj, self.aug_ratio)
        elif self.aug_type == 'subgraph':
            aug_x, aug_adj = subgraph(x, adj, self.aug_ratio)
        elif self.aug_type == 'dropN_metapath':
            aug_x, aug_adj = drop_node_types(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif self.aug_type == 'dropE_metapath':
            aug_x, aug_adj = drop_edge_types(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif self.aug_type == 'subgraph_metapath':
            aug_x, aug_adj = subgraph_metapath(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif self.aug_type == 'subgraph_metapath_list':
            aug_x, aug_adj = subgraph_metapath_list(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif self.aug_type == 'dropN_not_on_metapath':
            aug_x, aug_adj = drop_node_types(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)
        elif self.aug_type == 'dropE_not_on_metapath':
            aug_x, aug_adj = drop_edge_types(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)
        elif self.aug_type == 'subgraph_not_on_metapath':
            aug_x, aug_adj = subgraph_metapath(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)
        elif self.aug_type == 'subgraph_not_on_metapath_list':
            aug_x, aug_adj = subgraph_metapath_list(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)

        # print('after:')
        # print(len(torch.where(aug_adj == 1)[0]))

        aug_x = aug_x.to(self.device)
        aug_adj = aug_adj.to(self.device)

        aug_adj = self._norm(aug_adj)

        return aug_x, aug_adj

    def collate_fn(self, batch):

        x = self.x[batch]

        adj = self.adj[batch, :]
        adj = adj[:, batch]

        if self.node_types is not None:
            node_types = self.node_types[batch]
        else:
            node_types = None

        return x, adj, node_types


    def _add_self_loop(self, adj):
        return adj + torch.eye(adj.shape[0], device=adj.device)

    def _norm(self, adj):

        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        d_mat_inv_sqrt = torch.diag(deg_inv_sqrt).to(adj.device)
        adj = torch.mm(torch.mm(adj, d_mat_inv_sqrt).transpose(1,0), d_mat_inv_sqrt)

        return adj
