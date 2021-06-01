
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.gat_layer import GraphAttentionLayer
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.2):
        super(GAT, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        # self.layers.append(GraphConv(in_dim, hid_dim))
        self.emb = nn.Linear(in_dim, hid_dim)
        self.layers = nn.ModuleList()
        # for _ in range(n_layers-1):
        #     self.layers.append(GraphAttentionLayer(hid_dim, hid_dim, dropout=0.))
        #
        # self.out_layer = GraphAttentionLayer(hid_dim, out_dim, dropout=0., concat=False)
        for _ in range(n_layers-1):
            self.layers.append(GATConv(hid_dim, hid_dim))

        self.out_layer = GATConv(hid_dim, out_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):

        edge_index, values = dense_to_sparse(adj)
        N = adj.shape[0]
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],\
                value=values, sparse_sizes=(N,N)).to(x.device)
        x = self.emb(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, adj)

        x = self.out_layer(x, adj)

        # print(x.shape)
        return x
