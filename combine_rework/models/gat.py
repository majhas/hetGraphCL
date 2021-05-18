
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from torch_sparse import SparseTensor

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, n_heads=8, dropout=0.2):
        super(GAT, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        # self.layers.append(GraphConv(in_dim, hid_dim))
        self.layers = nn.ModuleList()

        self.emb = nn.Linear(in_dim, hid_dim)
        # self.layers.append(GATConv(in_dim, hid_dim, heads=n_heads))
        for i in range(1, n_layers):
            self.layers.append(GATConv(hid_dim, hid_dim, heads=n_heads))

        if n_layers == 1:
            self.layers.append(GATConv(hid_dim, out_dim, heads=n_heads))
        else:
            self.layers.append(GATConv(n_heads*hid_dim, out_dim, heads=n_heads))

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, adj):

        edge_index, values = dense_to_sparse(adj)
        N = adj.shape[0]
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],\
                value=values, sparse_sizes=(N,N)).to(x.device)

        x = self.emb(x)
        x = self.dropout(x)
        for layer in self.layers:
            x, att = layer(x, adj, return_attention_weights=True)
            x = self.act(x)
            
        return x, att
