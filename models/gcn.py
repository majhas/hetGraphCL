import torch
import torch.nn as nn

from .layers.gcn_layer import GraphConv
from .layers.proj_head import projection_head

class GCN(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout=0.2):
        super(GCN, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.embed = nn.ModuleList()
        for node_type_dim in self.in_dim:
            self.embed.append(nn.Linear(node_type_dim, hid_dim))

        self.layers = nn.ModuleList()

        for _ in range(n_layers-1):
            self.layers.append(GraphConv(hid_dim, hid_dim))

        self.out_layer = GraphConv(hid_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, adj):

        h = torch.zeros((adj.shape[0], self.hid_dim), device=adj.device)
        start = 0
        for i in range(len(self.in_dim)):
            end = start + x[i].shape[0]
            h[start:end, :] = self.embed[i](x[i])
            start = end

        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h, adj)
            h = self.act(h)

        h = self.out_layer(h, adj)

        return h
