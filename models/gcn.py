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

        # self.layers.append(GraphConv(in_dim, hid_dim))
        self.emb = nn.Linear(in_dim, hid_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers-1):
            self.layers.append(GraphConv(hid_dim, hid_dim))

        self.out_layer = GraphConv(hid_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, adj):

        x = self.emb(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, adj)
            x = self.act(x)

        x = self.out_layer(x, adj)

        return x
