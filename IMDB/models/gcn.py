import torch
import torch.nn as nn

from .layers.gcn_layer import GCNLayer
from .layers.readout import AvgReadout


class GraphCL(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, act='prelu', dropout=0.2):
        super(GraphCL, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_dim, hid_dim))

        for _ in range(n_layers):
            self.layers.append(GCNLayer(hid_dim, hid_dim))

        self.dropout = nn.Dropout(0.2)

        self.readout = AvgReadout()
        self.projection_head = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim))

    def forward(self, x, adj):

        for layer in self.layers:
            x = layer(x, adj)

        # x = self.readout(x)
        x = self.projection_head(x)
        return x

    def embed(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)

        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size = x1.size(0)
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        # print(x1_abs)
        # print(x2_abs)

        # print('Sim matrix')
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        # print(sim_matrix)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


class GCN_Finetune(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, act='prelu', dropout=0.2):
        super(GCN_Finetune, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_dim, hid_dim))

        for _ in range(n_layers):
            self.layers.append(GCNLayer(hid_dim, hid_dim))

        self.dropout = nn.Dropout(0.2)


        self.embedding = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GCNLayer(hid_dim, hid_dim, nn.ReLU()))

        self.out_layer = nn.Linear(hid_dim, out_dim)

    def forward(self, x, adj, mask=None):

        if mask is not None:
            x = x*mask[0]
            adj = adj*mask[1]

        size = x.shape[0]
        x = x.view(-1, x.shape[2])
        x = self.embedding(x)
        x = self.dropout(x)

        x = x.view(size, -1, x.shape[1])
        for layer in self.layers:
            x = layer(x, adj)

        x = self.out_layer(x)

        return x
