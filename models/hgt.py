import torch
import torch.nn as nn

from .layers.hgt_layer import HGTConv


class HGT(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_node_types, num_edge_types, n_heads, dropout=0.2):
        super(HGT, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types

        self.adapt_layers = nn.ModuleList()
        for _ in range(num_node_types):
            self.adapt_layers.append(nn.Linear(in_dim, hid_dim))

        self.layers = nn.ModuleList()

        for _ in range(n_layers-1):
            self.layers.append(HGTConv(hid_dim, hid_dim, num_node_types, num_edge_types, n_heads))

        self.out_layer = HGTConv(hid_dim, hid_dim, num_node_types, num_edge_types, n_heads, use_norm=False)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x, node_types, edge_index, edge_types):

        res = torch.zeros(x.size(0), self.hid_dim).to(x.device)
        for t_id in range(self.num_node_types):
            idx = (node_types == int(t_id))
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_layers[t_id](x[idx]))
        x = self.dropout(res)
        del res

        for layer in self.layers:
            x = layer(x, node_types, edge_index, edge_types)
            x = self.act(x)

        x = self.out_layer(x, node_types, edge_index, edge_types)

        return x
