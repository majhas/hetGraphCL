import torch
import torch.nn as nn

# class GCNLayer(nn.Module):
#     def __init__(self, in_dim, out_dim, bias=True):
#         super(GCNLayer, self).__init__()
#         self.fc = nn.Linear(in_dim, out_dim, bias=False)
#         self.act = nn.PReLU() if act == 'prelu' else act
#
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(out_dim))
#             self.bias.data.fill_(0.0)
#         else:
#             self.register_parameter('bias', None)
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     # Shape of seq: (batch, nodes, features)
#     def forward(self, x, adj, sparse=False):
#
#
#         x = self.fc(x)
#
#         if sparse:
#             out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x, 0)), 0)
#         else:
#             # print(adj.shape)
#             # print(seq_fts.shape)
#             out = torch.mm(adj, x)
#         if self.bias is not None:
#             out += self.bias
#
#         return self.act(out)

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConv(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
