"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=1.414)

class linear_eval(nn.Module):
    def __init__(self, in_size, out_size):
        super(linear_eval, self).__init__()

        self.predict = nn.Sequential(nn.Linear(in_size, out_size))

        # nn.init.xavier_normal_(self.predict.weight, gain=1.414)

    def forward(self, h):
        logits = self.predict(h)
        return logits

class ProjectionLayer(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(ProjectionLayer, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

        self.project.apply(weight_reset)


    def forward(self, feat):
        z = self.project(feat)
        return z


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

        self.project.apply(weight_reset)

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout, layer_number, project=False):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.projection_heads = nn.ModuleList()
        self.atts = []

        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
            if project:
                self.projection_heads.append(ProjectionLayer(out_size, out_size, out_size))
            # self.optimizers.append(torch.optim.Adam(Con_layer.parameters(), lr=0.005, weight_decay=5e-4))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h, layer_number, get_attention=True):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)
        if layer_number == 0:
            loss = 0

        projections = []
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            feat, att = self.gat_layers[i](new_g, h, get_attention)
            self.atts.append(att)
            # feat.retain_grad()
            # feat.requires_grad_(True)
            # if layer_number == 0:
            #     z, CLloss = self.con_layers[i].train_step(new_g, feat, att)
            #     loss += CLloss
            if len(self.projection_heads) > 0:
                z = self.projection_heads[i](feat)
                projections.append(z)

            semantic_embeddings.append(feat.flatten(1))
        # if layer_number == 0:
        #     loss /= len(self.meta_paths)
            # opt.zero_grad()
            # loss.backward(retain_graph=True)
            # opt.step()
            # print('Outer Epoch {:d} | Train Loss {:.4f}'.format(epoch + 1, loss.item()))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        if len(self.projection_heads) > 0:
            projections = torch.stack(projections, dim=1).permute(1, 0, 2, 3)
        else:
            projections = None

        return self.semantic_attention(semantic_embeddings), projections                    # (N, D * K)

    def test_step(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings), z                          # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.meta_paths = meta_paths

        self.hid_dim = hidden_size
        self.num_heads = num_heads

        self.layers = nn.ModuleList()

        project = False
        if len(num_heads) == 1:
            project = True

        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout, 0, project))

        for l in range(1, len(num_heads)):
            project = False
            if l == len(num_heads)-1:
                project = True
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout, l, project))
        # self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):

        loss = 0
        for i, gnn in enumerate(self.layers):
            h, z = gnn(g, h, i)

        return h, z

    def test_step(self, g, h):
        for i, gnn in enumerate(self.layers):
            h = gnn.test_step(g, h)
        return h
