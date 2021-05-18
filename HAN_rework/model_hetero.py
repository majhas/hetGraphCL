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

class ContrastLayer(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, temperature):
        super(ContrastLayer, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )

        self.project.apply(weight_reset)

        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, feat, pos, neg):
        z = self.project(feat)
        pos = self.project(pos)
        neg = self.project(neg)
        l_pos = F.cosine_similarity(z, pos, dim=2).flatten().unsqueeze(1) # NM * 1
        l_neg = F.cosine_similarity(z.expand(neg.shape), neg, dim=3).flatten(1, 2).transpose(0, 1) # NM * K sample
        # l_neg = F.cosine_similarity(z, neg, dim=2).flatten().unsqueeze(1) # NM * 1 sample1
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits /= self.temperature
        loss = self.cross_entropy_loss(logits, torch.zeros(logits.size(0), dtype=torch.long,
                                                        device=feat.device))
        return z, loss

    def sample(self, g, feat, att, num_neg=64):
        u, v, eid = g.edges(form='all', order='srcdst')
        att = att.mean(dim=1)[eid].squeeze()
        pos_id = [0] * feat.shape[0]
        _, counts = u.unique(return_counts=True)
        neighbors = torch.split_with_sizes(v, tuple(counts))
        for i, item in enumerate(torch.split_with_sizes(att, tuple(counts))):
            # res, ind = torch.topk(item, min(2, item.shape[0]), dim=0)
            # print(res[0])
            # if i > 0:
            #     if ind[0] != i:
            #         pos_id[i] = v[ind[0] + counts[i-1]]
            #     else:
            #         pos_id[i] = v[ind[-1] + counts[i-1]]
            # else:
            #     if ind[0] != i:
            #         pos_id[i] = v[ind[0]]
            #     else:
            #         pos_id[i] = v[ind[-1]]
            one_hot = F.gumbel_softmax(item, tau=1, hard=True).type(torch.bool)
            pos_id[i] = neighbors[i][one_hot]
        pos = feat[pos_id] # N * M * D
        pos_id = torch.LongTensor(pos_id).view(feat.shape[0], -1)
        node_id = torch.arange(0, feat.shape[0])
        negatives = node_id.expand(feat.shape[0], feat.shape[0])
        mask = torch.zeros(negatives.shape).scatter_(1, torch.LongTensor(pos_id), 1)
        diag = torch.eye(feat.shape[0])
        mask = (1 - (mask + diag)).type(torch.bool)
        neg_id = negatives * mask
        sample_id = torch.randperm(feat.shape[0])
        sample_id = sample_id[:num_neg]
        neg_id = neg_id[:, sample_id]
        neg = torch.stack([feat[item] for item in neg_id], dim=1) # N * K * M * D
        return pos, neg

    def sample1(self, g, feat, att):
        u, v, eid = g.edges(form='all', order='srcdst')
        att = att.mean(dim=1)[eid]
        pos_id = [0] * feat.shape[0]
        neg_id = [0] * feat.shape[0]
        uniques, counts = u.unique(return_counts=True)
        for i, item in enumerate(torch.split_with_sizes(att, tuple(counts))):
            _, ind = torch.topk(item, min(2, item.shape[0]), dim=0)
            _, small_ind = torch.topk(item, min(2, item.shape[0]), dim=0, largest=False)
            if i > 0:
                if ind[0] != i:
                    pos_id[i] = v[ind[0] + counts[i-1]]
                else:
                    pos_id[i] = v[ind[-1] + counts[i-1]]
                if small_ind[0] != i:
                    neg_id[i] = v[small_ind[0] + counts[i-1]]
                else:
                    neg_id[i] = v[small_ind[-1] + counts[i-1]]
            else:
                if ind[0] != i:
                    pos_id[i] = v[ind[0]]
                else:
                    pos_id[i] = v[ind[-1]]
                if small_ind[0] != i:
                    neg_id[i] = v[small_ind[0]]
                else:
                    neg_id[i] = v[small_ind[-1]]
        pos = feat[pos_id] # N * M * D
        neg = feat[neg_id] # N * M * D
        return pos, neg

    def train_step(self, g, feat, att, opt, epoch):
        # opt.zero_grad()
        # feat.requires_grad = True
        pos, neg = self.sample(g, feat, att)
        z, loss = self.forward(feat, pos, neg)
        # feat.requires_grad_(False)
        # opt.zero_grad()
        # loss.backward(retain_graph=True)
        # opt.step()
        # print('Epoch {:d} | Train Loss {:.4f}'.format(epoch + 1, loss.item()))
        return z, loss

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
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout, layer_number):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        if layer_number == 0:
            self.con_layers = nn.ModuleList()
        # self.optimizers = []
        for i in range(len(meta_paths)):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu,
                                           allow_zero_in_degree=True))
            if layer_number == 0:
                self.con_layers.append(ContrastLayer(out_size, out_size, out_size, 0.5))
            # self.optimizers.append(torch.optim.Adam(Con_layer.parameters(), lr=0.005, weight_decay=5e-4))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h, opt, layer_number, epoch, get_attention=True, no_loss=False):
        semantic_embeddings = []
        project_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)
        if layer_number == 0:
            loss = 0
        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            feat, att = self.gat_layers[i](new_g, h, get_attention)
            # feat.retain_grad()
            # feat.requires_grad_(True)
            if layer_number == 0:
                z, CLloss = self.con_layers[i].train_step(new_g, feat, att, opt, epoch)
                loss += CLloss
            semantic_embeddings.append(feat.flatten(1))
            project_embeddings.append(z.flatten(1))
        if not no_loss:
            if layer_number == 0:
                loss /= len(self.meta_paths)
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
                print('Outer Epoch {:d} | Train Loss {:.4f}'.format(epoch + 1, loss.item()))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)
        project_embeddings = torch.stack(project_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings), self.semantic_attention(project_embeddings)                          # (N, D * K)

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

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout, 0))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout, l))
        # self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h, opt, epoch, no_loss=False):
        for i, gnn in enumerate(self.layers):
            h, z = gnn(g, h, opt, i, epoch, True, no_loss)

        return h, z
        # return self.predict(h)

    def test_step(self, g, h):
        for i, gnn in enumerate(self.layers):
            h = gnn.test_step(g, h)
        return h
