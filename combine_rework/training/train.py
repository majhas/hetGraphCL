import numpy as np
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import networkx as nx

from torch_geometric.utils import k_hop_subgraph
from .augs import Augmentor

import multiprocessing as mp
from functools import partial

def compute_diag_sum(inp):
    inp = torch.diagonal(inp)
    return torch.sum(inp)

def NTXent_loss(z1, z2, temp=0.1):
    z1_abs = z1.norm(dim=1)
    z2_abs = z2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', z1, z2) / torch.einsum('i,j->ij', z1_abs, z2_abs)
    sim_matrix /= temp

    row_softmax_matrix = -F.log_softmax(sim_matrix, dim=1)
    colomn_softmax_matrix = -F.log_softmax(sim_matrix, dim=0)

    row_diag_sum = compute_diag_sum(row_softmax_matrix)
    colomn_diag_sum = compute_diag_sum(colomn_softmax_matrix)

    loss = (row_diag_sum + colomn_diag_sum) / (2 * len(row_softmax_matrix))

    return loss

def sample(g, z, att, node_types, num_neg=64):

    # row, col, values = att.coo()
    # att = torch.sparse_coo_tensor((row, col), values=values)
    att = torch.mean(att.to_dense(), dim=2)
    N = len(att)
    att[range(N), range(N)] = 0.
    att = att.to_sparse()

    att = att.coalesce()
    edge_index = att.indices()
    values = att.values()

    for edge, att in zip(edge_index.transpose(1,0), values):
        dist = 1./att.item()
        g.add_edge(edge[0].item(), edge[1].item(), distance=dist)

    func = partial(find_nearest, G=g, node_types=node_types)
    with mp.Pool(14) as p:
        pos_samples = p.map(func, range(len(node_types)))

    pos_samples = np.array([s if s is not None else i for i, s in enumerate(pos_samples)])
    pos = z[pos_samples]

    pos_id = torch.LongTensor(pos_samples).view(z.shape[0], -1)
    node_id = torch.arange(0, z.shape[0])
    negatives = node_id.expand(z.shape[0], z.shape[0])

    mask = torch.zeros(negatives.shape).scatter_(1, torch.LongTensor(pos_id), 1)
    diag = torch.eye(z.shape[0])
    mask = (1 - (mask + diag)).type(torch.bool)
    neg_id = negatives * mask
    sample_id = torch.randperm(z.shape[0])
    sample_id = sample_id[:num_neg]
    neg_id = neg_id[:, sample_id]
    neg_samples = torch.stack([z[item] for item in neg_id], dim=1) # N * K * M * D

    return pos, neg_samples

    # pos_samples = [find_nearest(node, g, node_types) for node in range(100)]
    # u, v, eid = g.edges(form='all', order='srcdst')
    # att = att.mean(dim=1)[eid].squeeze()
    # # print(feat.shape)
    # pos_id = [0] * feat.shape[0]
    #
    # _, counts = torch.unique(u, return_counts=True)
    # neighbors = torch.split_with_sizes(v, tuple(counts))
    # for i, item in enumerate(torch.split_with_sizes(att, tuple(counts))):
    #     one_hot = F.gumbel_softmax(item, tau=1, hard=True).type(torch.bool)
    #     # print(f'O: {one_hot}')
    #     # print(f'N: {neighbors[i].shape}')
    #     pos_id[i] = neighbors[i][one_hot]
    # pos = feat[pos_id] # N * M * D
    # pos_id = torch.LongTensor(pos_id).view(feat.shape[0], -1)
    # node_id = torch.arange(0, feat.shape[0])
    # negatives = node_id.expand(feat.shape[0], feat.shape[0])
    # mask = torch.zeros(negatives.shape).scatter_(1, torch.LongTensor(pos_id), 1)
    # diag = torch.eye(feat.shape[0])
    # mask = (1 - (mask + diag)).type(torch.bool)
    # neg_id = negatives * mask
    # sample_id = torch.randperm(feat.shape[0])
    # sample_id = sample_id[:num_neg]
    # neg_id = neg_id[:, sample_id]
    # neg = torch.stack([feat[item] for item in neg_id], dim=1) # N * K * M * D
    # return pos, neg

def att_contrastive_loss(g, z, att, node_types, temp=0.5):

    f = lambda x: torch.exp(x / temp)

    pos, neg = sample(g, z, att, node_types)
    l_pos = F.cosine_similarity(z, pos, dim=1).unsqueeze(1) # intra-positive pairs
    l_neg = F.cosine_similarity(z.expand(neg.shape), neg, dim=2).transpose(1, 0) # NM * K sample
    l_pos = f(l_pos)
    l_neg = f(l_neg).sum(dim=1).unsqueeze(1)

    # print(l_pos.shape)
    # print(l_neg.shape)
    loss = -torch.log((l_pos)/(l_pos+l_neg))

    return loss.mean()

def subset_typeofnode(G, nodetype):
    '''return those nodes in graph G that match type = typestr.'''
    return [name for name, d in G.nodes(data=True)
            if 'type' in d and (d['type'] ==nodetype)]

def find_nearest(fromnode, G, node_types):

    typeofnode = node_types[fromnode]
    #Calculate the length of paths from fromnode to all other nodes
    lengths = nx.single_source_dijkstra_path_length(G, fromnode, weight='distance')

    #We are only interested in a particular type of node
    subnodes = subset_typeofnode(G, typeofnode)
    subdict = {node: lengths[node] for node in subnodes if node in lengths and node != fromnode}

    #return the smallest of all lengths to get to typeofnode
    if subdict: #dict of shortest paths to all entrances/toilets
        nearest =  min(subdict, key=subdict.get) #shortest value among all the keys
        return nearest
    else: #not found, no path from source to typeofnode
        return None

# def to_dgl(x, adj, node_types):
#
#     # _, counts = np.unique(node_types, return_counts=True)
#
#     m_idx = np.where(node_types == 0)[0]
#     d_idx = np.where(node_types == 1)[0]
#     a_idx = np.where(node_types == 2)[0]
#
#     d_vs_m = adj[d_idx][:, m_idx]
#     a_vs_m = adj[a_idx][:, m_idx]
#
#     hg = dgl.heterograph(
#         {
#             ('director', 'dm', 'movie'): torch.nonzero(d_vs_m, as_tuple=True),
#             ('movie', 'md', 'director'): torch.nonzero(d_vs_m.transpose(1, 0), as_tuple=True),
#             ('actor', 'am', 'movie'): torch.nonzero(a_vs_m, as_tuple=True),
#             ('movie', 'ma', 'actor'): torch.nonzero(a_vs_m.transpose(1, 0), as_tuple=True)
#         },
#         num_nodes_dict={
#             'movie': len(m_idx),
#             'director': len(d_idx),
#             'actor': len(a_idx)
#         })
#
#     features = x[m_idx]
#     x = torch.FloatTensor(features)
#
#     num_nodes = hg.number_of_nodes('movie')
#
#     return x, hg

def train_cl(model, dataloader, opt, epochs, augs, node_types=None,
                metapath=None, metapath_list=None, aug_ratio=0.2, patience=10, device='cpu'):

    best = 1e9
    best_t = 0
    counter = 0

    model.to(device)
    g = nx.Graph()

    for n, typ in enumerate(node_types):
        g.add_node(n, type=typ)

    augmentor = Augmentor(aug_ratio, node_types, metapath, metapath_list)

    for epoch in range(epochs):

        for data in dataloader:

            # node_features, adj = dataloader
            x, node_idx, edge_index, edge_weight = data.x, data.node_idx, data.edge_index, data.edge_attr
            adj = torch.sparse_coo_tensor(edge_index, values=edge_weight, dtype=torch.float32).to_dense()

            coin_flip = random.random()
            x = x.to(device)
            adj = adj.to(device)

            out, att = model(x, adj)
            loss = att_contrastive_loss(g, out, att, node_types)

            # aug_x1, aug_adj1 = augmentor.apply_aug(node_features, adj, augs[0])
            # aug_x2, aug_adj2 = augmentor.apply_aug(node_features, adj, augs[1])
            #
            # # print(aug_x1.shape)
            # # print(aug_adj1.shape)
            # aug_x1, aug_adj1 = aug_x1.to(device), aug_adj1.to(device)
            # aug_x2, aug_adj2 = aug_x2.to(device), aug_adj2.to(device)
            #
            # out1, _ = model(aug_x1, aug_adj1)
            # out2, _ = model(aug_x2, aug_adj2)

            # out1 = out1.flatten(2, 3)
            # out2 = out2.flatten(2, 3)
            # loss2 = NTXent_loss(out1, out2)
            # loss = (loss1 + loss2)/2.
            # loss2 = 0
            # for i in range(len(model.meta_paths)):
            #
            # loss /= len(model.meta_paths)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f'Epoch: {epoch+1}\tLoss: {loss}')

        if loss < best:
            best = loss
            counter = 0
        else:
            counter += 1

        if counter == patience:
            print('Early stopping!')
            break

def train_finetune(model, dataloader, criterion, opt,
                    masks, epochs, patience=10, device='cpu'):

    best = 1e9
    best_t = 0
    counter = 0

    model.to(device)
    model.train()
    for epoch in range(epochs):

        opt.zero_grad()

        x, adj, node_types, labels = dataloader
        # x, g = to_dgl(x, adj, node_types)


        x = x.to(device)
        adj = adj.to(device)
        labels = labels.to(device)
        train_mask = masks[0]

        out = model(x, adj)[train_mask]
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, labels[train_mask])

        loss.backward()
        opt.step()

        with torch.no_grad():
            val_mask = masks[1]
            out = model(x, adj)[val_mask]
            out = F.log_softmax(out, dim=1)
            val_loss = criterion(out, labels[val_mask])
        # loss = model.train_step(dataloader, criterion, opt)

        # print(f'Epoch: {epoch+1}\tLoss: {loss}\tVal_Loss: {val_loss}')

        if val_loss < best:
            best = val_loss
            counter = 0
        else:
            counter += 1

        if counter == patience:
            print('Early stopping!')
            break

# def train_finetune(model, dataloader, criterion, opt,
#                     masks, epochs, patience=10, device='cpu'):
#
#     best = 1e9
#     best_t = 0
#     counter = 0
#
#     model.to(device)
#     model.train()
#     for epoch in range(epochs):
#
#         opt.zero_grad()
#
#         x, adj, node_types, labels = dataloader
#         x, g = to_dgl(x, adj, node_types)
#
#
#         x = x.to(device)
#         g = g.to(device)
#         labels = labels.to(device)
#         train_mask = masks[0]
#
#         out = model(g, x)[train_mask]
#         out = F.log_softmax(out, dim=1)
#         loss = criterion(out, labels[train_mask])
#
#         loss.backward()
#         opt.step()
#
#         with torch.no_grad():
#             val_mask = masks[1]
#             out = model(g, x)[val_mask]
#             out = F.log_softmax(out, dim=1)
#             val_loss = criterion(out, labels[val_mask])
#         # loss = model.train_step(dataloader, criterion, opt)
#
#         # print(f'Epoch: {epoch+1}\tLoss: {loss}\tVal_Loss: {val_loss}')
#
#         if val_loss < best:
#             best = val_loss
#             counter = 0
#         else:
#             counter += 1
#
#         if counter == patience:
#             print('Early stopping!')
#             break

def train_finetune_link(model, dataloader, criterion, opt,
                    edges, epochs, patience=10, device='cpu'):

    best = 1e9
    best_t = 0
    counter = 0

    model.to(device)
    model.train()
    for epoch in range(epochs):

        opt.zero_grad()

        x, adj = dataloader

        x = x.to(device)
        adj = adj.to(device)
        train_edges = torch.as_tensor(edges['train'])
        neg_samples = torch.as_tensor(edges['neg_train'])

        ones = torch.ones((train_edges.shape[1], 1))
        zeros = torch.zeros((neg_samples.shape[1], 1))

        train_edges = train_edges.transpose(1, 0)
        neg_samples = neg_samples.transpose(1, 0)

        train_edges = torch.cat((train_edges, neg_samples), dim=0)
        labels = torch.cat((ones, zeros), dim=0)

        idx_perm = np.random.permutation(len(train_edges))
        train_edges = train_edges[idx_perm]
        labels = labels[idx_perm]
        labels = labels.to(device)

        out = model(x, adj, train_edges[:, 0], train_edges[:, 1])
        loss = criterion(out, labels)

        loss.backward()
        opt.step()

        with torch.no_grad():
            val_edges = torch.as_tensor(edges['val'])
            neg_samples = torch.as_tensor(edges['neg_val'])
            ones = torch.ones((val_edges.shape[1], 1))
            zeros = torch.zeros((neg_samples.shape[1], 1))

            val_edges = val_edges.transpose(1, 0)
            neg_samples = neg_samples.transpose(1, 0)

            val_edges = torch.cat((val_edges, neg_samples), dim=0)
            labels = torch.cat((ones, zeros), dim=0)

            idx_perm = np.random.permutation(len(val_edges))
            val_edges = val_edges[idx_perm]
            labels = labels[idx_perm]
            labels = labels.to(device)

            out = model(x, adj, val_edges[:, 0], val_edges[:, 1])
            val_loss = criterion(out, labels)

        if val_loss < best:
            best = val_loss
            counter = 0
        else:
            counter += 1

        if counter == patience:
            print('Early stopping!')
            break

def evaluate(model, dataloader, criterion, masks, device='cpu'):

    model.eval()
    model.to(device)

    x, adj, node_types, labels = dataloader
    # x, g = to_dgl(x, adj, node_types)


    x = x.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    test_mask = masks[2]
    out = model(x, adj)[test_mask]
    out = F.log_softmax(out, dim=1)
    test_lbls = labels[test_mask]

    loss = criterion(out, test_lbls)
    preds = torch.argmax(out, dim=1)

    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]

    return loss, acc

# def evaluate(model, dataloader, criterion, masks, device='cpu'):
#
#     model.eval()
#     model.to(device)
#
#     x, adj, node_types, labels = dataloader
#     x, g = to_dgl(x, adj, node_types)
#
#
#     x = x.to(device)
#     g = g.to(device)
#     labels = labels.to(device)
#     test_mask = masks[2]
#     out = model(g, x)[test_mask]
#     out = F.log_softmax(out, dim=1)
#     test_lbls = labels[test_mask]
#
#     loss = criterion(out, test_lbls)
#     preds = torch.argmax(out, dim=1)
#
#     acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
#
#     return loss, acc

def evaluate_link(model, dataloader, criterion, edges, device='cpu'):

    model.eval()
    model.to(device)

    x, adj = dataloader

    x = x.to(device)
    adj = adj.to(device)
    test_edges = torch.as_tensor(edges['test'])
    neg_samples = torch.as_tensor(edges['neg_test'])

    ones = torch.ones((test_edges.shape[1], 1))
    zeros = torch.zeros((neg_samples.shape[1], 1))

    test_edges = test_edges.transpose(1, 0)
    neg_samples = neg_samples.transpose(1, 0)

    test_edges = torch.cat((test_edges, neg_samples), dim=0)
    labels = torch.cat((ones, zeros), dim=0)

    idx_perm = np.random.permutation(len(test_edges))
    test_edges = test_edges[idx_perm]
    labels = labels[idx_perm]
    labels = labels.to(device)

    out = model(x, adj, test_edges[:, 0], test_edges[:, 1])
    loss = criterion(out, labels)

    pred = (out > 0.)

    acc = torch.sum(pred == labels).float() / labels.shape[0]

    return loss, acc
