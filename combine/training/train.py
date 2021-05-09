import numpy as np
import dgl

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augs import Augmentor

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

def sample(g, feat, att, num_neg=64):
    u, v, eid = g.edges(form='all', order='srcdst')
    att = att.mean(dim=1)[eid].squeeze()
    # print(feat.shape)
    pos_id = [0] * feat.shape[0]
    _, counts = u.unique(return_counts=True)
    neighbors = torch.split_with_sizes(v, tuple(counts))
    for i, item in enumerate(torch.split_with_sizes(att, tuple(counts))):
        one_hot = F.gumbel_softmax(item, tau=1, hard=True).type(torch.bool)
        # print(f'O: {one_hot}')
        # print(f'N: {neighbors[i].shape}')
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

def contrastive_loss(gs, zs, atts, meta_paths, temp=0.5):

    loss1 = 0
    loss2 = 0
    for i, meta_path in enumerate(meta_paths):
        g1 = gs[0][tuple(meta_path)]
        att1 = atts[0][i]
        z1 = zs[0][i]

        # print(z1.shape)
        # print(att1.shape)
        pos1, neg1 = sample(g1, z1, att1)

        g2 = gs[0][tuple(meta_path)]
        att2 = atts[0][i]
        z2 = zs[0][i]

        pos2, neg2 = sample(g2, z2, att2)

        l_pos = F.cosine_similarity(z1, pos1, dim=2).flatten().unsqueeze(1) # NM * 1
        l_neg1 = F.cosine_similarity(z1.expand(neg1.shape), neg1, dim=3).flatten(1, 2).transpose(0, 1) # NM * K sample
        l_neg2 = F.cosine_similarity(z1.expand(neg2.shape), neg2, dim=3).flatten(1, 2).transpose(0, 1) # NM * K sample

        logits1 = torch.cat((l_pos, l_neg1, l_neg2), dim=1)
        logits1 /= temp
        loss1 += nn.CrossEntropyLoss()(logits1, torch.zeros(logits1.size(0), dtype=torch.long,
                                                        device=z1.device))

        l_pos = F.cosine_similarity(z2, pos2, dim=2).flatten().unsqueeze(1) # NM * 1
        l_neg1 = F.cosine_similarity(z2.expand(neg1.shape), neg1, dim=3).flatten(1, 2).transpose(0, 1) # NM * K sample
        l_neg2 = F.cosine_similarity(z2.expand(neg2.shape), neg2, dim=3).flatten(1, 2).transpose(0, 1) # NM * K sample

        logits2 = torch.cat((l_pos, l_neg1, l_neg2), dim=1)
        logits2 /= temp
        loss2 += nn.CrossEntropyLoss()(logits2, torch.zeros(logits2.size(0), dtype=torch.long,
                                                        device=z2.device))


    return (loss1 + loss2)/(2.*len(meta_paths))

def to_dgl(x, adj, node_types):

    # _, counts = np.unique(node_types, return_counts=True)

    m_idx = np.where(node_types == 0)[0]
    d_idx = np.where(node_types == 1)[0]
    a_idx = np.where(node_types == 2)[0]

    d_vs_m = adj[d_idx][:, m_idx]
    a_vs_m = adj[a_idx][:, m_idx]

    # print(len(m_idx))
    # print(d_vs_m.shape)
    # print(a_vs_m.shape)
    hg = dgl.heterograph(
        {
            ('director', 'dm', 'movie'): torch.nonzero(d_vs_m, as_tuple=True),
            ('movie', 'md', 'director'): torch.nonzero(d_vs_m.transpose(1, 0), as_tuple=True),
            ('actor', 'am', 'movie'): torch.nonzero(a_vs_m, as_tuple=True),
            ('movie', 'ma', 'actor'): torch.nonzero(a_vs_m.transpose(1, 0), as_tuple=True)
        },
        num_nodes_dict={
            'movie': len(m_idx),
            'director': len(d_idx),
            'actor': len(a_idx)
        })

    features = x[m_idx]
    x = torch.FloatTensor(features)

    num_nodes = hg.number_of_nodes('movie')

    return x, hg

def train_cl(model, dataloader, criterion, opt, epochs, augs, node_types=None,
                metapath=None, metapath_list=None, aug_ratio=0.2, patience=10, device='cpu'):

    best = 1e9
    best_t = 0
    counter = 0

    augmentor = Augmentor(aug_ratio, node_types, metapath, metapath_list)
    model.to(device)
    for epoch in range(epochs):

        for data in dataloader:

            # node_features, adj = dataloader
            node_features, node_idx, edge_index, edge_weight = data.x, data.node_idx, data.edge_index, data.edge_attr
            adj = torch.sparse_coo_tensor(edge_index, values=edge_weight, dtype=torch.float32).to_dense()

            aug_x1, aug_adj1 = augmentor.apply_aug(node_features, adj, augs[0])
            aug_x2, aug_adj2 = augmentor.apply_aug(node_features, adj, augs[1])

            # print(aug_x1.shape)
            # print(aug_adj1.shape)
            x1, g1 = to_dgl(aug_x1, aug_adj1, node_types)
            x2, g2 = to_dgl(aug_x2, aug_adj2, node_types)

            x1, g1 = x1.to(device), g2.to(device)
            x2, g2 = x1.to(device), g2.to(device)

            _, out1 = model(g1, x1)
            atts1 = model.layers[-1].atts
            gs1 = model.layers[-1]._cached_coalesced_graph

            _, out2 = model(g2, x2)
            atts2 = model.layers[-1].atts
            gs2 = model.layers[-1]._cached_coalesced_graph

            loss = contrastive_loss([gs1, gs2], [out1, out2], [atts1, atts2], model.meta_paths)
            # print(att_based_loss1)
            # print(att_based_loss2)
            #
            # loss1 = criterion(out1, out2)
            # loss2 = criterion(out2, out1)

            # print(loss1)
            # print(loss2)
            # loss = (att_based_loss1 + att_based_loss2 + loss1 + loss2)/4.

            loss.backward()
            opt.step()

            # loss = model.train_step(dataloader, criterion, opt)

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

        x, adj, labels = dataloader[0]

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

    x, adj, labels = dataloader[0]

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
