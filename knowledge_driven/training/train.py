import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augs import Augmentor
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

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


def train_cl(model, dataloader, criterion, opt, epochs, augs, node_types=None,
                metapath=None, metapath_list=None, aug_ratio=0.2, patience=30, device='cpu'):

    best = 1e9
    best_t = 0
    counter = 0

    augmentor = Augmentor(aug_ratio, node_types, metapath, metapath_list)
    model.to(device)
    for epoch in range(epochs):

        for data in dataloader:
            opt.zero_grad()

            # node_features, adj = dataloader
            node_features, node_idx, edge_index, edge_weight = data.x, data.node_idx, data.edge_index, data.edge_attr
            adj = torch.sparse_coo_tensor(edge_index, values=edge_weight, dtype=torch.float32).to_dense()

            if metapath_list is not None:
                metapath = random.sample(metapath_list, 1)[0] # randomly sample metapath from list

            print(metapath)
            aug_x1, aug_adj1 = augmentor.apply_aug(node_features, adj, augs[0], metapath)
            aug_x2, aug_adj2 = augmentor.apply_aug(node_features, adj, augs[1], metapath)

            aug_x1 = aug_x1.to(device)
            aug_adj1 = aug_adj1.to(device)

            aug_x2 = aug_x2.to(device)
            aug_adj2 = aug_adj2.to(device)

            out1 = model(aug_x1, aug_adj1)
            out2 = model(aug_x2, aug_adj2)

            # print(torch.cuda.memory_allocated(device)/(1024*1024*1024))

            loss = criterion(out1, out2)

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
                    masks, epochs, patience=30, device='cpu'):

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

    f1_micro = f1_score(test_lbls.cpu().numpy(), preds.cpu().numpy(), average='micro')
    f1_macro = f1_score(test_lbls.cpu().numpy(), preds.cpu().numpy(), average='macro')

    return loss, f1_micro, f1_macro

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

    out = torch.sigmoid(out)
    labels = labels.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    auc = roc_auc_score(labels, out)
    ap = average_precision_score(labels, out)

    return loss, auc, ap
