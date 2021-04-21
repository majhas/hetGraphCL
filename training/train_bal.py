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


def train_cl(model, dataloader, criterion, opt, epochs, augs, node_types=None,
                metapath=None, metapath_list=None, aug_ratio=0.2, patience=10, device='cpu'):

    best = 1e9
    best_t = 0
    counter = 0

    augmentor = Augmentor(aug_ratio, node_types, metapath, metapath_list)
    model.to(device)
    for epoch in range(epochs):

        opt.zero_grad()

        node_features, adj = dataloader
        aug_x1, aug_adj1 = augmentor.apply_aug(node_features, adj, augs[0])
        aug_x2, aug_adj2 = augmentor.apply_aug(node_features, adj, augs[1])

        aug_x1 = aug_x1.to(device)
        aug_adj1 = aug_adj1.to(device)

        aug_x2 = aug_x2.to(device)
        aug_adj2 = aug_adj2.to(device)

        out1 = model(aug_x1, aug_adj1)
        out2 = model(aug_x2, aug_adj2)

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
                    masks, epochs, patience=10, device='cpu'):

    best = 1e9
    best_t = 0
    counter = 0

    model.to(device)
    model.train()
    for epoch in range(epochs):

        opt.zero_grad()

        x, adj, labels = dataloader[0]

        x = [feat.to(device) for feat in x]
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

        print(f'Epoch: {epoch+1}\tLoss: {loss}\tVal_Loss: {val_loss}')

        if val_loss < best:
            best = loss
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

    x = [feat.to(device) for feat in x]
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
    # micro_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
    # macro_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')

    # print(f'Iter: {i}\tF1: {micro_f1}')
    # tot += acc*100
    # accs.append(acc * 100)
    # micro_f1s.append(micro_f1*100)
