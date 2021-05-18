from model_hetero import linear_eval
import torch
from sklearn.metrics import f1_score
import torch.nn as nn
from tools import evaluate_results_nc
import os
import numpy as np
import random
import dgl
import torch.nn.functional as F

import scipy.sparse as sp
from utils import load_data, EarlyStopping
from augs import Augmentor

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

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=1.414)

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func, model2):
    model.eval()
    model2.eval()
    with torch.no_grad():
        z = model.test_step(g, features)
        logits = model2(z)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def to_dgl(x, adj, node_types):

    # _, counts = np.unique(node_types, return_counts=True)

    m_idx = np.where(node_types == 0)[0]
    d_idx = np.where(node_types == 1)[0]
    a_idx = np.where(node_types == 2)[0]

    d_vs_m = adj[d_idx][:, m_idx]
    a_vs_m = adj[a_idx][:, m_idx]

    d_vs_m = sp.csr_matrix(d_vs_m)
    a_vs_m = sp.csr_matrix(a_vs_m)

    hg = dgl.heterograph(
        {
        ('director', 'dm', 'movie'): d_vs_m.nonzero(),
        ('movie', 'md', 'director'): d_vs_m.transpose().nonzero(),
        ('actor', 'am', 'movie'): a_vs_m.nonzero(),
        ('movie', 'ma', 'actor'): a_vs_m.transpose().nonzero()
        },
        num_nodes_dict={
            'movie': len(m_idx),
            'director': len(d_idx),
            'actor': len(a_idx)
        })
    features = x[m_idx]
    x = torch.FloatTensor(features.numpy())

    num_nodes = hg.number_of_nodes('movie')

    return x, hg

def load_metapaths(filepath, node_map):

    metapaths = []
    with open(filepath, 'r') as f:
        for line in f:
            metapath = line.split(',')
            metapath = np.array([node_map[n_type.strip()] for n_type in metapath])
            metapaths.append(metapath)

    return metapaths
def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask, adj, node_types = load_data(args['dataset'])
    adj = adj.todense()

    features_1 = np.zeros((len(np.where(node_types==1)[0]), features.shape[1])) # director
    features_2 = np.zeros((len(np.where(node_types==2)[0]), features.shape[1])) # actor

    best_micro_f1 = 0
    best_macro_f1 = 0

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    if args['hetero']:
        from model_hetero import HAN
        model = HAN(meta_paths=[['md', 'dm'], ['ma', 'am']],
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout'])

        # model = nn.DataParallel(model)
        model.to(args['device'])
        g = g.to(args['device'])
    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]

    model_finetune = linear_eval(in_size=args['hidden_units'] * args['num_heads'][-1], out_size=num_classes)
    model_finetune = nn.DataParallel(model_finetune)
    model_finetune.to(args['device'])

    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    model_dir = './checkpoints'


    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    nmi_mean_list = []
    nmi_std_list = []
    ari_mean_list = []
    ari_std_list = []

    node_map = {
            'movie': 0,
            'director': 1,
            'actor': 2,
        }
    metapath_list = load_metapaths('../data/IMDB/metapaths.txt', node_map)
    metapath = 'movie,director,movie'.split(',')
    metapath = np.array([node_map[n_type.strip()] for n_type in metapath])
    augmentor = Augmentor(aug_ratio=0.1, node_types=node_types, metapath=metapath, metapath_list=metapath_list)

    augs = ['dropN_not_on_metapath', 'maskN']
    for i in range(1):

        torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(0)))
        for epoch in range(args['num_epochs']):
            stopper = EarlyStopping(os.path.join(model_dir, 'model_epoch-{}.pth'.format(epoch)), patience=args['patience'])
            model.load_state_dict(torch.load(os.path.join(model_dir, 'epoch-{}.pth'.format(epoch))))
            model.train()
            model_finetune.train()
            model_finetune.apply(weight_reset)
            coin_flip = random.random()
            if coin_flip < 0.5:
                z, _ = model(g, features, optimizer, epoch)

            else:

                node_features = torch.as_tensor(np.concatenate((features.cpu(), features_1, features_2), axis=0))
                adj = torch.as_tensor(adj)
                aug_x1, aug_adj1 = augmentor.apply_aug(node_features, adj, augs[0])
                aug_x2, aug_adj2 = augmentor.apply_aug(node_features, adj, augs[1])

                x1, hg1 = to_dgl(aug_x1, aug_adj1, node_types)
                x2, hg2 = to_dgl(aug_x1, aug_adj1, node_types)

                x1 = x1.to(args['device'])
                hg1 = hg1.to(args['device'])

                x2 = x2.to(args['device'])
                hg2 = hg2.to(args['device'])

                _, out1 = model(hg1, x1, epoch, optimizer, no_loss=True)
                _, out2 = model(hg2, x2, epoch, optimizer, no_loss=True)

                # print(torch.cuda.memory_allocated(device)/(1024*1024*1024))

                loss = NTXent_loss(out1, out2)

                loss.backward()
                optimizer.step()
            torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch+1)))
            stopper1 = EarlyStopping(os.path.join(model_dir, 'finetune_epoch-{}.pth'.format(epoch)), patience=args['patience'])

            params = list(model.parameters()) + list(model_finetune.parameters())
            finetune_opt = torch.optim.Adam(params, 0.005,
                                        weight_decay=args['weight_decay'])
            for i in range(200):
                z = model.test_step(g, features)
                logits = model_finetune(z)

                loss = loss_fcn(logits[train_mask], labels[train_mask])

                finetune_opt.zero_grad()

                loss.backward(retain_graph=True)
                finetune_opt.step()

                train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
                val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn, model_finetune)
                early_stop = stopper1.step(val_loss.data.item(), val_acc, model_finetune)
                stopper.step(val_loss.data.item(), val_acc, model)

                if (i + 1) % 10 == 0:
                    print('Inner Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
                        'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                        i + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

                if early_stop:
                    break

            stopper1.load_checkpoint(model_finetune)
            stopper.load_checkpoint(model)
            with torch.no_grad():
                embeddings = model.test_step(g, features)

                svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
                    embeddings[test_mask].cpu().numpy(), labels[test_mask].cpu().numpy(), num_classes=num_classes)


            test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask, loss_fcn, model_finetune)
            print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
                test_loss.item(), test_micro_f1, test_macro_f1))

            best_micro_f1 = max(best_micro_f1, test_micro_f1)
            best_macro_f1 = max(best_macro_f1, test_macro_f1)

        print('Best Test Micro f1 {:.4f} | Best Test Macro f1 {:.4f}'.format(best_micro_f1, best_macro_f1))

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-dataset', type=str, default='IMDB')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true', default=True,
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
