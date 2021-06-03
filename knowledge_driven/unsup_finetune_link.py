import os
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import random

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import load_data, load_data_link, load_metapaths
from utils.dataset import GraphDataset
from utils.process import preprocess_adj

from models.model_finetune import LinkFinetune
from models.gcn import GCN
from models.hgt import HGT

from training.train import train_finetune_link, evaluate_link
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, average_precision_score



aug_choices = ['dropN', 'dropE', 'maskN','dropN_metapath', 'dropE_metapath', 'maskN_metapath', 'dropN_not_on_metapath', 'dropE_not_on_metapath', 'maskN_not_on_metapath']

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/DBLP/DBLP_processed', help='path to dataset')

    parser.add_argument('--model', default='gcn', choices=['gcn', 'hgt'], help='model to use')
    parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension for gnn')
    parser.add_argument('--out_dim', default=64, type=int, help='output dimension for gnn')
    parser.add_argument('--head_dim', default=64, type=int, help='projection head dimension')
    parser.add_argument('--n_layers', default=2, type=int, help='number of layers in model')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')

    parser.add_argument('--batch_size', default=None, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of iterations to train')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')

    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--load', default=None, help='Path to saved model parameter to load')
    return parser.parse_args()



def main(args):

    if 'cuda' in args.device:
        device = torch.device(args.device)

    node_features, adj, node_types, node_map, train_val_test_edges = load_data_link(args.filepath)
    n_fts = node_features.shape[-1]

    adj = preprocess_adj(adj)
    edge_idx = (adj.row, adj.col)
    values = adj.data
    shape = adj.shape

    node_features = torch.tensor(node_features, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_idx, values, shape, dtype=torch.float32)


    if args.load:

        print('-'*50)
        print('Load Model w/ Training Config')
        print('-'*50)

        load_dict = torch.load(args.load, map_location='cpu')
        [print(f'{k}: {load_dict[k]}') for k in load_dict if k != 'state_dict']
        print('-'*50)

        hid_dim = load_dict['hid_dim']
        out_dim = load_dict['out_dim']
        n_layers = load_dict['n_layers']
        dropout = load_dict['dropout']
        state_dict = load_dict['state_dict']
        lr = load_dict['lr']

    else:
        hid_dim=args.hid_dim
        out_dim=args.out_dim
        n_layers=args.n_layers
        dropout=args.dropout
        lr = args.lr
        state_dict = None

    aucs = []
    aps = []

    iters = 10

    for iter in range(iters):
        print('--- Initialize Model ---')

        model = GCN(
            in_dim=n_fts,
            hid_dim=hid_dim,
            out_dim=out_dim,
            n_layers=n_layers,
            dropout=dropout)

        if state_dict is not None:
            model.load_state_dict(state_dict)

        embs = model(node_features, adj).detach().cpu().numpy()
        clf = LinearSVC()
        train_edges = train_val_test_edges['train']
        neg_train = train_val_test_edges['neg_train']

        test_edges = train_val_test_edges['test']
        neg_test = train_val_test_edges['neg_test']

        ones = np.ones((train_edges.shape[1], 1))
        zeros = np.zeros((neg_train.shape[1], 1))

        train_edges = train_edges.transpose(1, 0)
        neg_train = neg_train.transpose(1, 0)

        train_edges = np.concatenate((train_edges, neg_train), axis=0)
        labels = np.concatenate((ones, zeros), axis=0).squeeze(1)

        idx_perm = np.random.permutation(len(train_edges))
        train_edges = train_edges[idx_perm]
        labels = labels[idx_perm]

        s = embs[train_edges[:, 0]]
        t = embs[train_edges[:, 1]]
        train_embs = np.concatenate((s,t), axis=1)

        clf.fit(train_embs, labels)

        ones = np.ones((test_edges.shape[1], 1))
        zeros = np.zeros((neg_test.shape[1], 1))

        test_edges = test_edges.transpose(1, 0)
        neg_test = neg_test.transpose(1, 0)

        test_edges = np.concatenate((test_edges, neg_test), axis=0)
        labels = np.concatenate((ones, zeros), axis=0).squeeze(1)

        s = embs[test_edges[:, 0]]
        t = embs[test_edges[:, 1]]
        test_embs = np.concatenate((s,t), axis=1)

        out = clf.predict(test_embs)
        auc = roc_auc_score(labels, out)
        ap = average_precision_score(labels, out)


        print(f'Iter: {iter}\tAUC: {auc}\tAP: {ap}')
        aucs.append(auc * 100)
        aps.append(ap * 100)

    aucs = np.stack(aucs)
    aps = np.stack(aps)
    print('AUC Mean:[{:.4f}]'.format(aucs.mean().item()))
    print('AUC Std :[{:.4f}]'.format(aucs.std().item()))

    print('AP Mean:[{:.4f}]'.format(aps.mean().item()))
    print('AP Std :[{:.4f}]'.format(aps.std().item()))




if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    main(args)
