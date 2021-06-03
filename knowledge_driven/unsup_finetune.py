import os
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import random

from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import load_data, load_metapaths
from utils.dataset import GraphDataset
from utils.process import preprocess_adj

from models.model_finetune import ModelFinetune
from models.gcn import GCN

from training.train import train_finetune, evaluate


aug_choices = ['dropN', 'dropE', 'maskN', 'dropN_metapath', 'dropE_metapath', 'maskN_metapath', 'dropN_not_on_metapath', 'dropE_not_on_metapath', 'maskN_not_on_metapath']

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
    parser.add_argument('--patience', default=10, type=int, help='patience before stopping training')
    return parser.parse_args()



def main(args):

    if 'cuda' in args.device:
        device = torch.device(args.device)

    node_features, adj, node_types, node_map, labels, train_val_test_idx = load_data(args.filepath)
    n_fts = node_features.shape[-1]
    n_classes = len(np.unique(labels))

    adj = preprocess_adj(adj)
    edge_idx = (adj.row, adj.col)
    values = adj.data
    shape = adj.shape

    node_features = torch.tensor(node_features, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_idx, values, shape, dtype=torch.float32)
    labels = torch.LongTensor(labels)
    masks = [torch.LongTensor(train_val_test_idx[split]) for split in train_val_test_idx]
    dataset = GraphDataset(
                node_features=node_features,
                adj=adj,
                labels=labels)



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
        state_dict = None
        lr = args.lr

    patience = args.patience
    f1_micros = []
    f1_macros = []

    iters = 10

    for iter in range(iters):
        print('--- Initialize Model ---')

        if args.model == 'gcn':
            gnn = GCN(
                in_dim=n_fts,
                hid_dim=hid_dim,
                out_dim=out_dim,
                n_layers=n_layers,
                dropout=dropout)

        if state_dict is not None:
            gnn.load_state_dict(state_dict)


        embs = gnn(node_features, adj).detach().numpy()
        labels = labels
        train_embs = embs[masks[0]]
        train_labels = labels[masks[0]]
        clf = LinearSVC()
        clf.fit(train_embs, train_labels)

        test_embs = embs[masks[2]]
        test_labels = labels[masks[2]]
        y_pred = clf.predict(test_embs)

        f1_micro = f1_score(test_labels, y_pred, average='micro')
        f1_macro = f1_score(test_labels, y_pred, average='macro')

        f1_micros.append(f1_micro * 100)
        f1_macros.append(f1_macro * 100)

    f1_micros = np.stack(f1_micros)
    f1_macros = np.stack(f1_macros)

    print('Mean:[{:.4f}]'.format(f1_micros.mean().item()))
    print('Std :[{:.4f}]'.format(f1_micros.std().item()))

    print('Mean:[{:.4f}]'.format(f1_macros.mean().item()))
    print('Std :[{:.4f}]'.format(f1_macros.std().item()))



if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(args)
