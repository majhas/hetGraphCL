import os
import csv
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import load_data, load_metapaths
from utils.dataset import GraphDataset
from utils.process import preprocess_adj

from models.model_finetune import ModelFinetune
from models.gcn import GCN
from models.hgt import HGT

from training.train import train_finetune, evaluate
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans


aug_choices = ['dropN', 'dropE', 'maskN', 'subgraph',
                'dropN_metapath', 'dropE_metapath', 'maskN_metapath', 'subgraph_metapath', 'subgraph_metapath_list',
                'dropN_not_on_metapath', 'dropE_not_on_metapath', 'maskN_not_on_metapath']


def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)

def evaluate_clusters(embeddings, labels, num_classes):
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return nmi_mean, nmi_std, ari_mean, ari_std

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/DBLP/DBLP_processed', help='path to dataset')

    parser.add_argument('--aug', default=None)
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
    # n_fts = [feat.shape[-1] for feat in node_features]
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

    nmi_table = []
    ari_table = []
    print('--- Initialize Model ---')
    for aug1 in aug_choices:

        if aug1 != args.aug:
            nmi_res = [0]*len(aug_choices)
            continue
        nmi_res = []
        ari_res = []

        for aug2 in aug_choices:


            load = args.load.replace('.pkl', f'_a1_{aug1}_a2_{aug2}.pkl')
            try:
                load_dict = torch.load(load, map_location='cpu')
            except:
                nmi_res.append(0)
                ari_res.append(0)
                continue

            print('-'*50)
            print('Load Model w/ Training Config')
            print('-'*50)

            [print(f'{k}: {load_dict[k]}') for k in load_dict if k != 'state_dict']
            print('-'*50)

            hid_dim = load_dict['hid_dim']
            out_dim = load_dict['out_dim']
            n_layers = load_dict['n_layers']
            dropout = load_dict['dropout']
            state_dict = load_dict['state_dict']
            lr = load_dict['lr']


            aucs = []
            aps = []
            iters = 50
            print('--- Start Contrastive Learning --- ')

            print('*'*100)
            print(f'Aug1: {aug1}\tAug2: {aug2}')
            print('*'*100)

            gnn = GCN(
                in_dim=n_fts,
                hid_dim=hid_dim,
                out_dim=out_dim,
                n_layers=n_layers,
                dropout=dropout)

            if state_dict is not None:
                gnn.load_state_dict(state_dict)
            model = ModelFinetune(gnn=gnn, n_classes=n_classes)

            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
            criterion = nn.NLLLoss()

            train_finetune(model, dataset, criterion, opt, masks, args.epochs, device=device)
            gnn.to('cpu')
            embs = gnn(node_features, adj)
            target_idx = np.where(node_types==0)[0]
            nmi_mean, nmi_std, ari_mean, ari_std = evaluate_clusters(embs[target_idx].detach().numpy(), labels.detach().numpy()[target_idx], n_classes)

            nmi_res.append('{:.4f} +- {:.4f}'.format(nmi_mean*100, nmi_std*100))
            ari_res.append('{:.4f} +- {:.4f}'.format(ari_mean*100, ari_std*100))

        nmi_table.append(nmi_res)
        ari_table.append(ari_res)

    nmi_table = pd.DataFrame(nmi_table, columns=aug_choices, index=aug_choices)
    ari_table = pd.DataFrame(ari_table, columns=aug_choices, index=aug_choices)
    # nmi_table.to_csv(args.load.replace('.pkl', '_semi_nmi_results_table.csv'))
    # ari_table.to_csv(args.load.replace('.pkl', '_semi_ari_results_table.csv'))
    print(nmi_table)
    print(ari_table)

if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(args)
