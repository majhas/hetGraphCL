import os
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import random

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch_geometric.data import Data

from utils.data_utils import load_data, load_metapaths
from utils.process import preprocess_adj
from utils.dataset import DataLoader

from models.model_hetero import HAN
from models.gat import GAT
from models.gcn import GCN
from models.graph_cl import GraphCL

from training.train import train_cl, NTXent_loss



aug_choices = ['dropN', 'dropE', 'maskN',
                'dropN_metapath', 'dropE_metapath', 'maskN_metapath', 'subgraph_metapath', 'subgraph_metapath_list',
                'dropN_not_on_metapath', 'dropE_not_on_metapath', 'maskN_not_on_metapath']# 'subgraph_not_on_metapath', 'subgraph_not_on_metapath_list']

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/DBLP/DBLP_processed', help='path to dataset')
    parser.add_argument('--aug1', default=None, choices=aug_choices,
                            help='type of first augmentation to apply')

    parser.add_argument('--aug2', default=None, choices=aug_choices,
                        help='type of first augmentation to apply')
    parser.add_argument('--aug_ratio', default=0.4, type=float, help='augmentation strength')
    parser.add_argument('--metapath', default=None, help='file to a list of metapaths')
    parser.add_argument('--metapath_list', default=None, help='file to a list of metapaths')

    parser.add_argument('--model', default='gcn', choices=['gcn', 'gat'], help='model to use')
    parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension for gnn')
    parser.add_argument('--out_dim', default=64, type=int, help='output dimension for gnn')
    parser.add_argument('--head_dim', default=64, type=int, help='projection head dimension')
    parser.add_argument('--n_layers', default=2, type=int, help='number of layers in model')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')

    parser.add_argument('--batch_size', default=None, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of iterations to train')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    parser.add_argument('--patience', default=10, type=int, help='patience before stopping training')

    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--save', default=None, help='Path to save file')
    return parser.parse_args()



def main(args):

    if 'cuda' in args.device:
        device = torch.device(args.device)

    node_features, adj, node_types, node_map, _, _ = load_data(args.filepath)
    n_fts = node_features.shape[-1]

    metapath = args.metapath.split(',')
    metapath = np.array([node_map[n_type.strip()] for n_type in metapath])
    metapath_list = load_metapaths(args.metapath_list, node_map)

    adj = preprocess_adj(adj)
    edge_idx = (adj.row, adj.col)
    edge_idx = np.array(edge_idx)
    values = adj.data
    shape = adj.shape

    node_features = torch.as_tensor(node_features, dtype=torch.float32)
    edge_idx = torch.as_tensor(edge_idx, dtype=torch.long)
    values = torch.as_tensor(values, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_idx, edge_attr=values)
    if args.batch_size is None:
        data.__dict__['node_idx'] = np.arange(len(node_features))
        dataloader = [data]
    else:
        n_parts = len(node_features)//args.batch_size
        dataloader = DataLoader(data, num_parts=n_parts)

    for aug1 in aug_choices:

        if args.aug1 is not None:
            if aug1 != args.aug1:
                continue


        for aug2 in aug_choices:

            if args.aug2 is not None:
                if aug2 != args.aug2:
                    continue

            print('--- Initialize Model ---')

            augs = [aug1, aug2]

            model = GAT(
                in_dim=n_fts,
                hid_dim=args.hid_dim,
                out_dim=args.out_dim,
                n_layers=args.n_layers,
                dropout=args.dropout)

            # model = GraphCL(gnn, args.head_dim)

            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
            criterion = lambda z1, z2: NTXent_loss(z1, z2)

            print('--- Start Contrastive Learning --- ')

            print('*'*100)
            print(f'Aug1: {aug1}\tAug2: {aug2}')
            print('*'*100)

            train_cl(model, dataloader, opt, args.epochs, augs=augs, node_types=node_types, \
                    metapath=metapath, metapath_list=metapath_list, aug_ratio=args.aug_ratio, patience=args.patience, device=device)

            if args.save:
                save_dict = {}
                save_dict['state_dict'] = model.state_dict()
                for key in args.__dict__.keys():
                    save_dict[key] = args.__dict__[key]

                save_dict['aug1'] = aug1
                save_dict['aug2'] = aug2

                torch.save(save_dict, args.save.replace('.pkl', f'_a1_{aug1}_a2_{aug2}.pkl'))



if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(args)
