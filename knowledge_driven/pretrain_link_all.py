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

from utils.data_utils import load_data_link, load_metapaths
from utils.process import preprocess_adj
from utils.dataset import DataLoader

from models.graph_cl import GraphCL
from models.gcn import GCN

from training.train import train_cl, NTXent_loss



aug_choices = ['dropN', 'dropE', 'maskN', 'dropN_metapath', 'dropE_metapath', 'maskN_metapath','dropN_not_on_metapath', 'dropE_not_on_metapath', 'maskN_not_on_metapath']

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/DBLP/DBLP_processed', help='path to dataset')
    parser.add_argument('--aug', default=None, choices=aug_choices,
                            help='type of first augmentation to apply')
    parser.add_argument('--aug_ratio', default=0.4, type=float, help='augmentation strength')
    parser.add_argument('--metapath', default=None, help='file to a list of metapaths')
    parser.add_argument('--metapath_list', default=None, help='file to a list of metapaths')

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

    parser.add_argument('--device', default='cuda', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--save', default=None, help='Path to save file')
    return parser.parse_args()



def main(args):

    if 'cuda' in args.device:
        device = torch.device(args.device)

    node_features, adj, node_types, node_map, _ = load_data_link(args.filepath)
    n_fts = node_features.shape[-1]

    metapath = args.metapath.split(',')
    metapath = np.array([node_map[n_type.strip()] for n_type in metapath])
    metapath_list = load_metapaths(args.metapath_list, node_map)
    # print(metapath_list)

    adj = preprocess_adj(adj)
    edge_idx = (adj.row, adj.col)
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

        if args.aug is not None:
            if aug1 != args.aug:
                continue



        for aug2 in aug_choices:
            if os.path.isfile(args.save.replace('.pkl', f'_a1_{aug1}_a2_{aug2}.pkl')):
                continue
            print('--- Initialize Model ---')

            augs = [aug1, aug2]

            if args.model == 'gcn':
                gnn = GCN(
                        in_dim=n_fts,
                        hid_dim=args.hid_dim,
                        out_dim=args.out_dim,
                        n_layers=args.n_layers,
                        dropout=args.dropout)

                model = nn.DataParallel(GraphCL(gnn=gnn, head_dim=args.head_dim))

            elif args.model == 'hgt':
                num_node_types = len(np.unique(node_types))
                num_edge_types = len(np.unique(edge_types.values))
                gnn = HGT(
                        in_dim=n_fts,
                        hid_dim=args.hid_dim,
                        out_dim=args.out_dim,
                        n_layers=args.n_layers,
                        num_node_types=num_node_types,
                        num_edge_types=num_edge_types,
                        n_heads=args.num_heads,
                        dropout=args.dropout)

                model = HetGraphCL(gnn=gnn, head_dim=args.head_dim)

            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
            criterion = lambda z1, z2: NTXent_loss(z1, z2)

            print('--- Start Contrastive Learning --- ')

            print('*'*100)
            print(f'Aug1: {aug1}\tAug2: {aug2}')
            print('*'*100)

            train_cl(model, dataloader, criterion, opt, args.epochs, augs=augs, node_types=node_types, \
                    metapath=metapath, metapath_list=metapath_list, aug_ratio=args.aug_ratio, device=device)

            if args.save:
                save_dict = {}
                save_dict['state_dict'] = model.gnn.state_dict()
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
