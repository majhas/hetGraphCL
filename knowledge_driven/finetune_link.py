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

from training.train import train_finetune_link, evaluate_link


aug_choices = ['dropN', 'dropE', 'maskN', 'dropN_metapath', 'dropE_metapath', 'maskN_metapath' ,'dropN_not_on_metapath', 'dropE_not_on_metapath', 'maskN_not_on_metapath']

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
    # n_fts = [feat.shape[-1] for feat in node_features]
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

    iters = 20

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

        model = LinkFinetune(gnn=gnn)

        # elif args.model == 'hgt':
        #     num_node_types = len(np.unique(node_types))
        #     num_edge_types = len(np.unique(edge_types.values))
        #     gnn = HGT(
        #             in_dim=n_fts,
        #             hid_dim=args.hid_dim,
        #             out_dim=args.out_dim,
        #             n_layers=args.n_layers,
        #             num_node_types=num_node_types,
        #             num_edge_types=num_edge_types,
        #             n_heads=args.num_heads,
        #             dropout=args.dropout)
        #
        #     model = HetGraphCL(gnn=gnn, head_dim=args.head_dim)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        train_finetune_link(model, [node_features, adj], criterion, opt, train_val_test_edges, args.epochs, device=device)
        loss, auc, ap = evaluate_link(model, [node_features, adj], criterion, train_val_test_edges, device=device)

        print(f'Iter: {iter}\tAUC: {auc}\tAP: {ap}')
        aucs.append(auc * 100)
        aps.append(ap*100)

    aucs = np.stack(aucs)
    aps = np.stack(aps)
    print('AUC Mean:[{:.4f}]'.format(aucs.mean().item()))
    print('AUC Std :[{:.4f}]'.format(aucs.std().item()))

    print('AP Mean:[{:.4f}]'.format(aps.mean().item()))
    print('AP Std :[{:.4f}]'.format(aps.std().item()))



        # if args.save:
        #     save_dict = {}
        #     save_dict['state_dict'] = model.state_dict()
        #     for key in args.__dict__.keys():
        #         save_dict[key] = args.__dict__[key]
        #
        #     save_dict['aug1'] = args.aug1
        #     save_dict['aug2'] = args.aug2
        #
        #     torch.save(save_dict, args.save.replace('.pkl', f'_a1_{args.aug1}_a2_{args.aug2}.pkl'))



if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(args)
