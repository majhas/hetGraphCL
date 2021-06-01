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
from models.gat import GAT

from training.train import train_finetune, evaluate

from plotly.figure_factory import create_table


aug_choices = ['dropN', 'dropE', 'maskN', 'dropN_metapath', 'dropE_metapath', 'maskN_metapath', 'dropN_not_on_metapath', 'dropE_not_on_metapath', 'maskN_not_on_metapath']

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/DBLP/DBLP_processed', help='path to dataset')
    parser.add_argument('--aug', default=None, choices=aug_choices,
                            help='type of first augmentation to apply')

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

    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--load', default=None, help='Path to saved model parameter to load')
    return parser.parse_args()



def main(args):

    if 'cuda' in args.device:
        device = torch.device(args.device)
        print(device)

    node_features, adj, node_types, node_map, labels, train_val_test_idx = load_data(args.filepath)
    # n_fts = [feat.shape[-1] for feat in node_features]
    n_fts = node_features.shape[-1]
    n_classes = len(np.unique(labels))

    adj = preprocess_adj(adj)
    edge_idx = (adj.row, adj.col)
    values = adj.data
    shape = adj.shape

    # node_features = [torch.tensor(feat, dtype=torch.float32) for feat in node_features]
    node_features = torch.as_tensor(node_features, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(edge_idx, values, shape, dtype=torch.float32).to_dense()
    labels = torch.LongTensor(labels)
    masks = [torch.LongTensor(train_val_test_idx[split]) for split in train_val_test_idx]
    dataset = GraphDataset(
                node_features=node_features,
                adj=adj,
                labels=labels)


    micro_results_table = []
    macro_results_table = []

    for aug1 in aug_choices:

        if args.aug is not None:
            if aug1 != args.aug:
                micro_res = [0]*len(aug_choices)
                macro_res = [0]*len(aug_choices)
                continue
        micro_res = []
        macro_res = []

        for aug2 in aug_choices:


            load = args.load.replace('.pkl', f'_a1_{aug1}_a2_{aug2}.pkl')
            try:
                load_dict = torch.load(load, map_location='cpu')
            except:
                micro_res.append(0)
                macro_res.append(0)
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


            f1_micros = []
            f1_macros = []
            iters = 20
            print('--- Start Contrastive Learning --- ')

            print('*'*100)
            print(f'Aug1: {aug1}\tAug2: {aug2}')
            print('*'*100)

            for iter in range(iters):

                if args.model == 'gcn':
                    gnn = GCN(
                            in_dim=n_fts,
                            hid_dim=hid_dim,
                            out_dim=out_dim,
                            n_layers=n_layers,
                            dropout=dropout)
                elif args.model == 'gat':
                    gnn = GAT(
                    in_dim=n_fts,
                    hid_dim=args.hid_dim,
                    out_dim=args.out_dim,
                    n_layers=args.n_layers,
                    dropout=args.dropout)

                if state_dict is not None:
                    gnn.load_state_dict(state_dict)

                model = ModelFinetune(gnn=gnn, n_classes=n_classes)

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
                criterion = nn.NLLLoss()

                train_finetune(model, dataset, criterion, opt, masks, args.epochs, device=device)
                loss, f1_micro, f1_macro = evaluate(model, dataset, criterion, masks, device=device)

                print(f'Iter: {iter}\tAcc: {f1_micro}')
                f1_micros.append(f1_micro * 100)
                f1_macros.append(f1_macro * 100)

            f1_micros = np.stack(f1_micros)
            f1_macros = np.stack(f1_macros)

            print('Mean:[{:.4f}]'.format(f1_micros.mean().item()))
            print('Std :[{:.4f}]'.format(f1_micros.std().item()))

            micro_res.append('{:.2f} +- {:.2f}'.format(f1_micros.mean().item(), f1_micros.std().item()))
            macro_res.append('{:.2f} +- {:.2f}'.format(f1_macros.mean().item(), f1_macros.std().item()))

        micro_results_table.append(micro_res)
        macro_results_table.append(macro_res)



    micro_results_table = pd.DataFrame(micro_results_table, columns=aug_choices, index=aug_choices)
    macro_results_table = pd.DataFrame(macro_results_table, columns=aug_choices, index=aug_choices)
    micro_results_table.to_csv(args.load.replace('.pkl', '_micro_results_table.csv'))
    macro_results_table.to_csv(args.load.replace('.pkl', '_macro_results_table.csv'))

    print(micro_results_table)
    print(macro_results_table)

    # table = create_table(micro_results_table, index=True)
    # table.update_layout(
    #     autosize=False,
    #     width=2400,
    #     height=500
    # )

    # table.write_image(args.load.replace('.pkl', 'micro_semisup_results_table.png'))

    # print(macro_results_table)
    # table = create_table(macro_results_table, index=True)
    # table.update_layout(
    #     autosize=False,
    #     width=2400,
    #     height=500
    # )
    # table.write_image(args.load.replace('.pkl', 'macro_semisup_results_table.png'))

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
