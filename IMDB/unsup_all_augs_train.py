import os
import csv
import argparse
import numpy as np
import scipy.sparse as sp
import random
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from utils.aug import *
from utils.data_utils import *
from utils.dataset import HetGraphDataset
from utils import process
from models.graph_cl import GraphCL, HetGraphCL
from models.gcn import GCN
from models.hgt import HGT
from models.logreg import LogReg
from sklearn.metrics import f1_score


aug_choices = [None, 'dropN', 'dropE', 'maskN',
                'dropN_metapath', 'dropE_metapath', 'subgraph_metapath', 'subgraph_metapath_list']
                # ,'dropN_not_on_metapath', 'dropE_not_on_metapath', 'subgraph_not_on_metapath', 'subgraph_not_on_metapath_list']

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/IMDB_processed', help='path to dataset')
    parser.add_argument('--aug1', default=None, choices=aug_choices,
                            help='type of first augmentation to apply')
    parser.add_argument('--aug2', default=None, choices=aug_choices,
                        help='type of second augmentation to apply')
    parser.add_argument('--aug_ratio', default=0.2, type=float, help='augmentation strength')

    parser.add_argument('--model', default='gcn', choices=['gcn', 'hgt'], help='model to use')
    parser.add_argument('--hid_dim', default=16, type=int, help='hidden dimension for gnn')
    parser.add_argument('--out_dim', default=16, type=int, help='output dimension for gnn')
    parser.add_argument('--head_dim', default=16, type=int, help='projection head dimension')
    parser.add_argument('--n_layers', default=1, type=int, help='number of layers in model')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of iterations to train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--num_heads', default=8, type=int, help='number of attention heads')
    parser.add_argument('--metapath', default=None, help='file to a list of metapaths')
    parser.add_argument('--metapath_list', default=None, help='file to a list of metapaths')
    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--save', default=None, help='Path to save file')
    parser.add_argument('--load', default=None, help='Path to saved model parameter to load')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    return parser.parse_args()



def main(args):

    if 'cuda' in args.device:
        device = torch.device(args.device)

    x, adj, node_types, labels, train_val_test_idx = load_data(args.filepath)

    edge_types = {
        (0, 0): 0,
        (1, 1): 0,
        (2, 2): 0,
        (0, 1): 1,
        (1, 0): 1,
        (0, 2): 2,
        (2, 0): 2
    }

    n_classes = len(np.unique(labels))
    n_fts = x.shape[-1]

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    if args.metapath:
        metapath = np.array(args.metapath.split(','), dtype=np.int)
    elif args.metapath_list:
        metapath = load_metapaths(args.metapath_list)
    else:
        metapath = None

    adj = adj.todense()

    x = torch.tensor(x, dtype=torch.float32)
    adj = torch.tensor(adj, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    x = x.unsqueeze(0)
    adj = adj.unsqueeze(0)
    labels = labels.unsqueeze(0)


    dataset = HetGraphDataset(x=x, adj=adj, node_types=node_types, edge_types=edge_types,
                                aug_type=None, aug_ratio=args.aug_ratio, device=device)

    print('--- Initialize Model ---')

    metapath_list = load_metapaths('data/metapaths.txt')
    print('--- Start Contrastive Learning --- ')

    for aug1 in aug_choices:

        if aug1 == 'subgraph_metapath_list' or aug1 == 'subgraph_not_on_metapath_list':
            metapath1 = metapath_list
        else:
            metapath1 = metapath

        for aug2 in aug_choices:
            if os.path.isfile(args.save.replace('.pkl', f'_a1_{aug1}_a2_{aug2}.pkl')):
                continue
            # if aug2 is not 'dropE_metapath':
            #     continue
            if aug2 == 'subgraph_metapath_list' or aug2 == 'subgraph_not_on_metapath_list':
                metapath2 = metapath_list
            else:
                metapath2 = metapath

            best = 1e9
            best_t = 0
            patience = 10
            # temp = 0.5

            if args.model == 'gcn':
                gnn = GCN(in_dim=n_fts, hid_dim=args.hid_dim, out_dim=args.out_dim, n_layers=args.n_layers, dropout=args.dropout)
                model = GraphCL(gnn=gnn, head_dim=args.head_dim)

            elif args.model == 'hgt':
                num_node_types = len(np.unique(node_types))
                num_edge_types = len(np.unique(edge_types.values))
                gnn = HGT(in_dim=n_fts, hid_dim=args.hid_dim, out_dim=args.out_dim, n_layers=args.n_layers,
                            num_node_types=num_node_types, num_edge_types=num_edge_types, n_heads=args.num_heads, dropout=args.dropout)
                model = HetGraphCL(gnn=gnn, head_dim=args.head_dim)

            model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
            #                                                  factor=0.1,
            #                                                  patience=10,
            #                                                  verbose=True)


            print('*'*100)
            print(f'Aug1: {aug1}\tAug2: {aug2}')
            print('*'*100)

            for epoch in range(args.epochs):

                dataset1 = HetGraphDataset(x=dataset.x, adj=dataset.adj, node_types=dataset.node_types, edge_types=dataset.edge_types,
                                            aug_type=aug1, metapath=metapath1, aug_ratio=args.aug_ratio, device=device)

                dataset2 = HetGraphDataset(x=dataset.x, adj=dataset.adj, node_types=dataset.node_types, edge_types=dataset.edge_types,
                                            aug_type=aug2, metapath=metapath2, aug_ratio=args.aug_ratio, device=device)

                dataloader1 = DataLoader(dataset1, batch_size=args.batch_size)
                dataloader2 = DataLoader(dataset2, batch_size=args.batch_size)

                loss = model.train_step(dataloader1, dataloader2, opt)
                # epoch_loss = torch.mean(torch.tensor(losses))
                # scheduler.step(loss)
                print(f'Epoch: {epoch+1}\tLoss: {loss}')

                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    # torch.save(model.state_dict(), args.save_name)
                else:
                    cnt_wait += 1

                if cnt_wait == patience:
                    print('Early stopping!')
                    break

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
