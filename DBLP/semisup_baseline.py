import os
import csv
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from utils.dataset import HetGraphDataset
from utils.aug import *
from utils.data_utils import *
from utils import process
from models.graph_cl import GraphCL, HetGraphCL
from models.gcn import GCN
from models.hgt import HGT
from models.logreg import LogReg
from models.model_finetune import ModelFinetune, HetModelFinetune
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from plotly.figure_factory import create_table

def parse_args():

    aug_choices = ['Identical', 'dropN', 'dropE', 'maskN', 'dropN_metapath', 'dropE_metapath', 'subgraph_metapath', 'subgraph_metapath_list']

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/DBLP_processed', help='path to dataset')
    parser.add_argument('--model', default='gcn', choices=['gcn', 'hgt'], help='model to use')
    parser.add_argument('--model_name', default='gcn', help='Path to saved model parameter to load')
    parser.add_argument('--hid_dim', default=16, type=int, help='hidden dimension for gnn')
    parser.add_argument('--out_dim', default=16, type=int, help='output dimension for gnn')
    parser.add_argument('--n_layers', default=1, type=int, help='number of layers in model')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of iterations to train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--save', default=None, help='Path to saved model parameter to load')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    return parser.parse_args()

def main(args):

    # set seed
    # seed = saved_dict['seed']
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if 'cuda' in args.device:
        device = torch.device(args.device)


    x, adj, node_types, labels, train_val_test_idx = load_data(args.filepath)

    n_classes = len(np.unique(labels))
    n_fts = [feat.shape[-1] for feat in x]

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    adj = adj.todense()

    nll_loss = nn.NLLLoss()

    x = [torch.tensor(feat, dtype=torch.float32).unsqueeze(0) for feat in x]
    adj = torch.tensor(adj, dtype=torch.float32).unsqueeze(0)
    node_types = torch.tensor(node_types, dtype=torch.int).unsqueeze(0)
    labels = torch.tensor(labels, device=device)

    dataset = HetGraphDataset(x=x, adj=adj, node_types=node_types, aug_type=None, device=device)
    dataloader = DataLoader(dataset, 1)

    nll_loss = nn.NLLLoss()

    accs = []
    micro_f1s = []
    macro_f1s = []

    tot = 0
    iters = 10

    for i in range(iters):
        best = 1e9
        best_t = 0
        patience = 10
        cnt_wait = 0

        if args.model == 'gcn':
            model = GCN(in_dim=n_fts, hid_dim=args.hid_dim, out_dim=args.out_dim,
                            n_layers=args.n_layers, dropout=args.dropout)

            model = ModelFinetune(model, n_classes)
            # graphcl = GraphCL(gnn=gnn, head_dim=saved_dict['head_dim'])

        elif args.model == 'hgt':
            num_node_types = len(np.unique(node_types))
            num_edge_types = len(np.unique(edge_types.values))
            model = HGT(in_dim=x.shape[-1], hid_dim=args.hid_dim, out_dim=n_classes, n_layers=args.n_layers,
                        num_node_types=num_node_types, num_edge_types=num_edge_types, n_heads=8, dropout=args.dropout)
            # graphcl = HetGraphCL(gnn=gnn, head_dim=saved_dict['head_dim'])

        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
        #                                                  factor=0.1,
        #                                                  patience=10,
        #                                                  verbose=False)
        for epoch in range(args.epochs):
            for x, adj in dataloader:

                loss = model.train_step(dataloader, labels, nll_loss, opt, train_idx)
                val_loss, _ = model.eval_step(dataloader, labels, nll_loss, val_idx)


            # epoch_loss = torch.mean(torch.tensor(losses))
            # scheduler.step(val_loss)
            print(f'Epoch: {epoch+1}\tLoss: {loss}\tVal Loss: {val_loss}')

            if val_loss < best:
                best = val_loss
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
            save_dict['state_dict'] = model.state_dict()
            for key in args.__dict__.keys():
                save_dict[key] = args.__dict__[key]

            torch.save(save_dict, args.save)


        # if args.model == 'gcn':
        #     logits = model(x, adj)
        # elif args.model == 'hgt':
        #     logits = model(x, n_types, edge_index, e_types)

        test_loss, logits = model.eval_step(dataloader, labels, nll_loss, test_idx)
        preds = torch.argmax(logits, dim=1)
        test_lbls = labels[test_idx]
        
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        micro_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
        macro_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')

        print(f'Iter: {i}\tF1: {micro_f1}')
        tot += acc*100
        accs.append(acc * 100)
        micro_f1s.append(micro_f1*100)
        macro_f1s.append(macro_f1*100)

    print('Average accuracy:[{:.4f}]'.format(tot / iters))
    accs = torch.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))

    print('Average Micro F1:[{:.4f}]'.format(np.sum(micro_f1s) / iters))
    micro_f1s = np.stack(micro_f1s)
    print('Mean:[{:.4f}]'.format(micro_f1s.mean().item()))
    print('Std :[{:.4f}]'.format(micro_f1s.std().item()))

    print('Average Macro F1:[{:.4f}]'.format(np.sum(macro_f1s) / iters))
    macro_f1s = np.stack(macro_f1s)
    print('Mean:[{:.4f}]'.format(macro_f1s.mean().item()))
    print('Std :[{:.4f}]'.format(macro_f1s.std().item()))

if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(args)
