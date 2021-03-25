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

aug_choices = [None, 'dropN', 'dropE', 'maskN',
                'dropN_metapath', 'dropE_metapath', 'subgraph_metapath', 'subgraph_metapath_list',
                'dropN_not_on_metapath', 'dropE_not_on_metapath', 'subgraph_not_on_metapath', 'subgraph_not_on_metapath_list']

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/IMDB_processed', help='path to dataset')
    parser.add_argument('--model', default='gcn', choices=['gcn', 'hgt'], help='model to use')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of iterations to train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--load', default=None, help='Path to saved model parameter to load')
    parser.add_argument('--model_name', default='gcn', help='Path to saved model parameter to load')
    parser.add_argument('--clf', default='svm', help='classifier', choices=['logreg', 'svm'])
    return parser.parse_args()



def main(args):

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if 'cuda' in args.device:
        device = torch.device(args.device)

    print('-'*50)
    print('Load Model w/ Training Config')
    print('-'*50)

    config = os.path.join(args.load, f'{args.model_name}_a1_None_a2_None.pkl')
    saved_dict = torch.load(config)

    [print(f'{k}: {saved_dict[k]}') for k in saved_dict if k != 'state_dict']
    print('-'*50)

    # print(saved_dict['state_dict'])
    x, adj, node_types, labels, train_val_test_idx = load_data(args.filepath)
    n_classes = len(np.unique(labels))

    edge_types = {
        (0, 0): 0,
        (1, 1): 0,
        (2, 2): 0,
        (0, 1): 1,
        (1, 0): 1,
        (0, 2): 2,
        (2, 0): 2
    }
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']


    train_idx = [idx for idx in train_idx]
    val_idx = [idx for idx in val_idx]
    test_idx = [idx for idx in test_idx]

    # print(len(train_idx))
    # print(len(val_idx))
    # print(len(test_idx))

    # adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = adj.todense()

    x = torch.tensor(x, dtype=torch.float32, device=device)
    adj = torch.tensor(adj, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    x = x.unsqueeze(0)
    adj = adj.unsqueeze(0)

    dataset = HetGraphDataset(x=x, adj=adj, node_types=node_types, edge_types=edge_types, aug_type=None, device=device)
    dataloader = DataLoader(dataset, args.batch_size)

    # train_lbls = labels[train_idx]
    # val_lbls = labels[val_idx]
    # test_lbls = labels[test_idx]

    iters = 10

    results_table = []
    # xent = nn.CrossEntropyLoss()
    nll_loss = nn.NLLLoss()

    for aug1 in aug_choices:
        if aug1 == 'Identical':
            aug1 = None
        res = []
        for aug2 in aug_choices:
            if aug2 == 'Identical':
                aug2 = None

            accs = []
            micro_f1s = []
            macro_f1s = []

            tot = 0
            load_path = os.path.join(args.load, f'{args.model_name}_a1_{aug1}_a2_{aug2}.pkl')
            saved_dict = torch.load(load_path)

            print(f'----- Aug1: {aug1} | Aug2: {aug2} -----')

            for i in range(iters):
                best = 1e9
                best_t = 0
                patience = 10

                if args.model == 'gcn':
                    gnn = GCN(in_dim=x.shape[-1], hid_dim=saved_dict['hid_dim'], out_dim=saved_dict['out_dim'], n_layers=saved_dict['n_layers'],
                                dropout=saved_dict['dropout'])

                    # graphcl = GraphCL(gnn=gnn, head_dim=saved_dict['head_dim'])
                    gnn.load_state_dict(saved_dict['state_dict'])
                    model = ModelFinetune(gnn, n_classes)

                elif args.model == 'hgt':
                    num_node_types = len(np.unique(node_types))
                    num_edge_types = len(np.unique(edge_types.values))
                    gnn = HGT(in_dim=x.shape[-1], hid_dim=saved_dict['hid_dim'], out_dim=saved_dict['out_dim'], n_layers=saved_dict['n_layers'],
                                num_node_types=num_node_types, num_edge_types=num_edge_types, n_heads=8, dropout=saved_dict['dropout'])
                    # graphcl = HetGraphCL(gnn=gnn, head_dim=saved_dict['head_dim'])
                    gnn.load_state_dict(saved_dict['state_dict'])
                    model = HetModelFinetune(gnn, n_classes)


                model.to(device)

                opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
                # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                #                                                  factor=0.1,
                #                                                  patience=10,
                #                                                  verbose=False)
                for epoch in range(args.epochs):

                    loss = model.train_step(dataloader, labels, nll_loss, opt, train_idx)

                    with torch.no_grad():
                        val_loss, logits = model.eval_step(dataloader, labels, nll_loss, val_idx)
                        # logits = model(x, adj)
                        # logits = logits[val_idx]
                        # logits = F.log_softmax(logits, dim=1)
                        # val_loss = nll_loss(logits, val_lbls)

                    # epoch_loss = torch.mean(torch.tensor(losses))
                    # scheduler.step(val_loss)
                    # print(f'Epoch: {epoch+1}\tLoss: {loss}\tVal Loss: {val_loss}')

                    if val_loss < best:
                        best = val_loss
                        best_t = epoch
                        cnt_wait = 0
                        # torch.save(model.state_dict(), args.save_name)
                    else:
                        cnt_wait += 1

                    if cnt_wait == patience:
                        # print('Early stopping!')
                        break


                loss, logits = model.eval_step(dataloader, labels, nll_loss, test_idx)
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
            # print('Mean:[{:.4f}]'.format(micro_f1s.mean().item()))
            print('Std :[{:.4f}]'.format(micro_f1s.std().item()))

            print('Average Macro F1:[{:.4f}]'.format(np.sum(macro_f1s) / iters))
            macro_f1s = np.stack(macro_f1s)
            # print('Mean:[{:.4f}]'.format(macro_f1s.mean().item()))
            print('Std :[{:.4f}]'.format(macro_f1s.std().item()))
            print()
            res.append('{:.2f} +- {:.2f}'.format(tot/iters, accs.std().item()))

        results_table.append(res)

    results_table = pd.DataFrame(results_table, columns=aug_choices, index=aug_choices)
    print(results_table)
    table = create_table(results_table, index=True)
    table.update_layout(
        autosize=False,
        width=2800,
        height=1000
    )
    table.write_image(os.path.join(args.load, f'{args.model_name}_semisup_results_table.png'))

if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(args)
