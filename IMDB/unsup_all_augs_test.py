import os
import csv
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import torch
import torch.nn as nn

from utils.aug import *
from utils.data_utils import *
from utils import process
from models.graph_cl import GraphCL
from models.gcn import GCN
from models.logreg import LogReg
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


    x, adj, node_types, labels, train_val_test_idx = load_data(args.filepath)
    n_classes = len(np.unique(labels))



    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']


    # train_idx = [idx for idx in train_idx]
    # val_idx = [idx for idx in val_idx]
    # test_idx = [idx for idx in test_idx]

    # print(len(train_idx))
    # print(len(val_idx))
    # print(len(test_idx))

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = adj.todense()

    x = torch.tensor(x, dtype=torch.float32, device=device)
    adj = torch.tensor(adj, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    xent = nn.CrossEntropyLoss()

    print('--- Initialize Model ---')
    # print(saved_dict)

    results_table = []

    for aug1 in aug_choices:
        if aug1 == 'Identical':
            aug1 = None
        res = []
        for aug2 in aug_choices:
            if aug2 == 'Identical':
                aug2 = None

            print(f'----- Aug1: {aug1} | Aug2: {aug2} -----')
            load_path = os.path.join(args.load, f'{args.model_name}_a1_{aug1}_a2_{aug2}.pkl')
            saved_dict = torch.load(load_path)

            if args.model == 'gcn':
                model = GCN(in_dim=x.shape[-1], hid_dim=saved_dict['hid_dim'], out_dim=saved_dict['out_dim'],
                                n_layers=saved_dict['n_layers'], dropout=saved_dict['dropout'])

                model.load_state_dict(saved_dict['state_dict'])
                model.to(device)
                model.eval()
                embs = model(x, adj).detach()
                # print(embs.shape)

            elif args.model == 'hgt':
                num_node_types = len(np.unique(node_types))
                num_edge_types = len(np.unique(edge_types.values))
                model = HGT(in_dim=n_fts, hid_dim=saved_dict['hid_dim'], out_dim=saved_dict['out_dim'], n_layers=saved_dict['n_layers'],
                            num_node_types=num_node_types, num_edge_types=num_edge_types, n_heads=8, dropout=saved_dict['dropout'])

                model.load_state_dict(saved_dict['state_dict'])
                model.to(device)
                model.eval()

                edge_index = torch.where(adj > 0)
                edge_index = torch.stack(edge_index)
                e_types = torch.FloatTensor([edge_types[(node_types[n1.item()], node_types[n2.item()])] for n1, n2 in zip(edge_index[0], edge_index[1])])
                n_types = torch.tensor(node_types, device=x.device)
                e_types = e_types.to(adj.device)

                embs = model(x, n_types, edge_index, e_types).detach()

            train_embs = embs[train_idx, :]
            val_embs = embs[val_idx, :]
            test_embs = embs[test_idx, :]

            train_lbls = labels[train_idx]
            val_lbls = labels[val_idx]
            test_lbls = labels[test_idx]

            accs = []
            micro_f1s = []
            macro_f1s = []

            tot = 0
            iter = 20
            for iter in range(iter):

                if args.clf == 'svm':

                    svm = LinearSVC(random_state=iter)
                    svm = svm.fit(train_embs.cpu().numpy(), train_lbls.cpu().numpy())
                    preds = svm.predict(test_embs.cpu().numpy())

                    acc = np.sum(preds == test_lbls.cpu().numpy()) / test_lbls.shape[0]
                    micro_f1 = f1_score(test_lbls.cpu(), preds, average='micro')
                    macro_f1 = f1_score(test_lbls.cpu(), preds, average='macro')

                elif args.clf == 'logreg':
                    log_reg = LogReg(in_dim=embs.shape[1], out_dim=n_classes)
                    opt = torch.optim.Adam(log_reg.parameters(), lr=args.lr, weight_decay=0.0)
                    log_reg.to(device)

                    # log_reg = LogisticRegression(random_state=iter).fit(train_embs.cpu(), train_lbls.cpu())

                    for _ in range(args.epochs):
                        # for x, y in train_dataloader:
                        log_reg.train()
                        opt.zero_grad()

                        logits = log_reg(embs[train_idx, :])
                        loss = xent(logits, train_lbls)

                        loss.backward()
                        opt.step()

                    log_reg.eval()
                    logits = log_reg(test_embs)
                    preds = torch.argmax(logits, dim=1)

                    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
                    micro_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
                    macro_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')


                accs.append(acc * 100)
                micro_f1s.append(micro_f1*100)
                macro_f1s.append(macro_f1*100)

                # print('Iter: {}\tLoss: {:.4f}\tacc:[{:.4f}]'.format(iter+1, 0, acc*100))
                tot += acc*100


            print('Average accuracy:[{:.4f}]'.format(tot / iter))
            accs = np.stack(accs)
            print('Mean:[{:.4f}]'.format(accs.mean().item()))
            print('Std :[{:.4f}]'.format(accs.std().item()))

            print('Average Micro F1:[{:.4f}]'.format(np.sum(micro_f1s) / iter))
            micro_f1s = np.stack(micro_f1s)
            print('Mean:[{:.4f}]'.format(micro_f1s.mean().item()))
            print('Std :[{:.4f}]'.format(micro_f1s.std().item()))

            print('Average Macro F1:[{:.4f}]'.format(np.sum(macro_f1s) / iter))
            macro_f1s = np.stack(macro_f1s)
            print('Mean:[{:.4f}]'.format(macro_f1s.mean().item()))
            print('Std :[{:.4f}]'.format(macro_f1s.std().item()))

            res.append('{:.2f} +- {:.2f}'.format(tot/iter, accs.std().item()))

        results_table.append(res)

    results_table = pd.DataFrame(results_table, columns=aug_choices, index=aug_choices)
    print(results_table)
    table = create_table(results_table, index=True)
    table.update_layout(
        autosize=False,
        width=2800,
        height=1000
    )
    table.write_image(os.path.join(args.load, f'{args.model_name}_unsup_results_table.png'))

if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(args)
