import os
import csv
import argparse
import numpy as np
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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/IMDB_processed', help='path to dataset')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of iterations to train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--load', default=None, help='Path to saved model parameter to load')
    parser.add_argument('--clf', default='svm', help='classifier', choices=['logreg', 'svm'])
    return parser.parse_args()



def main(args):

    saved_dict = torch.load(args.load)

    print('-'*50)
    print('Load Model w/ Training Config')
    print('-'*50)

    [print(f'{k}: {saved_dict[k]}') for k in saved_dict if k != 'state_dict']
    print('-'*50)

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


    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']


    train_idx = [idx for idx in train_idx]
    val_idx = [idx for idx in val_idx]
    test_idx = [idx for idx in test_idx]

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
    model = GCN(in_dim=x.shape[-1], hid_dim=saved_dict['hid_dim'], out_dim=saved_dict['out_dim'], n_layers=saved_dict['n_layers'])
    # print(model.state_dict())
    # print(saved_dict['state_dict'])
    model.load_state_dict(saved_dict['state_dict'])

    model.to(device)
    # print(model)

    model.eval()

    embs = model(x, adj).detach()
    print(embs.shape)

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
    iter = 10
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

        print('Iter: {}\tLoss: {:.4f}\tacc:[{:.4f}]'.format(iter+1, 0, acc*100))
        tot += acc*100

    print('Average accuracy:[{:.4f}]'.format(tot / 50))
    accs = np.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))

    print('Average Micro F1:[{:.4f}]'.format(np.sum(micro_f1s) / 50))
    micro_f1s = np.stack(micro_f1s)
    print('Mean:[{:.4f}]'.format(micro_f1s.mean().item()))
    print('Std :[{:.4f}]'.format(micro_f1s.std().item()))

    print('Average Macro F1:[{:.4f}]'.format(np.sum(macro_f1s) / 50))
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
