import os
import csv
import argparse
import numpy as np
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
from models.graph_cl import GraphCL
from models.gcn import GCN
from models.logreg import LogReg
from models.model_finetune import ModelFinetune
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def parse_args():

    aug_choices = ['Identical', 'dropN', 'dropE', 'maskN', 'dropN_metapath', 'dropE_metapath', 'subgraph_metapath', 'subgraph_metapath_list']

    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/IMDB_processed', help='path to dataset')
    parser.add_argument('--hid_dim', default=32, type=int, help='hidden dimension for gnn')
    parser.add_argument('--out_dim', default=32, type=int, help='output dimension for gnn')
    parser.add_argument('--head_dim', default=32, type=int, help='projection head dimension')
    parser.add_argument('--n_layers', default=1, type=int, help='number of layers in model')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--epochs', default=200, type=int, help='number of iterations to train')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--metapath', default=None, help='file to a list of metapaths')
    parser.add_argument('--metapath_list', default=None, help='file to a list of metapaths')
    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--load', default=None, help='Path to saved model parameter to load')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
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

    adj = adj.todense()

    x = torch.tensor(x, dtype=torch.float32)
    adj = torch.tensor(adj, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    nll_loss = nn.NLLLoss()

    x = x.unsqueeze(0)
    adj = adj.unsqueeze(0)
    dataset = HetGraphDataset(x=x, adj=adj, node_types=node_types, aug_type=None, device=device)
    dataloader = DataLoader(dataset, args.batch_size)

    train_lbls = labels[train_idx]
    val_lbls = labels[val_idx]
    test_lbls = labels[test_idx]

    accs = []
    micro_f1s = []
    macro_f1s = []

    tot = 0
    iters = 20

    for i in range(iters):
        best = 1e9
        best_t = 0
        patience = 10

        gnn = GCN(in_dim=x.shape[-1], hid_dim=saved_dict['hid_dim'], out_dim=saved_dict['out_dim'],
                        n_layers=saved_dict['n_layers'], dropout=saved_dict['dropout'])
        # print(model.state_dict())
        gnn.load_state_dict(saved_dict['state_dict'])
        # print(saved_dict['state_dict'])
        # graphCL = GraphCL(gnn, saved_dict['head_dim'])
        # graphCL.load_state_dict(saved_dict['state_dict'])
        # gnn = graphCL.gnn
        # state_dict = gnn.state_dict()
        # loaded_state_dict = {k:v for k,v in saved_dict['state_dict'].items() if k in state_dict.keys()}
        model = ModelFinetune(gnn, n_classes)
        model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
        #                                                  factor=0.1,
        #                                                  patience=10,
        #                                                  verbose=False)
        for epoch in range(args.epochs):
            for x, adj in dataset:

                x = x.squeeze(0)
                adj = adj.squeeze(0)

                opt.zero_grad()

                logits = model(x, adj)
                logits = logits[train_idx]
                logits = F.log_softmax(logits, dim=1)
                loss = nll_loss(logits, train_lbls)

                # losses.append(loss)

                loss.backward()
                opt.step()

                with torch.no_grad():
                    logits = model(x, adj)
                    logits = logits[val_idx]
                    logits = F.log_softmax(logits, dim=1)
                    val_loss = nll_loss(logits, val_lbls)

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
                print('Early stopping!')
                break

        model.eval()
        logits = model(x, adj)
        logits = logits[test_idx]
        preds = torch.argmax(logits, dim=1)

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
