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
from models.gcn import GraphCL, GCN_Finetune
from models.logreg import LogReg
from sklearn.metrics import f1_score

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath', default='data/IMDB_processed', help='path to dataset')
    parser.add_argument('--aug1', default='node', choices=['node', 'edge', 'subgraph', 'mask', 'meta_subgraph'],
                            help='type of first augmentation to apply')
    parser.add_argument('--aug2', default='meta_subgraph', choices=['node', 'edge', 'subgraph', 'mask', 'meta_subgraph'],
                        help='type of second augmentation to apply')
    parser.add_argument('--hid_dim', default=512, type=int, help='hidden dimension for layers')
    parser.add_argument('--head_dim', default=300, type=int, help='projection head dimension')
    parser.add_argument('--n_layers', default=1, type=int, help='number of layers in model')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of iterations to train')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--metapath_file', default=None, help='file to a list of metapaths')
    parser.add_argument('--device', default='cuda:0', help='specify device to use')
    parser.add_argument('--seed', type=int, default=39, help='seed')
    parser.add_argument('--save', default=None, help='Path to save file')
    parser.add_argument('--load', default=None, help='Path to saved model parameter to load')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout')
    return parser.parse_args()



def main(args):

    if 'cuda' in args.device:
        device = torch.device(args.device)

    x, adj, node_types, labels, train_val_test_idx = load_data(args.filepath)
    n_classes = len(np.unique(labels))

    print(labels.shape)
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    metapath1 = [2, 0, 1, 0, 2]
    metapath2 = [2, 0, 2]

    print('--- Apply First Augmentation ---')
    aug_adj1 = aug_metapath_subgraph(adj, node_types, metapath2)

    print('--- Apply Second Augmentation ---')
    aug_adj2 = aug_metapath_subgraph(adj, node_types, metapath2)

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
    aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

    adj = adj.todense()
    aug_adj1 = aug_adj1.todense()
    aug_adj2 = aug_adj2.todense()

    x = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    adj = torch.tensor(adj, dtype=torch.float32, device=device).unsqueeze(0)
    aug_adj1 = torch.tensor(aug_adj1, dtype=torch.float32, device=device).unsqueeze(0)
    aug_adj2 = torch.tensor(aug_adj2, dtype=torch.float32, device=device).unsqueeze(0)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    xent = nn.CrossEntropyLoss()

    print('--- Initialize Model ---')

    if args.load:
        saved_dict = torch.load(args.load)
        model = GraphCL(in_dim=x.shape[2], hid_dim=saved_dict['hid_dim'], out_dim=saved_dict['head_dim'], n_layers=saved_dict['n_layers'])
        model.load_state_dict(saved_dict['state_dict'])
        model.to(device)
        print(saved_dict)
    else:
        model = GraphCL(in_dim=x.shape[2], hid_dim=args.hid_dim, out_dim=args.head_dim, n_layers=args.n_layers)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                         factor=0.1,
                                                         patience=20,
                                                         verbose=True)

        print('--- Start Contrastive Learning --- ')
        for epoch in range(args.epochs):

            opt.zero_grad()
            out1 = model(x, adj)
            out2 = model(x, aug_adj2)

            out1 = out1.squeeze(0)
            out2 = out2.squeeze(0)
            loss = model.loss_cl(out1, out2)
            loss.backward()
            opt.step()

            scheduler.step(loss)
            print(f'Epoch: {epoch+1}\tLoss: {loss}')

        if args.save:
            save_dict = {}
            save_dict['state_dict'] = model.state_dict()
            for key in args.__dict__.keys():
                save_dict[key] = args.__dict__[key]

            torch.save(save_dict, args.save)

    model.eval()
    embs = model.embed(x, adj).detach()
    # train_embs = embs[:, train_idx, :]
    # val_embs = embs[:, val_idx, :]
    # test_embs = embs[:, test_idx, :]

    train_lbls = labels[train_idx]
    val_lbls = labels[val_idx]
    test_lbls = labels[test_idx]

    accs = []
    micro_f1s = []
    macro_f1s = []

    tot = 0

    train_mask = torch.ones(embs.shape)
    train_mask[:, val_idx, :] = 0
    train_mask[:, test_idx, :] = 0

    test_mask = torch.ones(embs.shape)
    test_mask[:, val_idx, :] = 0
    test_mask[:, train_idx, :] = 0

    train_adj_mask = torch.clone(adj)
    train_adj_mask[:, val_idx, :] = 0
    train_adj_mask[:, val_idx, :] = 0
    train_adj_mask[:, test_idx, :] = 0
    train_adj_mask[:, test_idx, :] = 0

    test_adj_mask = torch.clone(adj)
    test_adj_mask[:, val_idx, :] = 0
    test_adj_mask[:, val_idx, :] = 0
    test_adj_mask[:, test_idx, :] = 0
    test_adj_mask[:, test_idx, :] = 0

    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)
    train_adj_mask = train_adj_mask.to(device)
    test_adj_mask = test_adj_mask.to(device)

    embs = embs.squeeze(0)
    for iter in range(50):

        model_finetune = LogReg(in_dim=embs.shape[1], hid_dim=512, out_dim=n_classes)
        # model_finetune = GCN_Finetune(in_dim=embs.shape[2], hid_dim=args.hid_dim, out_dim=n_classes, n_layers=args.n_layers, dropout=args.dropout)
        opt = torch.optim.Adam(model_finetune.parameters(), lr=0.01, weight_decay=0.0)
        model_finetune.to(device)


        for _ in range(150):
            model_finetune.train()
            opt.zero_grad()

            logits = model_finetune(embs[train_idx, :])
            # logits = model_finetune(embs, adj, mask=[train_mask, train_adj_mask])
            # logits = logits.squeeze(0)
            # logits = logits[train_idx]
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        model_finetune.eval()
        logits = model_finetune(embs[test_idx, :])
        # logits = model_finetune(embs, adj, [test_mask, test_adj_mask])
        # logits = logits.squeeze(0)
        # logits = logits[test_idx]
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        micro_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')
        macro_f1 = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')

        accs.append(acc * 100)
        micro_f1s.append(micro_f1*100)
        macro_f1s.append(macro_f1*100)

        print('Iter: {}\tLoss: {:.4f}\tacc:[{:.4f}]'.format(iter+1, loss, acc*100))
        tot += acc*100

    print('Average accuracy:[{:.4f}]'.format(tot / 50))
    accs = torch.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))

    print('Average Micro F1:[{:.4f}]'.format(np.sum(micro_f1s) / 50))
    micro_f1s = np.stack(micro_f1s)
    print('Mean:[{:.4f}]'.format(micro_f1s.mean().item()))
    print('Std :[{:.4f}]'.format(micro_f1s.std().item()))

    print('Average Macro F1:[{:.4f}]'.format(np.sum(macro_f1s) / 50))
    macro_f1s = np.stack(macro_f1s)
    print('Mean:[{:.4f}]'.format(micro_f1s.mean().item()))
    print('Std :[{:.4f}]'.format(micro_f1s.std().item()))

if __name__ == '__main__':
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    main(args)
