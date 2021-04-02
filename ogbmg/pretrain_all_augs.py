import argparse
from tqdm import tqdm
import sys
import os

from sklearn.metrics import f1_score
from pyHGT.data import *
from pyHGT.model import *
from warnings import filterwarnings
filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, ParameterDict, Parameter
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from utils.aug import Augmentor

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

aug_choices = ['dropN', 'dropE', 'maskN',
                'dropN_metapath', 'dropE_metapath', 'subgraph_metapath']


parser = argparse.ArgumentParser(description='Training GNN on ogbn-mag benchmark')

parser.add_argument('--data_dir', type=str, default='data/OGB_MAG.pk',
                    help='The address of preprocessed graph.')
parser.add_argument('--model_dir', type=str, default='./hgt_4layer',
                    help='The address for storing the trained models.')
parser.add_argument('--plot', action='store_true',
                    help='Whether to plot the loss/acc curve')
parser.add_argument('--cuda', type=int, default=0,
                    help='Avaiable GPU ID')
parser.add_argument('--conv_name', type=str, default='hgt',
                    choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
                    help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
parser.add_argument('--n_hid', type=int, default=512,
                    help='Number of hidden dimension')
parser.add_argument('--n_heads', type=int, default=8,
                    help='Number of attention head')
parser.add_argument('--n_layers', type=int, default=4,
                    help='Number of GNN layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout ratio')
parser.add_argument('--sample_depth', type=int, default=6,
                    help='How many numbers to sample the graph')
parser.add_argument('--sample_width', type=int, default=520,
                    help='How many nodes to be sampled per layer per type')

parser.add_argument('--n_epoch', type=int, default=100,
                    help='Number of epoch to run')
parser.add_argument('--n_pool', type=int, default=8,
                    help='Number of process to sample subgraph')
parser.add_argument('--n_batch', type=int, default=32,
                    help='Number of batch (sampled graphs) for each epoch')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of output nodes for training')
parser.add_argument('--clip', type=float, default=1.0,
                    help='Gradient Norm Clipping')

parser.add_argument('--prev_norm', help='Whether to add layer-norm on the previous layers', action='store_true')
parser.add_argument('--last_norm', help='Whether to add layer-norm on the last layers',     action='store_true')
parser.add_argument('--use_RTE',   help='Whether to use RTE',     action='store_true')


parser.add_argument('--aug_ratio', default=0.2, type=float, help='augmentation strength')
parser.add_argument('--metapath', default=None, help='file to a list of metapaths')
parser.add_argument('--head_dim', default=300, type=int, help='projection head dimension')
parser.add_argument('--save', default=None, help='Path to save file')
parser.add_argument('--seed', default=123, type=int, help='seed')


args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def compute_diag_sum(inp):
    inp = torch.diagonal(inp)
    return torch.sum(inp)

def con_loss(x1, x2, temp=0.1):

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix /= temp
    # print(sim_matrix.shape)
    row_softmax_matrix = -F.log_softmax(sim_matrix, dim=1)

    colomn_softmax_matrix = -F.log_softmax(sim_matrix, dim=0)

    row_diag_sum = compute_diag_sum(row_softmax_matrix)
    colomn_diag_sum = compute_diag_sum(colomn_softmax_matrix)
    contrastive_loss = (row_diag_sum + colomn_diag_sum) / (2 * len(row_softmax_matrix))

    return contrastive_loss

def ogbn_sample(seed, samp_nodes):
    np.random.seed(seed)
    ylabel      = torch.LongTensor(graph.y[samp_nodes])
    feature, times, edge_list, indxs, _ = sample_subgraph(graph, \
                inp = {'paper': np.concatenate([samp_nodes, graph.years[samp_nodes]]).reshape(2, -1).transpose()}, \
                sampled_depth = args.sample_depth, sampled_number = args.sample_width, \
                    feature_extractor = feature_MAG)
    node_feature, node_type, edge_time, edge_index, edge_type, node_dict, edge_dict = \
            to_torch(feature, times, edge_list, graph)
    # print(node_dict)
    # print(edge_dict)
    train_mask = graph.train_mask[indxs['paper']]
    valid_mask = graph.valid_mask[indxs['paper']]
    test_mask  = graph.test_mask[indxs['paper']]
    ylabel     = graph.y[indxs['paper']]
    return node_feature, node_type, edge_time, edge_index, edge_type, (train_mask, valid_mask, test_mask), ylabel

def prepare_data(pool, task_type = 'train', s_idx = 0, n_batch = args.n_batch, batch_size = args.batch_size):
    '''
        Sampled and prepare training and validation data using multi-process parallization.
    '''
    jobs = []
    if task_type == 'train':
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), \
                            np.random.choice(target_nodes, args.batch_size, replace = False)]))
            jobs.append(p)
    elif task_type == 'sequential':
        for i in np.arange(n_batch):
            target_papers = graph.test_paper[(s_idx + i) * batch_size : (s_idx + i + 1) * batch_size]
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    elif task_type == 'variance_reduce':
        target_papers = graph.test_paper[s_idx * args.batch_size : (s_idx + 1) * args.batch_size]
        for batch_id in np.arange(n_batch):
            p = pool.apply_async(ogbn_sample, args=([randint(), target_papers]))
            jobs.append(p)
    return jobs

metapath = np.array(args.metapath.split(','), dtype=np.int)
graph = dill.load(open(args.data_dir, 'rb'))
evaluator = Evaluator(name='ogbn-mag')
device = torch.device('cuda:0')
target_nodes = np.arange(len(graph.node_feature['paper']))


pool = mp.Pool(args.n_pool)
st = time.time()
jobs = prepare_data(pool)

for aug1 in aug_choices:

    if aug1 != 'subgraph_metapath':
        continue

    for aug2 in aug_choices:

        if os.path.isfile(args.save.replace('.pkl', f'_a1_{aug1}_a2_{aug2}.pkl')):
            continue

        print('*'*100)
        print(f'Aug1: {aug1}\tAug2: {aug2}')
        print('*'*100)

        gnn = GNN(conv_name = args.conv_name, in_dim = len(graph.node_feature['paper'][0]), \
                  n_hid = args.n_hid, n_heads = args.n_heads, n_layers = args.n_layers, dropout = args.dropout,\
                  num_types = len(graph.get_types()), num_relations = len(graph.get_meta_graph()) + 1,\
                  prev_norm = args.prev_norm, last_norm = args.last_norm, use_RTE = args.use_RTE).float().to(device)

        proj_head = nn.Sequential(
                        nn.Linear(args.n_hid, args.head_dim),
                        nn.ReLU(),
                        nn.Linear(args.head_dim, args.head_dim)
                    ).to(device)

        model = nn.Sequential(gnn, proj_head).to(device)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],     'weight_decay': 0.0}
            ]


        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-06)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                                max_lr = 5e-4, total_steps = args.n_batch * args.n_epoch + 1)

        stats = []
        res   = []
        best_val   = 0
        train_step = 0
        cnt_wait = 0
        patience = 10
        best = 1e9

        for epoch in np.arange(args.n_epoch):
            '''
                Prepare Training and Validation Data
            '''
            datas = [job.get() for job in jobs]
            pool.close()
            pool.join()
            '''
                After the data is collected, close the pool and then reopen it.
            '''
            pool = mp.Pool(args.n_pool)
            jobs = prepare_data(pool)
            et = time.time()
            print('Data Preparation: %.1fs' % (et - st))

            '''
                Train
            '''
            model.train()
            losses = []
            for node_feature, node_type, edge_time, edge_index, edge_type, _, _ in datas:

                # print(node_feature.shape)
                augmentor = Augmentor(aug_ratio=args.aug_ratio, metapath=metapath)

                x1, edge_index1, edge_type1, edge_time1 = augmentor.apply_aug(x=node_feature, edge_index=edge_index, node_types=node_type,
                                                                                edge_types=edge_type, edge_attr=edge_time, aug_type=aug1)


                node_rep1 = gnn.forward(x1.to(device), node_type.to(device), \
                                       edge_time1.to(device), edge_index1.to(device), edge_type1.to(device))

                del x1, edge_index1, edge_type1, edge_time1

                x2, edge_index2, edge_type2, edge_time2 = augmentor.apply_aug(x=node_feature, edge_index=edge_index, node_types=node_type,
                                       edge_types=edge_type, edge_attr=edge_time, aug_type=aug2)


                node_rep2 = gnn.forward(x2.to(device), node_type.to(device), \
                                       edge_time2.to(device), edge_index2.to(device), edge_type2.to(device))

                del x2, edge_index2, edge_type2, edge_time2

                node_rep1 = proj_head(node_rep1)
                node_rep2 = proj_head(node_rep2)

                loss = con_loss(node_rep1, node_rep2)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                train_step += 1
                scheduler.step(train_step)

                losses.append(loss.detach().cpu().item())
                del node_rep1, node_rep2, loss

            avg_loss = np.mean(losses)
            print(f'Epoch: {epoch+1}\tLoss: {avg_loss}')

            if avg_loss < best:
                best = avg_loss
                cnt_wait = 0
                # torch.save(model.state_dict(), args.save_name)
            else:
                cnt_wait += 1

            if cnt_wait == patience:
                print('Early stopping!')
                break

        if args.save:
            save_dict = {}
            save_dict['state_dict'] = gnn.state_dict()
            for key in args.__dict__.keys():
                save_dict[key] = args.__dict__[key]

            save_dict['aug1'] = aug1
            save_dict['aug2'] = aug2

            torch.save(save_dict, args.save.replace('.pkl', f'_a1_{aug1}_a2_{aug2}.pkl'))
