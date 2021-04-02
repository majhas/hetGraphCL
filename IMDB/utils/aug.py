import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

class Augmentor():
    def __init__(self, aug_ratio=0.2, metapath=None):
        self.aug_ratio = aug_ratio
        self.metapath = metapath

    def apply_aug(x, adj, node_types=None, edge_types=None, return_keep=False):

        if self.aug_type == None:
            aug_x, aug_adj = x, adj
        elif self.aug_type == 'dropN':
            aug_x, aug_adj = drop_nodes(x, adj, self.aug_ratio)
        elif self.aug_type == 'dropE':
            aug_x, aug_adj = drop_edges(x, adj, self.aug_ratio)
        elif self.aug_type == 'maskN':
            aug_x, aug_adj = mask_nodes(x, adj, self.aug_ratio)
        elif self.aug_type == 'subgraph':
            aug_x, aug_adj = subgraph(x, adj, self.aug_ratio)
        elif self.aug_type == 'dropN_metapath':
            aug_x, aug_adj = drop_node_types(x, adj, node_types, self.metapath, self.aug_ratio)
        elif self.aug_type == 'dropE_metapath':
            aug_x, aug_adj = drop_edge_types(x, adj, node_types, self.metapath, self.aug_ratio)
        elif self.aug_type == 'subgraph_metapath':
            aug_x, aug_adj = subgraph_metapath(x, adj, node_types, self.metapath, self.aug_ratio)
        elif self.aug_type == 'subgraph_metapath_list':
            aug_x, aug_adj = subgraph_metapath_list(x, adj, node_types, self.metapath, self.aug_ratio)
        elif self.aug_type == 'dropN_not_on_metapath':
            aug_x, aug_adj = drop_node_types(x, adj, node_types, self.metapath, self.aug_ratio, inverse=True)
        elif self.aug_type == 'dropE_not_on_metapath':
            aug_x, aug_adj = drop_edge_types(x, adj, node_types, self.metapath, self.aug_ratio, inverse=True)
        elif self.aug_type == 'subgraph_not_on_metapath':
            aug_x, aug_adj = subgraph_metapath(x, adj, node_types, self.metapath, self.aug_ratio, inverse=True)
        elif self.aug_type == 'subgraph_not_on_metapath_list':
            aug_x, aug_adj = subgraph_metapath_list(x, adj, node_types, self.metapath, self.aug_ratio, inverse=True)





def mask_nodes(x, adj, aug_ratio=0.2):

    num_nodes = adj.shape[0]
    drop_num = int(num_nodes  * aug_ratio)
    nodes = np.arange(num_nodes)

    idx_perm = np.random.permutation(num_nodes)

    nodes = nodes[idx_perm]
    nodes_dropped = nodes[:drop_num]
    nodes = nodes[drop_num:]

    aug_x = torch.clone(x)
    aug_x[nodes_dropped] = 0

    return aug_x, adj


def drop_edges(x, adj, aug_ratio=0.2):

    edge_idx = torch.nonzero(adj)

    num_edges = len(edge_idx[0])
    drop_num = int(num_edges  * aug_ratio)

    idx_perm = np.random.permutation(num_edges)

    edge_idx1 = edge_idx[0][idx_perm]
    edge_idx2 = edge_idx[1][idx_perm]

    edges_dropped1 = edge_idx1[:drop_num]
    edges_dropped2 = edge_idx2[:drop_num]

    edges_idx1 = edge_idx1[drop_num:]
    edges_idx2 = edge_idx2[drop_num:]

    aug_adj = torch.clone(adj)
    aug_adj[edges_dropped1, edges_dropped2] = 0
    # aug_adj[edges_dropped2, edges_dropped1] = 0

    return x, aug_adj

def drop_edge_types(x, adj, node_types, metapath, aug_ratio=0.2, inverse=False):

    ''' TODO '''

    _, adj_meta = subgraph_metapath(x, adj, node_types, metapath, inverse=inverse, drop=False)
    edge_idx = torch.nonzero(adj_meta)

    num_edges = len(edge_idx[0])
    drop_num = int(num_edges  * aug_ratio)

    idx_perm = np.random.permutation(num_edges)

    edge_idx1 = edge_idx[0][idx_perm]
    edge_idx2 = edge_idx[1][idx_perm]

    edges_dropped1 = edge_idx1[:drop_num]
    edges_dropped2 = edge_idx2[:drop_num]

    # edges_idx1 = edge_idx1[drop_num:]
    # edges_idx2 = edge_idx2[drop_num:]

    aug_adj = torch.clone(adj)
    aug_adj[edges_dropped1, edges_dropped2] = 0
    aug_adj[edges_dropped2, edges_dropped1] = 0
    return x, aug_adj

def drop_nodes(x, adj, aug_ratio=0.2):

    num_nodes = adj.shape[0]
    # print(num_nodes)
    drop_num = int(num_nodes  * aug_ratio)
    nodes = np.arange(num_nodes)
    idx_perm = np.random.permutation(num_nodes)

    nodes = nodes[idx_perm]
    nodes_dropped = nodes[:drop_num]
    nodes = nodes[drop_num:]

    aug_x = torch.clone(x)
    aug_x[nodes_dropped] = 0

    aug_adj = torch.clone(adj)
    aug_adj[nodes_dropped, :] = 0
    aug_adj[:, nodes_dropped] = 0

    return aug_x, aug_adj

def drop_node_types(x, adj, node_types, metapath, aug_ratio=0.2, inverse=False):

    if inverse:
        targeted_nodes = np.array([idx for idx, node_type in enumerate(node_types) if node_type not in metapath])
    else:
        targeted_nodes = np.array([idx for idx, node_type in enumerate(node_types) if node_type in metapath])

    num_nodes = len(targeted_nodes)
    drop_num = int(num_nodes  * aug_ratio)

    idx_perm = np.random.permutation(num_nodes)

    targeted_nodes = targeted_nodes[idx_perm]
    nodes_dropped = targeted_nodes[:drop_num]
    targeted_nodes = targeted_nodes[drop_num:]

    aug_x = torch.clone(x)
    aug_x[nodes_dropped] = 0

    aug_adj = torch.clone(adj)
    aug_adj[nodes_dropped, :] = 0
    aug_adj[:, nodes_dropped] = 0

    return aug_x, aug_adj

def subgraph(x, adj, aug_ratio=0.2):

    edge_idx = torch.where(adj == 1)

    nodes = torch.unique(torch.cat((edge_idx[0], edge_idx[1])))
    num_nodes = len(nodes)
    drop_num = int(num_nodes  * aug_ratio)
    keep_num = num_nodes - drop_num
    sample = np.random.choice(nodes)

    adj = adj.type(torch.LongTensor)
    aug_adj = torch.zeros(adj.shape, device=adj.device)

    import time
    for i in range(keep_num):

        neighbors = torch.where(adj[sample] == 1)[0]
        # print(neighbors)
        # print(neighbors.shape)
        sample_idx = torch.LongTensor([sample]).expand(neighbors.shape)
        # print(sample_idx)
        aug_adj[sample_idx, neighbors] = 1

        if len(neighbors) == 0:
            sample = np.random.choice(nodes)
        else:
            sample = np.random.choice(neighbors)

        if len(torch.where(aug_adj == 1)[0]) > keep_num:
            break
        # start = time.time()
        #
        # if i < len(subset):
        #     sample = subset[i]
        # else:
        #     sample = np.random.choice([int(node) for node in nodes if int(node) not in subset])
        #     subset.append(sample)
        # print(f'Time: {time.time()-start}')

        # start = time.time()
        # neighbors = [int(n) for n in torch.where(adj[sample] == 1)[0] if int(n) not in subset]
        # print(f'Time: {time.time()-start}')
        # neighbors = [n for n in neighbors if n not in queue if n not in subset]
        # print(f'Queue: {queue}')
        # start = time.time()
        # subset = list(np.concatenate((subset, neighbors)).astype(int))

        # print(len(subset))
        # if len(subset) >= keep_num:
        #     subset = subset[:keep_num]
        #     break

        # print(f'Time: {time.time()-start}')
        # if len(queue) == 0:
        #     n = np.random.choice([int(node) for node in nodes if int(node) not in subset])
        #     queue.append(n)

    print(f'SUbset: {subset}')
    nodes_dropped = [int(node) for node in nodes if int(node) not in subset]
    print(len(nodes_dropped))
    print(num_nodes)
    aug_x = torch.clone(x)
    aug_x[nodes_dropped] = 0

    aug_adj = torch.clone(adj)
    aug_adj[nodes_dropped, :] = 0
    aug_adj[:, nodes_dropped] = 0

    return aug_x, aug_adj

def subgraph_metapath(x, adj, node_types, metapath, aug_ratio=0.2, inverse=False, drop=True):

    aug_adj = torch.zeros(adj.shape)

    for i in range(len(metapath)-1):

        source_type = metapath[i]
        target_type = metapath[i+1]

        if inverse:
            targets = np.where(node_types != target_type)[0]
        else:
            targets = np.where(node_types == target_type)[0]

        for t in targets:

            neighbors = torch.nonzero(adj[t], as_tuple=False).squeeze(1).tolist()
            neighbors = list(set(neighbors))

            if inverse:
                neighbors = np.array([n for n in neighbors if node_types[n] != source_type])
            else:
                neighbors = np.array([n for n in neighbors if node_types[n] == source_type])

            if len(neighbors) > 0:
                aug_adj[t, neighbors] = 1

    seen = set(torch.flatten(torch.nonzero(aug_adj, as_tuple=False)).tolist())
    if inverse:
        drop_node_list = [idx for idx in range(len(adj)) if idx in seen]
    else:
        # drop_node_list = [idx for idx, node_type in enumerate(node_types) if node_type not in metapath]
        # print(len(drop_node_list))
        drop_node_list = [idx for idx in range(len(adj)) if idx in seen]

    aug_x = torch.clone(x)
    aug_x[drop_node_list] = 0

    if drop:
        aug_x, aug_adj = drop_nodes(aug_x, aug_adj, aug_ratio)

    return aug_x, aug_adj

def subgraph_metapath_list(x, adj, node_types, metapath_list, aug_ratio=0.2, inverse=False):

    num_metapaths = len(metapath_list)
    rand_metapath = np.random.randint(num_metapaths)
    metapath = metapath_list[rand_metapath]
    return subgraph_metapath(x, adj, node_types, metapath, aug_ratio=aug_ratio, inverse=inverse)



if __name__ == "__main__":
    main()
