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

    def apply_aug(self, x, edge_index, node_types=None, edge_types=None, edge_attr=None, aug_type=None):

        if aug_type == None:
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = x, edge_index, edge_types, edge_attr
        elif aug_type == 'dropN':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = drop_nodes(x, edge_index, edge_types,
                                                                        edge_attr, self.aug_ratio)
        elif aug_type == 'dropE':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = drop_edges(x, edge_index, edge_types,
                                                                        edge_attr, self.aug_ratio)
        elif aug_type == 'maskN':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = mask_nodes(x, edge_index, edge_types,
                                                                        edge_attr, self.aug_ratio)
        elif aug_type == 'subgraph':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = subgraph(x, edge_index, edge_types,
                                                                        edge_attr, self.aug_ratio)
        elif aug_type == 'dropN_metapath':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = drop_node_types(x, edge_index, node_types,
                                                                        edge_types, edge_attr, self.metapath, self.aug_ratio)
        elif aug_type == 'dropE_metapath':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = drop_edge_types(x, edge_index, node_types,
                                                                        edge_types, edge_attr, self.metapath, self.aug_ratio)
        elif aug_type == 'subgraph_metapath':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = subgraph_metapath(x, edge_index, node_types,
                                                                        edge_types, edge_attr, self.metapath, self.aug_ratio)
        elif aug_type == 'subgraph_metapath_list':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = subgraph_metapath_list(x, edge_index, node_types,
                                                                        edge_types, edge_attr, self.metapath, self.aug_ratio)
        elif aug_type == 'dropN_not_on_metapath':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = drop_node_types(x, edge_index, node_types,
                                                                        edge_types, edge_attr, self.metapath, self.aug_ratio, inverse=True)
        elif aug_type == 'dropE_not_on_metapath':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = drop_edge_types(x, edge_index, node_types,
                                                                        edge_types, edge_attr, self.metapath, self.aug_ratio, inverse=True)
        elif aug_type == 'subgraph_not_on_metapath':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = subgraph_metapath(x, edge_index, node_types,
                                                                        edge_types, edge_attr, self.metapath, self.aug_ratio, inverse=True)
        elif aug_type == 'subgraph_not_on_metapath_list':
            aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = subgraph_metapath_list(x, edge_index, node_types,
                                                                        edge_types, edge_attr, self.metapath, self.aug_ratio, inverse=True)

        return aug_x, aug_edge_index, aug_edge_types, aug_edge_attr


def mask_nodes(x, edge_index, edge_types, edge_attr, aug_ratio=0.2):

    num_nodes = x.shape[0]
    drop_num = int(num_nodes  * aug_ratio)
    nodes = np.arange(num_nodes)

    idx_perm = np.random.permutation(num_nodes)

    nodes = nodes[idx_perm]
    nodes_dropped = nodes[:drop_num]
    nodes = nodes[drop_num:]

    aug_x = torch.clone(x)
    aug_x[nodes_dropped] = 0

    return aug_x, edge_index, edge_types, edge_attr


def drop_edges(x, edge_index, edge_types, edge_attr, aug_ratio=0.2):

    num_edges = len(edge_index[0])
    drop_num = int(num_edges  * aug_ratio)

    idx_perm = np.random.permutation(num_edges)
    edge_keep = idx_perm[drop_num:]

    aug_edge_index = edge_index[:, edge_keep]

    aug_edge_types = edge_types[edge_keep]
    aug_edge_attr = edge_attr[edge_keep]

    return x, aug_edge_index, aug_edge_types, aug_edge_attr

def drop_edge_types(x, edge_index, node_types, edge_types, edge_attr, metapath, aug_ratio=0.2, inverse=False):

    targeted_edge_index = np.array([i for i, et in enumerate(edge_types) if et.item() in metapath[1::2]])

    num_edges = len(targeted_edge_index)
    drop_num = int(num_edges  * aug_ratio)

    idx_perm = np.random.permutation(num_edges)
    edges_dropped = targeted_edge_index[idx_perm[:drop_num]]

    edges_kept = np.array([i for i in range(edge_index.shape[1]) if i not in edges_dropped])
    aug_edge_index = edge_index[:, edges_kept]
    aug_edge_types = edge_types[edges_kept]
    aug_edge_attr = edge_attr[edges_kept]

    return x, aug_edge_index, aug_edge_types, aug_edge_attr

def drop_nodes(x, edge_index, edge_types, edge_attr, aug_ratio=0.2):

    num_nodes = x.shape[0]
    # print(num_nodes)
    drop_num = int(num_nodes  * aug_ratio)
    nodes = np.arange(num_nodes)
    idx_perm = np.random.permutation(num_nodes)

    nodes = nodes[idx_perm]
    nodes_dropped = nodes[:drop_num]
    nodes_kept = nodes[drop_num:]

    aug_x = torch.clone(x)
    aug_x[nodes_dropped] = 0

    edges_kept = [i for i, et in enumerate(zip(edge_index[0].numpy(), edge_index[1].numpy())) if et[0] not in nodes_dropped if et[1] not in nodes_dropped]
    edges_kept = np.array(edges_kept, dtype=int)

    aug_edge_index = edge_index[:, edges_kept]
    aug_edge_types = edge_types[edges_kept]
    aug_edge_attr = edge_attr[edges_kept]

    return aug_x, aug_edge_index, aug_edge_types, aug_edge_attr

def drop_node_types(x, edge_index, node_types, edge_types, edge_attr, metapath, aug_ratio=0.2, inverse=False):

    node_types = np.array(node_types)

    if inverse:
        targeted_nodes = np.array([idx for idx, node_type in enumerate(node_types) if node_type not in metapath[::2]])
    else:
        targeted_nodes = np.array([idx for idx, node_type in enumerate(node_types) if node_type in metapath[::2]])


    num_nodes = len(targeted_nodes)
    drop_num = int(num_nodes  * aug_ratio)

    idx_perm = np.random.permutation(num_nodes)

    targeted_nodes = targeted_nodes[idx_perm]
    nodes_dropped = targeted_nodes[:drop_num]
    nodes_kept = targeted_nodes[drop_num:]

    aug_x = torch.clone(x)
    aug_x[nodes_dropped] = 0

    edges_kept = [i for i, et in enumerate(zip(edge_index[0].numpy(), edge_index[1].numpy())) if et[0] not in nodes_dropped if et[1] not in nodes_dropped]
    edges_kept = np.array(edges_kept, dtype=int)

    aug_edge_index = edge_index[:, edges_kept]
    aug_edge_types = edge_types[edges_kept]
    aug_edge_attr = edge_attr[edges_kept]

    return aug_x, aug_edge_index, aug_edge_types, aug_edge_attr

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

def subgraph_metapath(x, edge_index, node_types, edge_types, edge_attr, metapath, aug_ratio=0.2, inverse=False, drop=True):

    adj = sp.csr_matrix((np.ones(edge_index.shape[1]), edge_index.numpy()))
    edge_types_sparse = sp.csr_matrix((edge_types, edge_index.numpy()), shape=adj.shape)
    edge_attr_sparse = sp.csr_matrix((edge_attr, edge_index.numpy()), shape=adj.shape)

    aug_edge_index = sp.csr_matrix(adj.shape)
    aug_edge_types = sp.csr_matrix(adj.shape)
    aug_edge_attr = sp.csr_matrix(adj.shape)

    # aug_edge_index = []
    # aug_edge_types = []
    # aug_edge_attr = []
    for i in range(0, len(metapath)-1, 2):

        source_type = metapath[i]
        edge_type = metapath[i+1]
        target_type = metapath[i+2]

        if inverse:
            targets = np.where(node_types != target_type)[0]
        else:
            targets = np.where(node_types == target_type)[0]

        for t in targets:

            neighbors = adj.getrow(t).indices
            if inverse:
                neighbors = np.array([n for n in neighbors if node_types[n] != source_type if edge_types_sparse[t, n] != edge_type])
            else:
                neighbors = np.array([n for n in neighbors if node_types[n] == source_type if edge_types_sparse[t, n] == edge_type])


            if len(neighbors) > 0:
                aug_edge_index[t, neighbors] = 1
                aug_edge_types[t, neighbors] = edge_types_sparse[t, neighbors]
                aug_edge_attr[t, neighbors] = edge_attr_sparse[t, neighbors]

    aug_edge_index = aug_edge_index.tocoo()
    row = torch.LongTensor(aug_edge_index.row)
    col = torch.LongTensor(aug_edge_index.col)
    aug_edge_index = torch.vstack((row, col))

    aug_edge_types = torch.tensor(aug_edge_types.data).type_as(edge_types)
    aug_edge_attr = torch.tensor(aug_edge_attr.data).type_as(edge_attr)

    # aug_edge_index = torch.vstack((aug_edge_index.row, aug_edge_index.col), dtype=torch.LongTensor)
    # aug_edge_type = aug_edge_type.data
    # aug_edge_attr = aug_edge_attr.data

    seen = set(torch.flatten(aug_edge_index).tolist())
    drop_node_list = [idx for idx in range(len(x)) if idx in seen]

    aug_x = torch.clone(x)
    aug_x[drop_node_list] = 0

    if drop:
        aug_x, aug_edge_index, aug_edge_types, aug_edge_attr = drop_nodes(aug_x, aug_edge_index, aug_edge_types, aug_edge_attr, aug_ratio)

    return aug_x, aug_edge_index, aug_edge_types, aug_edge_attr

def subgraph_metapath_list(x, adj, node_types, metapath_list, aug_ratio=0.2, inverse=False):

    num_metapaths = len(metapath_list)
    rand_metapath = np.random.randint(num_metapaths)
    metapath = metapath_list[rand_metapath]
    return subgraph_metapath(x, adj, node_types, metapath, aug_ratio=aug_ratio, inverse=inverse)



if __name__ == "__main__":
    main()
