import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

class Augmentor():
    def __init__(self, aug_ratio=0.2, node_types=None, metapath=None, metapath_list=None):
        self.aug_ratio = aug_ratio
        self.node_types = node_types
        self.metapath = metapath
        self.metapath_list = metapath_list

    def apply_aug(self, x, adj, aug_type):

        if aug_type == None:
            aug_x, aug_adj = x, adj
        elif aug_type == 'dropN':
            aug_x, aug_adj = drop_nodes(x, adj, self.aug_ratio)
        elif aug_type == 'dropE':
            aug_x, aug_adj = drop_edges(x, adj, self.aug_ratio)
        elif aug_type == 'maskN':
            aug_x, aug_adj = mask_nodes(x, adj, self.aug_ratio)
        elif aug_type == 'subgraph':
            aug_x, aug_adj = subgraph(x, adj, self.aug_ratio)
        elif aug_type == 'dropN_metapath':
            aug_x, aug_adj = drop_node_types(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif aug_type == 'dropE_metapath':
            aug_x, aug_adj = drop_edge_types(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif aug_type == 'subgraph_metapath':
            aug_x, aug_adj = subgraph_metapath(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif aug_type == 'subgraph_metapath_list':
            aug_x, aug_adj = subgraph_metapath_list(x, adj, self.node_types, self.metapath_list, self.aug_ratio)
        elif aug_type == 'dropN_not_on_metapath':
            aug_x, aug_adj = drop_node_types(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)
        elif aug_type == 'dropE_not_on_metapath':
            aug_x, aug_adj = drop_edge_types(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)
        elif aug_type == 'subgraph_not_on_metapath':
            aug_x, aug_adj = subgraph_metapath(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)
        elif aug_type == 'subgraph_not_on_metapath_list':
            aug_x, aug_adj = subgraph_metapath_list(x, adj, self.node_types, self.metapath_list, self.aug_ratio, inverse=True)


        return aug_x, aug_adj

def remove_nodes(x, dropped):

    aug_x = []
    start = 0
    for node_feats in x:
        end = len(node_feats) + start
        drop_partition = np.array([node for node in dropped if node > start if node < end])
        drop_partition -= start
        node_feats[drop_partition] = 0
        aug_x.append(node_feats)
        start = end

    return aug_x

def mask_nodes(x, adj, aug_ratio=0.2):

    num_nodes = adj.shape[0]
    drop_num = int(num_nodes  * aug_ratio)
    nodes = np.arange(num_nodes)

    idx_perm = np.random.permutation(num_nodes)

    nodes = nodes[idx_perm]
    nodes_dropped = nodes[:drop_num]
    nodes = nodes[drop_num:]
    aug_x = remove_nodes(x, nodes_dropped)

    return aug_x, adj


def drop_edges(x, adj, aug_ratio=0.2):

    adj = adj.coalesce()
    edge_idx = adj.indices()
    values = adj.values()
    shape = adj.shape

    num_edges = len(edge_idx[0])
    drop_num = int(num_edges  * aug_ratio)
    idx_perm = np.random.permutation(num_edges)

    shuffled_edges = edge_idx[:, idx_perm]
    shuffled_values = values[idx_perm]

    edges_dropped = shuffled_edges[:, :drop_num]
    # aug_values = aug_values[drop_num:]
    # dropped_edges = aug_edge_idx[:, :drop_num].transpose(1, 0)
    # dropped_edges_flipped = torch.tensor([[edge[1], edge[0]] for edge in dropped_edges.numpy()])
    # dropped_edges = torch.cat((dropped_edges, dropped_edges_flipped), dim=0)
    # test = dropped_edges[0]

    aug_adj = adj.to_dense()
    aug_adj[edges_dropped[0], edges_dropped[1]] = 0.
    aug_adj = aug_adj.to_sparse()

    # edges_kept = np.array([i for i, edge in enumerate(edge_idx.transpose(1, 0)) if edge not in dropped_edges])
    # print(edges_kept.shape)
    # aug_edge_idx = edge_idx[:, edges_kept]
    # aug_values = values[edges_kept]

    # aug_adj = torch.sparse_coo_tensor(aug_edge_idx, aug_values, shape)

    return x, aug_adj

def drop_edge_types(x, adj, node_types, metapath, aug_ratio=0.2, inverse=False):

    _, adj_meta = subgraph_metapath(x, adj, node_types, metapath, inverse=inverse, drop=False)

    # adj_meta = torch.triu(adj_meta.to_dense()).to_sparse().coalesce()
    adj_meta = adj_meta.coalesce()
    edge_idx = adj_meta.indices()
    values = adj_meta.values()
    shape = adj.shape

    num_edges = len(edge_idx[0])
    drop_num = int(num_edges  * aug_ratio)

    idx_perm = np.random.permutation(num_edges)

    shuffled_edges = edge_idx[:, idx_perm]
    shuffled_values = values[idx_perm]

    edges_dropped = shuffled_edges[:, :drop_num]

    aug_adj = adj.to_dense()
    aug_adj[edges_dropped[0], edges_dropped[1]] = 0.
    # aug_adj[edges_dropped[1], edges_dropped[0]] = 0.
    aug_adj = aug_adj.to_sparse()
    # aug_edge_idx = edge_idx[:, idx_perm]
    # aug_values = values[idx_perm]
    #
    # dropped_edges = aug_edge_idx[:, :drop_num].transpose(1, 0)
    # dropped_edges_flipped = torch.tensor([[edge[1], edge[0]] for edge in dropped_edges.numpy()])
    # dropped_edges = torch.cat((dropped_edges, dropped_edges_flipped), dim=0)
    #
    # adj = adj.coalesce()
    # edge_idx = adj.indices()
    # values = adj.values()
    #
    # edges_kept = np.array([i for i, edge in enumerate(edge_idx.transpose(1, 0)) if edge not in dropped_edges])
    #
    # aug_edge_idx = edge_idx[:, edges_kept]
    # aug_values = values[edges_kept]
    #
    # aug_adj = torch.sparse_coo_tensor(aug_edge_idx, aug_values, shape)

    return x, aug_adj

def drop_nodes(x, adj, aug_ratio=0.2):

    adj = adj.coalesce()
    edge_idx = adj.indices()
    values = adj.values()
    shape = adj.shape

    num_nodes = adj.shape[0]
    # print(num_nodes)
    drop_num = int(num_nodes  * aug_ratio)
    nodes = np.arange(num_nodes)
    idx_perm = np.random.permutation(num_nodes)

    nodes = nodes[idx_perm]
    nodes_dropped = nodes[:drop_num]
    nodes = nodes[drop_num:]

    aug_x = remove_nodes(x, nodes_dropped)

    edges_kept = [i for i, et in enumerate(zip(edge_idx[0].numpy(), edge_idx[1].numpy())) if et[0] not in nodes_dropped if et[1] not in nodes_dropped]

    aug_edge_idx = edge_idx[:, edges_kept]
    aug_values = values[edges_kept]

    aug_adj = torch.sparse_coo_tensor(aug_edge_idx, aug_values, shape)

    return aug_x, aug_adj


def drop_node_types(x, adj, node_types, metapath, aug_ratio=0.2, inverse=False):


    if inverse:
        targeted_nodes = np.array([idx for idx, node_type in enumerate(node_types) if node_type not in metapath])
    else:
        targeted_nodes = np.array([idx for idx, node_type in enumerate(node_types) if node_type in metapath])

    adj = adj.coalesce()
    edge_idx = adj.indices()
    values = adj.values()
    shape = adj.shape

    num_nodes = len(targeted_nodes)
    drop_num = int(num_nodes  * aug_ratio)

    idx_perm = np.random.permutation(num_nodes)

    targeted_nodes = targeted_nodes[idx_perm]
    nodes_dropped = targeted_nodes[:drop_num]
    targeted_nodes = targeted_nodes[drop_num:]

    aug_x = remove_nodes(x, nodes_dropped)

    edges_kept = [i for i, et in enumerate(zip(edge_idx[0].numpy(), edge_idx[1].numpy())) if et[0] not in nodes_dropped if et[1] not in nodes_dropped]
    aug_edge_idx = edge_idx[:, edges_kept]
    aug_values = values[edges_kept]

    aug_adj = torch.sparse_coo_tensor(aug_edge_idx, aug_values, shape)

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

    # 4311
    # [  510  4311 18481 18482 18502 18973 19312 19623 19772 26108]
    adj = adj.coalesce()
    edge_idx = adj.indices().numpy()

    aug_edge_idx = []
    aug_values = []
    for i in range(len(metapath)-1):
        source_type = metapath[i]
        target_type = metapath[i+1]

        if inverse:
            targets = np.where(node_types != target_type)[0]
        else:
            targets = np.where(node_types == target_type)[0]

        for t in targets:

            neighbors = adj[t].coalesce().indices()[0].numpy()
            values = adj[t].coalesce().values().numpy()

            if inverse:
                keep_idx = np.array([i for i, n in enumerate(neighbors) if node_types[n] != source_type])

            else:
                # neighbors = np.array([n for n in neighbors if node_types[n] == source_type])
                keep_idx = np.array([i for i, n in enumerate(neighbors) if node_types[n] == source_type])

            if len(keep_idx) > 0:
                neighbors = neighbors[keep_idx]
                values = values[keep_idx]
                # print(f'target: {t}')
                # print(f'neighbors: {neighbors}')
                # print(f'values: {values}')
                # aug_adj[t, neighbors] = values
                # t_expanded = np.broadcast_to(t, neighbors.shape)
                [aug_edge_idx.append([t, n]) for n in neighbors]
                [aug_values.append(v) for v in values]
                # aug_adj[t, neighbors] = values


    aug_edge_idx = torch.LongTensor(aug_edge_idx).transpose(1, 0)
    values = torch.FloatTensor(values)
    aug_adj = torch.sparse_coo_tensor(aug_edge_idx, aug_values, adj.shape)

    seen = set(torch.flatten(aug_edge_idx).tolist())
    drop_node_list = [idx for idx in range(len(x)) if idx in seen]

    aug_x = remove_nodes(x, drop_node_list)

    if drop:
        aug_x, aug_adj = drop_nodes(aug_x, aug_adj, aug_ratio)

    return aug_x, aug_adj

# def subgraph_metapath(x, adj, node_types, metapath, aug_ratio=0.2, inverse=False, drop=True):
#
#     # adj = adj.coalesce()
#     aug_edge_idx = []
#     aug_values = []
#
#     for i in range(len(metapath)-1):
#
#         source_type = metapath[i]
#         target_type = metapath[i+1]
#
#         if inverse:
#             targets = np.where(node_types != target_type)[0]
#         else:
#             targets = np.where(node_types == target_type)[0]
#
#         for t in targets:
#
#             neighbors = adj[t].coalesce().indices()[0].numpy()
#             neighbors = list(set(neighbors))
#
#             if inverse:
#                 neighbors = np.array([n for n in neighbors if node_types[n] != source_type])
#             else:
#                 neighbors = np.array([n for n in neighbors if node_types[n] == source_type])
#
#             if len(neighbors) > 0:
#                 for n in neighbors:
#                     aug_edge_idx.append([t, n])
#                     aug_edge_idx.append([n, t])
#                     aug_values.append(adj[t, n])
#                     aug_values.append(adj[n, t])
#
#     aug_edge_idx = torch.tensor(aug_edge_idx).transpose()
#     aug_values = torch.tensor(aug_values)
#     aug_adj = torch.sparse_coo_tensor(aug_edge_idx, aug_values, adj.shape)
#
#     seen = set(torch.flatten(edge_idx).tolist())
#     drop_node_list = [idx for idx in range(len(adj)) if idx in seen]
#
#     aug_x = remove_nodes(x, drop_node_list)
#
#     if drop:
#         aug_x, aug_adj = drop_nodes(aug_x, aug_adj, aug_ratio)
#
#     return aug_x, aug_adj

def subgraph_metapath_list(x, adj, node_types, metapath_list, aug_ratio=0.2, inverse=False):

    # print(metapath_list)
    num_metapaths = len(metapath_list)
    rand_metapath = np.random.randint(num_metapaths)
    metapath = metapath_list[rand_metapath]
    return subgraph_metapath(x, adj, node_types, metapath, aug_ratio=aug_ratio, inverse=inverse)



if __name__ == "__main__":
    main()
