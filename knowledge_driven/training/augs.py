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

    def apply_aug(self, x, adj, aug_type, metapath=None):

        if metapath is not None:
            self.metapath = metapath

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
        elif aug_type == 'maskN_metapath':
            aug_x, aug_adj = mask_nodes_types(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif aug_type == 'subgraph_metapath':
            aug_x, aug_adj = subgraph_metapath(x, adj, self.node_types, self.metapath, self.aug_ratio)
        elif aug_type == 'subgraph_metapath_list':
            aug_x, aug_adj = subgraph_metapath_list(x, adj, self.node_types, self.metapath_list, self.aug_ratio)
        elif aug_type == 'dropN_not_on_metapath':
            aug_x, aug_adj = drop_node_types(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)
        elif aug_type == 'dropE_not_on_metapath':
            aug_x, aug_adj = drop_edge_types(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)
        elif aug_type == 'maskN_not_on_metapath':
            aug_x, aug_adj = mask_nodes_types(x, adj, self.node_types, self.metapath, self.aug_ratio, inverse=True)

        return aug_x, aug_adj



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

def mask_nodes_types(x, adj, node_types, metapath, aug_ratio=0.2, inverse=False):
    if inverse:
        targeted_nodes = np.array([idx for idx, node_type in enumerate(node_types) if node_type not in metapath])
    else:
        targeted_nodes = np.array([idx for idx, node_type in enumerate(node_types) if node_type in metapath])

    num_nodes = len(targeted_nodes)
    drop_num = int(num_nodes  * aug_ratio)

    idx_perm = np.random.permutation(num_nodes)

    targeted_nodes = targeted_nodes[idx_perm]
    nodes_dropped = targeted_nodes[:drop_num]

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

    _, adj_meta = subgraph_metapath(x, adj, node_types, metapath, drop=False)
    edge_idx = torch.nonzero(adj_meta)

    if inverse:
        # meta_edge_list = edge_idx.tolist()
        # edge_list = torch.nonzero(adj).tolist()
        # edge_idx = torch.tensor([edge for edge in edge_list if edge not in meta_edge_list])
        a = torch.clone(adj)
        adj_meta[adj_meta>0.] = 1.0
        a[a>0.] = 1.0
        non_meta = adj - adj_meta
        non_meta[non_meta<0.] = 0.
        edge_idx = torch.nonzero(a)

    edge_idx = edge_idx.transpose(1,0)
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
    # aug_adj[edges_dropped2, edges_dropped1] = 0
    return x, aug_adj

def drop_nodes(x, adj, aug_ratio=0.2):

    num_nodes = adj.shape[0]
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

    aug_adj = torch.zeros(adj.shape)
    num_nodes = len(adj)
    num_kept = num_nodes - int(num_nodes  * aug_ratio)

    center_node_id = random.randint(0, num_kept - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(num_kept - 1):

        n_id = sub_node_id_list[i]
        all_neighbor_list = list(set(all_neighbor_list).union(set(torch.nonzero(adj[n_id], as_tuple=False).squeeze(1).tolist())))
        # all_neighbor_list = list(set(all_neighbor_list))
        # neighbors = [n for n in all_neighbor_list if n not in sub_node_id_list]
        neighbors = list(set(all_neighbor_list) ^ set(sub_node_id_list))
        if len(neighbors) > 0:
            sample_node = random.sample(neighbors, 1)[0]
            aug_adj[n_id, sample_node] = adj[n_id, sample_node]
            sub_node_id_list.append(sample_node)
        else:
            break

    seen = set(torch.flatten(torch.nonzero(aug_adj, as_tuple=False)).tolist())
    drop_node_list = [idx for idx in range(len(adj)) if idx not in seen]

    aug_x = torch.clone(x)
    aug_x[drop_node_list] = 0

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
                aug_adj[t, neighbors] = adj[t, neighbors]

    seen = set(torch.flatten(torch.nonzero(aug_adj, as_tuple=False)).tolist())

    drop_node_list = [idx for idx in range(len(adj)) if idx not in seen]

    aug_x = torch.clone(x)
    aug_x[drop_node_list] = 0

    if drop:
        aug_x, aug_adj = drop_nodes(aug_x, aug_adj, aug_ratio)

    return aug_x, aug_adj

def subgraph_metapath_list(x, adj, node_types, metapath_list, aug_ratio=0.2):

    num_metapaths = len(metapath_list)
    rand_metapath = np.random.randint(num_metapaths)
    metapath = metapath_list[rand_metapath]
    return subgraph_metapath(x, adj, node_types, metapath, aug_ratio=aug_ratio)



if __name__ == "__main__":
    main()
