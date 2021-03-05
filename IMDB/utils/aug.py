import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

def main():
    pass


def aug_random_mask(input_feature, drop_percent=0.2):

    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):

    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))


    edge_num = int(len(row_idx) / 2)      # 9228 / 2
    add_drop_num = int(edge_num * percent / 2)
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)


    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0

    '''
    above finish drop edges
    '''
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:

        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1

    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    # input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = np.array(sorted(random.sample(all_node_list, drop_num)))

    # aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    # aug_input_adj = delete_row_col(input_adj, drop_node_list)

    # aug_input_fea[drop_node_list, :] = 0
    aug_input_adj = torch.clone(input_adj)
    aug_input_adj[drop_node_list, :] = 0
    aug_input_adj[:, drop_node_list] = 0

    # aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return input_fea, aug_input_adj


def aug_subgraph(input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    node_num = input_adj.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):

        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()

        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break


    drop_node_list = np.array(sorted([i for i in all_node_list if not i in sub_node_id_list]))

    # aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    # aug_input_adj = delete_row_col(input_adj, drop_node_list)
    aug_input_adj = input_adj.deepcopy()
    aug_input_adj[drop_node_list, :] = 0
    aug_input_adj[:, drop_node_list] = 0

    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_adj


# def metapath_subgraph(input_adj, node_types, metapath, undirected=True):
#
#     input_adj = torch.tensor(input_adj.todense().tolist())
#     # input_feat = input_feat.squeeze(0)
#     node_num = input_adj.shape[0]
#
#     sub_node_id_list = [i for i in range(node_num) if node_types[i] == metapath[0]]
#     metapath_queue = [0 for _ in sub_node_id_list]
#     walk_queue = [n for n in sub_node_id_list]
#
#     aug_input_adj = np.zeros(input_adj.shape)
#     for _ in range(node_num - 1):
#         # print(f'Metapath queue: {metapath_queue}')
#         # print(f'Walk Queue: {walk_queue}')
#         # print(f'Subgraph Nodes: {sub_node_id_list}')
#
#         metapath_idx = metapath_queue.pop(0) + 1
#         curr_node = walk_queue.pop(0)
#
#         if metapath_idx == len(metapath):
#             break
#
#         # print(f'Current Metapath: {metapath[metapath_idx]}')
#         # print(f'Current Node: {curr_node}')
#
#         neighbors = torch.nonzero(input_adj[curr_node], as_tuple=False).squeeze(1).tolist()
#         neighbors = list(set(neighbors))
#
#
#         # neighbors = [n for n in neighbors if n not in sub_node_id_list if node_types[n] == metapath[metapath_idx] if n not in walk_queue]
#         neighbors = [n for n in neighbors if node_types[n] == metapath[metapath_idx]]
#
#         for n in neighbors:
#             aug_input_adj[curr_node, n] = 1
#             if undirected:
#                 aug_input_adj[n, curr_node] = 1
#
#         neighbors = [n for n in neighbors if n not in sub_node_id_list if n not in walk_queue]
#         # print(f'Neighbors: {neighbors}\n')
#         walk_queue += neighbors
#         metapath_queue += [metapath_idx for _ in neighbors]
#
#         if len(walk_queue) != 0:
#             if curr_node not in sub_node_id_list:
#                 sub_node_id_list.append(curr_node)
#         else:
#             break
#
#     return aug_input_adj

def aug_metapath_subgraph(input_adj, node_types, metapath):

    input_adj = torch.tensor(input_adj.todense().tolist())
    node_num = input_adj.shape[0]

    aug_input_adj = np.zeros(input_adj.shape)

    for i in range(len(metapath)-1):

        source_type = metapath[i]
        target_type = metapath[i+1]

        targets = np.where(node_types == target_type)[0]

        for t in targets:

            neighbors = torch.nonzero(input_adj[t], as_tuple=False).squeeze(1).tolist()
            neighbors = list(set(neighbors))
            neighbors = np.array([n for n in neighbors if node_types[n] == source_type])

            aug_input_adj[t, neighbors] = 1

    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_adj

def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out



















if __name__ == "__main__":
    main()
