import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
import pickle
import torch

from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling

def load_data(filepath):

    if 'DBLP' in filepath:
        return load_DBLP(filepath)
    elif 'IMDB' in filepath:
        return load_IMDB(filepath)
    elif 'ACM' in filepath:
        return load_ACM(filepath)
    else:
        raise('Dataset name not in filepath')

def load_data_link(filepath):

    features, adj, node_types, node_map, _, _ = load_data(filepath)
    # no duplicates
    upper_adj = sp.triu(adj)
    # (2 x n_edges)
    edges = np.array(upper_adj.nonzero())
    neg_samples = negative_sampling(torch.as_tensor(edges), num_neg_samples=edges.shape[1])
    indices = np.arange(edges.shape[1])
    rand_seed = 1566911444
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=1566911444)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=1566911444)

    train_edges = edges[:, train_idx]
    val_edges = edges[:, val_idx]
    test_edges = edges[:, test_idx]

    train_idx = train_idx[train_idx<neg_samples.shape[1]]
    val_idx = train_idx[train_idx<neg_samples.shape[1]]
    test_idx = train_idx[train_idx<neg_samples.shape[1]]

    neg_train_edges = neg_samples[:, train_idx]
    neg_val_edges = neg_samples[:, val_idx]
    neg_test_edges = neg_samples[:, test_idx]

    adj[val_edges[0], val_edges[1]] = 0.
    adj[val_edges[1], val_edges[0]] = 0.

    adj[test_edges[0], test_edges[1]] = 0.
    adj[test_edges[1], test_edges[0]] = 0.

    train_val_test_edges = {
        'train': train_edges,
        'val': val_edges,
        'test': test_edges,
        'neg_train': neg_train_edges,
        'neg_val': neg_val_edges,
        'neg_test': neg_test_edges
    }

    return features, adj, node_types, node_map, train_val_test_edges

def load_ACM(filepath):

    node_types = np.load(filepath + '/node_types.npy')

    features_0 = scipy.sparse.load_npz(filepath + '/features.npz') # paper
    features_1 = np.eye(len(np.where(node_types==1)[0])) # author
    features_2 = np.eye(len(np.where(node_types==2)[0])) # subject

    features_0 = features_0.toarray()

    features_list = [features_0, features_1, features_2]
    n_nodes_list = [feat.shape[0] for feat in features_list]
    n_dim_list = [feat.shape[-1] for feat in features_list]

    features = np.zeros((np.sum(n_nodes_list), np.sum(n_dim_list)), dtype=float)

    start_node = 0
    start_feat = 0
    for feat in features_list:
        end_node = start_node + feat.shape[0]
        end_feat = start_feat + feat.shape[1]
        features[start_node:end_node, start_feat:end_feat] = feat
        start_node = end_node
        start_feat = end_feat

    adj = scipy.sparse.load_npz(filepath + '/adj.npz')
    labels = np.load(filepath + '/labels.npy')
    train_val_test_idx = np.load(filepath + '/train_val_test_idx.npz')

    node_map = {
        'paper': 0,
        'author': 1,
        'subject': 2,
    }

    return features, adj, node_types, node_map, labels, train_val_test_idx

def load_DBLP(filepath):

    features_0 = scipy.sparse.load_npz(filepath + '/features_0.npz') # authors
    features_1 = scipy.sparse.load_npz(filepath + '/features_1.npz') # papers
    features_2 = scipy.sparse.load_npz(filepath + '/features_2.npz') # terms
    features_3 = np.eye(20, dtype=np.float32) # venue

    features_0 = features_0.toarray()
    features_1 = features_1.toarray()
    features_2 = features_2.toarray()

    features_list = [features_0, features_1, features_2, features_3]
    n_nodes_list = [feat.shape[0] for feat in features_list]
    n_dim_list = [feat.shape[-1] for feat in features_list]

    features = np.zeros((np.sum(n_nodes_list), np.sum(n_dim_list)), dtype=float)

    start_node = 0
    start_feat = 0
    for feat in features_list:
        print(feat.shape)
        end_node = start_node + feat.shape[0]
        end_feat = start_feat + feat.shape[1]
        features[start_node:end_node, start_feat:end_feat] = feat
        start_node = end_node
        start_feat = end_feat

    adj = scipy.sparse.load_npz(filepath + '/adjM.npz')
    node_types = np.load(filepath + '/node_types.npy')
    labels = np.load(filepath + '/labels.npy')
    train_val_test_idx = np.load(filepath + '/train_val_test_idx.npz')

    node_map = {
        'author': 0,
        'paper': 1,
        'term': 2,
        'venue': 3
    }

    return features, adj, node_types, node_map, labels, train_val_test_idx

def load_IMDB(filepath):
    features_0 = scipy.sparse.load_npz(filepath + '/features_0.npz') # movie
    features_1 = scipy.sparse.load_npz(filepath + '/features_1.npz') # director
    features_2 = scipy.sparse.load_npz(filepath + '/features_2.npz') # actor

    features_0 = features_0.toarray()
    features_1 = features_1.toarray()
    features_2 = features_2.toarray()

    # features = [features_0, features_1, features_2]
    # features = [np.concatenate((features_0, features_1, features_2), axis=0)]
    n0 = features_0.shape[0]
    n1 = features_1.shape[0]
    n2 = features_2.shape[0]

    feat_0_len = features_0.shape[-1]
    feat_1_len = features_1.shape[-1]
    feat_2_len = features_2.shape[-1]

    features = np.zeros((n0+n1+n2, feat_0_len+feat_1_len+feat_2_len), dtype=float)

    start_node = 0
    start_feat = 0
    for feat in [features_0, features_1, features_2]:
        print(feat.shape)
        end_node = start_node + feat.shape[0]
        end_feat = start_feat + feat.shape[1]
        features[start_node:end_node, start_feat:end_feat] = feat
        start_node = end_node
        start_feat = end_feat


    adj = scipy.sparse.load_npz(filepath + '/adjM.npz')
    node_types = np.load(filepath + '/node_types.npy')
    labels = np.load(filepath + '/labels.npy')
    train_val_test_idx = np.load(filepath + '/train_val_test_idx.npz')

    node_map = {
        'movie': 0,
        'director': 1,
        'actor': 2,
    }

    return features, adj, node_types, node_map, labels, train_val_test_idx

def load_IMDB_link(filepath):
    features, adj, node_types, node_map, _, _ = load_IMDB(filepath)

    # no duplicates
    upper_adj = sp.triu(adj)
    # (2 x n_edges)
    edges = np.array(upper_adj.nonzero())
    neg_samples = negative_sampling(torch.as_tensor(edges), num_neg_samples=edges.shape[1])

    indices = np.arange(len(edges[0]))
    rand_seed = 1566911444
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=1566911444)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=1566911444)


    train_edges = edges[:, train_idx]
    neg_train_edges = neg_samples[:, train_idx]

    val_edges = edges[:, val_idx]
    neg_val_edges = neg_samples[:, val_idx]

    test_edges = edges[:, test_idx]
    neg_test_edges = neg_samples[:, test_idx]

    adj[val_edges[0], val_edges[1]] = 0.
    adj[val_edges[1], val_edges[0]] = 0.

    adj[test_edges[0], test_edges[1]] = 0.
    adj[test_edges[1], test_edges[0]] = 0.

    train_val_test_edges = {
        'train': train_edges,
        'val': val_edges,
        'test': test_edges,
        'neg_train': neg_train_edges,
        'neg_val': neg_val_edges,
        'neg_test': neg_test_edges
    }

    return features, adj, node_types, node_map, train_val_test_edges

def load_metapaths(filepath, node_map):

    metapaths = []
    with open(filepath, 'r') as f:
        for line in f:
            metapath = line.split(',')
            metapath = np.array([node_map[n_type.strip()] for n_type in metapath])
            metapaths.append(metapath)

    return metapaths
