import networkx as nx
import numpy as np
import scipy
import pickle

def load_data(filepath):

    # G00 = nx.read_adjlist(filepath + '/2/2-0-1-0-2.adjlist', create_using=nx.MultiDiGraph)
    #
    # G00 = nx.to_numpy_matrix(G00)
    # print(G00.shape)
    # # print(G00[:2])
    # print(len(np.where(G00 == 1)[0]))
    features_0 = scipy.sparse.load_npz(filepath + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(filepath + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(filepath + '/features_2.npz')

    features_0 = features_0.toarray()
    features_1 = features_1.toarray()
    features_2 = features_2.toarray()

    features = np.concatenate((features_0, features_1, features_2), axis=0)

    adj = scipy.sparse.load_npz(filepath + '/adjM.npz')
    node_types = np.load(filepath + '/node_types.npy')
    labels = np.load(filepath + '/labels.npy')
    train_val_test_idx = np.load(filepath + '/train_val_test_idx.npz')

    # print(features.shape)
    # print(adj.shape)
    # print(node_types.shape)
    # print(labels.shape)

    return features, adj, node_types, labels, train_val_test_idx

def load_metapaths(filepath):

    metapaths = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            metapaths.append(np.array(line, dtype=int))

    return np.array(metapaths)
