import networkx as nx
import numpy as np
import scipy
import pickle

def load_data(filepath):


    features_0 = scipy.sparse.load_npz(filepath + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(filepath + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(filepath + '/features_2.npz')
    features_3 = np.eye(20, dtype=np.float32)

    features_0 = features_0.toarray()
    features_1 = features_1.toarray()
    features_2 = features_2.toarray()

    features = [features_0, features_1, features_2, features_3]

    adj = scipy.sparse.load_npz(filepath + '/adjM.npz')
    node_types = np.load(filepath + '/node_types.npy')
    labels = np.load(filepath + '/labels.npy')
    train_val_test_idx = np.load(filepath + '/train_val_test_idx.npz')



    return features, adj, node_types, labels, train_val_test_idx

def load_metapaths(filepath):

    metapaths = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            metapaths.append(np.array(line, dtype=int))

    return np.array(metapaths)
