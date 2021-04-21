import networkx as nx
import numpy as np
import scipy
import pickle


def load_data(filepath):

    if 'DBLP' in filepath:
        return load_DBLP(filepath)
    elif 'IMDB' in filepath:
        return load_IMDB(filepath)

    else:
        raise('Dataset name not in filepath')

def load_DBLP(filepath):

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

    node_map = {
        'author': 0,
        'paper': 1,
        'term': 2,
        'venue': 3
    }

    return features, adj, node_types, node_map, labels, train_val_test_idx

def load_IMDB(filepath):
    features_0 = scipy.sparse.load_npz(filepath + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(filepath + '/features_1.npz')
    features_2 = scipy.sparse.load_npz(filepath + '/features_2.npz')

    features_0 = features_0.toarray()
    features_1 = features_1.toarray()
    features_2 = features_2.toarray()

    features = [features_0, features_1, features_2]
    # features = [np.concatenate((features_0, features_1, features_2), axis=0)]

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

def load_LASTFM(filepath):
    pass

    node_map = {
        'user': 0,
        'artist': 1,
        'tag': 2
    }

    return features, adj, node_types, node_map, labels, train_val_test_idx

def load_metapaths(filepath, node_map):

    metapaths = []
    with open(filepath, 'r') as f:
        for line in f:
            metapath = line.split(',')
            metapath = np.array([node_map[n_type.strip()] for n_type in metapath])
            metapaths.append(metapath)

    return metapaths
