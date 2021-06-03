import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import scipy

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir

# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 64,
    'dropout': 0.5,
    'weight_decay': 0.001,
    'num_epochs': 100,
    'patience': 100
}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    # args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    # args['dataset'] = 'DBLP'
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

def load_acm(remove_self_loop):
    url = 'dataset/ACM3025.pkl'
    data_path = get_download_dir() + '/ACM3025.pkl'
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    labels, features = torch.from_numpy(data['label'].todense()).long(), \
                       torch.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g, subject_g]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/ACM.mat'
    download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    authors = np.unique(p_vs_a.nonzero()[1])
    subjects = np.unique(p_vs_l.nonzero()[1])

    authors_map = {a: idx for idx, a in enumerate(authors)}
    subject_map = {s: idx for idx, s in enumerate(subjects)}

    N = len(p_selected) + len(authors) + len(subjects)
    adj = np.zeros((N, N))

    pa_p, pa_a = p_vs_a.nonzero()
    for p, a in zip(pa_p, pa_a):
        idx = authors_map[a]
        a_idx = len(p_selected) + idx
        adj[p, a_idx] = 1.0
        adj[a_idx, p] = 1.0

    pl_p, pl_l = p_vs_l.nonzero()
    for p, s in zip(pl_p, pl_l):
        idx = subject_map[s]
        s_idx = len(p_selected) + len(authors) + idx
        adj[p, s_idx] = 1.0
        adj[s_idx, p] = 1.0

    node_types = [0]*len(p_selected) + [1]*len(authors) + [2]*len(subjects)
    node_types = np.array(node_types)
    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask, adj, node_types

def load_DBLP(prefix='../data/DBLP/DBLP_processed'):
    features = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    masks = [torch.LongTensor(train_val_test_idx[split]) for split in train_val_test_idx]
    train_idx, val_idx, test_idx = masks[0], masks[1], masks[2]
    _, counts = np.unique(type_mask, return_counts=True)
    a_vs_p = adjM[:counts[0], :]
    t_vs_p = adjM[np.sum(counts[:2]):np.sum(counts[:3]), :]
    v_vs_p = adjM[np.sum(counts[:3]):, :]

    hg = dgl.heterograph({
        ('author', 'ap', 'paper'): a_vs_p.nonzero(),
        ('paper', 'pa', 'author'): a_vs_p.transpose().nonzero(),
        ('term', 'tp', 'paper'): t_vs_p.nonzero(),
        ('paper', 'pt', 'term'): t_vs_p.transpose().nonzero(),
        ('venue', 'vp', 'paper'): v_vs_p.nonzero(),
        ('paper', 'pv', 'venue'): v_vs_p.transpose().nonzero()
    })

    features = torch.FloatTensor(features)

    num_classes = len(np.unique(labels))

    num_nodes = hg.number_of_nodes('author')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask, adjM, type_mask

def load_IMDB(prefix='../data/IMDB/IMDB_processed'):
    type_mask = np.load(prefix + '/node_types.npy')
    features = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()

    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz')
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    masks = [torch.LongTensor(train_val_test_idx[split]) for split in train_val_test_idx]
    train_idx, val_idx, test_idx = masks[0], masks[1], masks[2]
    _, counts = np.unique(type_mask, return_counts=True)
    d_vs_m = adjM[counts[0]:sum(counts[:2]), :]
    a_vs_m = adjM[np.sum(counts[:2]):, :]

    hg = dgl.heterograph({
        ('director', 'dm', 'movie'): d_vs_m.nonzero(),
        ('movie', 'md', 'director'): d_vs_m.transpose().nonzero(),
        ('actor', 'am', 'movie'): a_vs_m.nonzero(),
        ('movie', 'ma', 'actor'): a_vs_m.transpose().nonzero()
    })

    features = torch.FloatTensor(features)

    num_classes = len(np.unique(labels))

    num_nodes = hg.number_of_nodes('movie')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask, adjM, type_mask

def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'ACMRaw':
        return load_acm_raw(remove_self_loop)
    elif dataset == 'DBLP':
        return load_DBLP()
    elif dataset == 'IMDB':
        return load_IMDB()
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))

class EarlyStopping(object):
    def __init__(self, filename, patience=10):
        dt = datetime.datetime.now()
        # self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
        #     dt.date(), dt.hour, dt.minute, dt.second)
        self.filename = filename
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
