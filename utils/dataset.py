import torch
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(
        self,
        node_features,
        adj,
        labels):

        self.node_features = node_features
        self.adj = adj
        self.labels = labels

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):

        return self.node_features, self.adj, self.labels
