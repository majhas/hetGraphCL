import torch
from torch.utils.data import Dataset
from torch_geometric.data import RandomNodeSampler

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

class DataLoader(RandomNodeSampler):

    def __collate__(self, node_idx):
        node_idx = node_idx[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)
        adj, _ = self.adj.saint_subgraph(node_idx)
        row, col, edge_idx = adj.coo()
        data.edge_index = torch.stack([row, col], dim=0)

        for key, item in self.data:
            if isinstance(item, torch.Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            elif isinstance(item, torch.Tensor) and item.size(0) == self.E:
                data[key] = item[edge_idx]
            else:
                data[key] = item

        data.node_idx = node_idx
        return data
