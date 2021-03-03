import os
import argparse
import numpy as np
import scipy.sparse as sp
import torch


from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import DataLoader
from utils.aug import metapath_subgraph

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='data/ogbl-biokg', help='path to dataset')
    # parser.add_argument('--')

    return parser.parse_args()

def main(args):

    root = os.path.dirname(args.dataset)
    name = os.path.basename(args.dataset)

    dataset = PygLinkPropPredDataset(name=name, root=root)

    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
    graph = dataset[0] # pyg graph object containing only training edges

    print(graph.num_nodes_dict)

    # train_loader = DataLoader(dataset[train_edge], batch_size=32, shuffle=True)
    print(graph.edge)
if __name__ == '__main__':
    args = parse_args()
    main(args)
