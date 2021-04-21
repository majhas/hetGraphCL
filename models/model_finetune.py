import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelFinetune(nn.Module):

    def __init__(self, gnn, n_classes):
        super(ModelFinetune, self).__init__()

        self.gnn = gnn
        self.out_layer = nn.Linear(self.gnn.out_dim, n_classes)

    def forward(self, x, adj):

        x = self.gnn(x, adj)
        x = self.out_layer(x)
        return x

    def train_step(self, dataloader, criterion, opt, mask):

        self.train()
        for x, adj, labels in dataloader:

            x = x
            adj = adj
            # print(x.shape)
            # print(adj.shape)
            opt.zero_grad()

            out = self.forward(x, adj)
            out = out[mask]
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, labels[mask])


            loss.backward()
            opt.step()

        return loss

    def eval_step(self, dataloader, labels, criterion, mask):

        self.eval()


        for x, adj in dataloader:


            out = self.forward(x, adj)
            out = out[mask]
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, labels[mask])

        return loss, out

class HetModelFinetune(nn.Module):

    def __init__(self, gnn, n_classes):
        super(HetModelFinetune, self).__init__()

        self.gnn = gnn
        self.out_layer = nn.Linear(self.gnn.out_dim, n_classes)

    def forward(self, x, node_types, edge_index, edge_types):

        x = self.gnn(x, node_types, edge_index, edge_types)
        x = self.out_layer(x)
        return x

    def train_step(self, dataloader, labels, criterion, opt, mask):

        for x, adj in dataloader:

            x = x.squeeze(0)
            adj = adj.squeeze(0)

            dataset = dataloader.dataset
            edge_index = torch.stack(torch.where(adj > 0))

            node_types = dataset.node_types
            edge_types = torch.FloatTensor([dataset.edge_types[(node_types[n1.item()], node_types[n2.item()])] for n1, n2 in zip(edge_index[0], edge_index[1])])

            node_types = torch.tensor(node_types, device=x.device)
            edge_types = edge_types.to(adj.device)

            opt.zero_grad()

            out = self.forward(x, node_types, edge_index, edge_types)
            out = F.log_softmax(out[mask], dim=1)
            loss = criterion(out, labels[mask])


            loss.backward()
            opt.step()

        return loss

    def eval_step(self, dataloader, labels, criterion, mask):

        self.eval()

        for x, adj in dataloader:

            x = x.squeeze(0)
            adj = adj.squeeze(0)

            dataset = dataloader.dataset
            edge_index = torch.stack(torch.where(adj > 0))

            node_types = dataset.node_types
            edge_types = torch.FloatTensor([dataset.edge_types[(node_types[n1.item()], node_types[n2.item()])] for n1, n2 in zip(edge_index[0], edge_index[1])])

            node_types = torch.tensor(node_types, device=x.device)
            edge_types = edge_types.to(adj.device)

            out = self.forward(x, node_types, edge_index, edge_types)
            out = F.log_softmax(out[mask], dim=1)

            loss = criterion(out, labels[mask])

        return loss, out
