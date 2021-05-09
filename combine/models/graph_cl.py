import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphCL(nn.Module):

    def __init__(self, gnn, head_dim=300):
        super(GraphCL, self).__init__()

        self.gnn = gnn
        in_dim = self.gnn.hid_dim*self.gnn.num_heads
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, head_dim))

    def forward(self, g, x):

        for i, gnn in enumerate(self.gnn.layers):
            h = gnn(g, h, i)

        return h
        return x


class HetGraphCL(nn.Module):

    def __init__(self, gnn, head_dim=300):
        super(HetGraphCL, self).__init__()

        self.gnn = gnn
        self.projection_head = nn.Sequential(nn.Linear(self.gnn.out_dim, head_dim), nn.ReLU(inplace=True), nn.Linear(head_dim, head_dim))

    def forward(self, x, node_types, edge_index, edge_types):

        x = self.gnn(x, node_types, edge_index, edge_types)
        x = self.projection_head(x)
        return x



    def train_step(self, aug1_dataloader, aug2_dataloader, opt):

        for (x1, adj1), (x2, adj2) in zip(aug1_dataloader, aug2_dataloader):

            x1 = x1.squeeze(0)
            adj1 = adj1.squeeze(0)

            x2 = x2.squeeze(0)
            adj2 = adj2.squeeze(0)

            aug1_dataset = aug1_dataloader.dataset
            aug2_dataset = aug2_dataloader.dataset

            edge_index1 = torch.where(adj1 > 0)
            edge_index2 = torch.where(adj2 > 0)

            edge_index1 = torch.stack(edge_index1)
            edge_index2 = torch.stack(edge_index2)

            node_types = aug1_dataset.node_types
            edge_types1 = torch.FloatTensor([aug1_dataset.edge_types[(node_types[n1.item()], node_types[n2.item()])] for n1, n2 in zip(edge_index1[0], edge_index1[1])])
            edge_types2 = torch.FloatTensor([aug2_dataset.edge_types[(node_types[n1.item()], node_types[n2.item()])] for n1, n2 in zip(edge_index2[0], edge_index2[1])])

            node_types = torch.tensor(node_types, device=x1.device)
            edge_types1 = edge_types1.to(adj1.device)
            edge_types2 = edge_types2.to(adj2.device)

            opt.zero_grad()


            out1 = self.forward(x1, node_types, edge_index1, edge_types1)
            out2 = self.forward(x2, node_types, edge_index2, edge_types2)

            loss = self.loss_cl(out1, out2)


            loss.backward()
            opt.step()

        return loss
