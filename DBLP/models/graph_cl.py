import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_diag_sum(inp):
    inp = torch.diagonal(inp)
    return torch.sum(inp)

def con_loss(x1, x2, temp=0.1):

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix /= temp
    # print(sim_matrix.shape)
    row_softmax_matrix = -F.log_softmax(sim_matrix, dim=1)

    colomn_softmax_matrix = -F.log_softmax(sim_matrix, dim=0)

    row_diag_sum = compute_diag_sum(row_softmax_matrix)
    colomn_diag_sum = compute_diag_sum(colomn_softmax_matrix)
    contrastive_loss = (row_diag_sum + colomn_diag_sum) / (2 * len(row_softmax_matrix))

    return contrastive_loss

class GraphCL(nn.Module):

    def __init__(self, gnn, head_dim=300):
        super(GraphCL, self).__init__()

        self.gnn = gnn
        self.projection_head = nn.Sequential(nn.Linear(self.gnn.out_dim, head_dim), nn.ReLU(inplace=True), nn.Linear(head_dim, head_dim))

    def forward(self, x, adj):

        x = self.gnn(x, adj)
        x = self.projection_head(x)
        return x

    def embed(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)

        return x

    def loss_cl(self, x1, x2, temp=0.1):
        # T = 0.1
        # batch_size = x1.size(0)
        # x1_abs = x1.norm(dim=1)
        # x2_abs = x2.norm(dim=1)
        #
        # # print(x1_abs.shape)
        # # print(x2_abs.shape)
        # # print('Sim matrix')
        # sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        # # print(sim_matrix.shape)
        # sim_matrix = torch.exp(sim_matrix / T)
        # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # loss = - torch.log(loss).mean()
        loss = con_loss(x1, x2, temp)
        return loss

    def train_step(self, aug1_dataloader, aug2_dataloader, opt):

        self.train()
        for (x1, adj1), (x2, adj2) in zip(aug1_dataloader, aug2_dataloader):

            opt.zero_grad()

            x1 = [feat.squeeze(0) for feat in x1]
            x2 = [feat.squeeze(0) for feat in x2]

            adj1 = adj1[0]
            adj2 = adj2[0]

            out1 = self.forward(x1, adj1)
            out2 = self.forward(x2, adj2)

            loss = self.loss_cl(out1, out2)

            loss.backward()
            opt.step()

        return loss

class HetGraphCL(nn.Module):

    def __init__(self, gnn, head_dim=300):
        super(HetGraphCL, self).__init__()

        self.gnn = gnn
        self.projection_head = nn.Sequential(nn.Linear(self.gnn.out_dim, head_dim), nn.ReLU(inplace=True), nn.Linear(head_dim, head_dim))

    def forward(self, x, node_types, edge_index, edge_types):

        x = self.gnn(x, node_types, edge_index, edge_types)
        x = self.projection_head(x)
        return x


    def loss_cl(self, x1, x2, T=0.1):

        # batch_size = x1.size(0)
        # x1_abs = x1.norm(dim=1)
        # x2_abs = x2.norm(dim=1)
        #
        # # print(x1_abs.shape)
        # # print(x2_abs.shape)
        # # print('Sim matrix')
        # sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        # # print(sim_matrix.shape)
        # sim_matrix = torch.exp(sim_matrix / T)
        # pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        # loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        # loss = - torch.log(loss).mean()
        loss = con_loss(x1, x2)
        return loss

    def train_step(self, aug1_dataloader, aug2_dataloader, opt):

        for (x1, adj1), (x2, adj2) in zip(aug1_dataloader, aug2_dataloader):

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
