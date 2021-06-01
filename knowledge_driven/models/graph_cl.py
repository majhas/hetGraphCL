import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphCL(nn.Module):

    def __init__(self, gnn, head_dim=300):
        super(GraphCL, self).__init__()

        self.gnn = gnn
        self.projection_head = nn.Sequential(
            nn.Linear(self.gnn.out_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Linear(head_dim, head_dim),
            # nn.ReLU(inplace=True),
            # nn.Linear(head_dim, head_dim)
            )

    def forward(self, x, adj):

        x = self.gnn(x, adj)
        x = self.projection_head(x)
        return x

    def embed(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)

        return x

    def train_step(self, g1, g2, criterion, opt):

        self.train()
        # for (x1, adj1), (x2, adj2) in dataloader:
        (x1, adj1), (x2, adj2) = g1, g2
        opt.zero_grad()

        out1 = self.forward(x1, adj1)
        out2 = self.forward(x2, adj2)

        loss = self.loss_cl(out1, out2)

        loss.backward()
        opt.step()

        return loss
