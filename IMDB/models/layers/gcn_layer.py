import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, act='prelu'):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, x, adj, sparse=False):

        batch = x.size(0)
        in_dim = x.size(2)
        x = x.view(-1, in_dim)
        x = self.fc(x)
        x = x.view(batch, -1, x.size(1))
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(x, 0)), 0)
        else:
            # print(adj.shape)
            # print(seq_fts.shape)
            out = torch.bmm(adj, x)
        if self.bias is not None:
            out += self.bias

        return self.act(out)
