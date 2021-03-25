import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=0):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)


        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):

        x = self.fc(x)
        
        return x
