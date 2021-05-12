import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GCNConv

class GCNNet(nn.Module):
    def __init__(self,
                in_channels,
                hidden_channels,
                n_layers,
                n_classes):
        super().__init__()
        modules = []
        for i in range(n_layers - 1):
            modules.append((GCNConv(in_channels if i == 0 else hidden_channels, hidden_channels), 'x, edge_index -> x'))
            modules.append(nn.ReLU(inplace = True))
        modules.append((GCNConv(hidden_channels, n_classes), 'x, edge_index -> x'))
        self.net = Sequential('x, edge_index', modules)
    def forward(self, x, edge_index):
        return self.net(x, edge_index)