import torch
import torch.nn as nn
import numpy as np
from scipy import sparse as sp

from .gated_gcn_layer import GatedGCNLayer

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = torch.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class GatedGCNNet(nn.Module):
    def __init__(self, params):
        super().__init__()

        in_dim_node = params.in_dim_node 
        in_dim_edge = params.in_dim_edge 
        hidden_dim = params.hidden_dim
        n_classes = params.n_classes
        dropout = params.dropout
        n_layers = params.L
        self.batch_norm = params.batch_norm
        self.residual = params.residual
        self.n_classes = n_classes
        self.device = torch.device('cuda' if params.cuda else 'cpu')
        self.pos_enc_dim = params.pos_enc_dim
        
        self.embedding_e = nn.Linear(in_dim_edge, in_dim_node) # edge feat is a float
        if params.pos_enc_dim:
            self.embedding_pos_enc = nn.Linear(self.pos_enc_dim, in_dim_node)
        self.layers = nn.ModuleList([ GatedGCNLayer(hidden_dim if i > 0 else in_dim_node, hidden_dim, dropout,
                                                    self.batch_norm, self.residual) for i in range(n_layers) ])
        self.MLP_layer = MLPReadout(hidden_dim, n_classes)
    
    def positional_encoding(self, g, pos_enc_dim):
        n = g.num_nodes()
        A = g.adj(scipy_fmt = 'coo')
        N = sp.diags(g.in_degrees().clip(min = 1).numpy() ** -0.5, dtype = float)
        L = sp.eye(n) - N.dot(A).dot(N)
        eig_value, eig_vector = np.linalg.eig(L.toarray())
        idx = eig_value.argsort()
        eig_value =eig_value[idx]
        eig_vector = np.real(eig_vector[:, idx])
        g.ndata['pos_enc'] = torch.Tensor(eig_vector[:, 1 : pos_enc_dim + 1])

    def forward(self, g, h):
        self.positional_encoding(g, self.pos_enc_dim)
        e = g.edata['f']
        e = self.embedding_e(e)
        if self.pos_enc_dim:
            h = h + self.embedding_pos_enc(g.ndata['pos_enc'])
        
        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        # output
        h_out = self.MLP_layer(h)

        return h_out
