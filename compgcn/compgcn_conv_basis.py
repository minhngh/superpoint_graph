from .helper import *
from .message_passing import MessagePassing

class CompGCNConvBasis(MessagePassing):
	def __init__(self, in_channels, out_channels, num_bases, act=lambda x:x, params=None):
		super(self.__class__, self).__init__()

		self.p 			= params
		self.in_channels	= in_channels
		self.out_channels	= out_channels

		self.num_bases 		= num_bases
		self.act 		= act
		self.device		= None # Should be False for graph classification tasks

		# self.w_loop		= get_param((in_channels, out_channels));
		# self.w_in		= get_param((in_channels, out_channels));
		self.w_out		= get_param((in_channels, out_channels))
		self.w_rel 		= get_param((in_channels, out_channels))

		self.rel_basis 		= get_param((self.num_bases, in_channels))

		self.drop		= torch.nn.Dropout(self.p.dropout)
		# self.bn			= torch.nn.BatchNorm1d(out_channels)
		
		if self.p.bias: self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

	def forward(self, x, edge_index, rel_emb, edge_norm=None):
		if self.device is None:
			self.device = edge_index.device

		rel_embed = torch.mm(rel_emb, self.rel_basis)

		num_ent   = x.size(0)
		
		self.out_index = edge_index

		self.out_norm    = self.compute_norm(self.out_index, num_ent)
	
		out_res		= self.propagate('add', self.out_index,  x=x, rel_embed=rel_embed, edge_norm=self.out_norm,	mode='out')
		out		= self.drop(out_res)

		if self.p.bias: out = out + self.bias
		# if self.b_norm: out = self.bn(out)

		return self.act(out), torch.matmul(rel_embed, self.w_rel)

	def rel_transform(self, ent_embed, rel_embed):
		if   self.p.opn == 'corr': 	trans_embed  = ccorr(ent_embed, rel_embed)
		elif self.p.opn == 'sub': 	trans_embed  = ent_embed - rel_embed
		elif self.p.opn == 'mult': 	trans_embed  = ent_embed * rel_embed
		else: raise NotImplementedError

		return trans_embed

	def message(self, x_j, rel_embed, edge_norm, mode):
		weight 	= getattr(self, 'w_{}'.format(mode))
		xj_rel  = self.rel_transform(x_j, rel_embed)
		out	= torch.mm(xj_rel, weight)

		return out if edge_norm is None else out * edge_norm.view(-1, 1)

	def update(self, aggr_out):
		return aggr_out

	def compute_norm(self, edge_index, num_ent):
		row, col	= edge_index
		edge_weight 	= torch.ones_like(row).float()
		deg		= scatter_add( edge_weight, row, dim=0, dim_size=num_ent)	# Summing number of weights of the edges [Computing out-degree] [Should be equal to in-degree (undireted graph)]
		deg_inv		= deg.pow(-0.5)							# D^{-0.5}
		deg_inv[deg_inv	== float('inf')] = 0
		norm		= deg_inv[row] * edge_weight * deg_inv[col]			# D^{-0.5}

		return norm

	def __repr__(self):
		return '{}({}, {})'.format(
			self.__class__.__name__, self.in_channels, self.out_channels)
