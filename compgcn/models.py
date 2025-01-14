from .helper import *
from .compgcn_conv import CompGCNConv
from .compgcn_conv_basis import CompGCNConvBasis

import torch
class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, params=None):
		super(CompGCNBase, self).__init__(params)

		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim

		self.device		= torch.device('cuda' if params.cuda else 'cpu')
		# if self.p.num_bases > 0:
		# 	self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		# else:
		# 	if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
		# 	else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim, act=self.act, params=self.p)
			self.conv3 = CompGCNConv(self.p.embed_dim,    self.p.embed_dim, act=self.act, params=self.p)
			self.conv4 = CompGCNConv(self.p.embed_dim,    self.p.embed_dim, act=self.act, params=self.p)
			self.conv5 = CompGCNConv(self.p.embed_dim,    self.p.embed_dim, act=self.act, params=self.p)
			self.conv6 = CompGCNConv(self.p.embed_dim,    self.p.embed_dim, act=self.act, params=self.p)
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim, act=self.act, params=self.p)

		# self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, node_features, edge_features, inverse_edge_features, edge_index, inverse_edge_index, drop1, drop2):

		# r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r, ir	= self.conv1(node_features, edge_features, inverse_edge_features, edge_index, inverse_edge_index)
		x	= drop1(x)
		x, r, ir	= self.conv2(x, r, ir, edge_index, inverse_edge_index) 
		x	= drop2(x) 
		x, r, ir = self.conv3(x, r, ir, edge_index, inverse_edge_index)
		x = drop2(x)
		x, r, ir = self.conv4(x, r, ir, edge_index, inverse_edge_index)
		x = drop2(x)
		x, r, ir = self.conv5(x, r, ir, edge_index, inverse_edge_index)
		x = drop2(x)
		x, r, ir = self.conv6(x, r, ir, edge_index, inverse_edge_index)
		x = drop2(x)
		# sub_emb	= torch.index_select(x, 0, sub)
		# rel_emb	= torch.index_select(r, 0, rel)

		return x, r, ir


class CompGCN_TransE(CompGCNBase):
	def __init__(self, edge_type, params=None):
		super(self.__class__, self).__init__(params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb + rel_emb

		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score	= torch.sigmoid(x)

		return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class CompGCN_ConvE(CompGCNBase):
	def __init__(self, params=None):
		super(self.__class__, self).__init__(params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		self.fc_1 = torch.nn.Linear(self.p.embed_dim, self.p.n_classes)
		self.relu = torch.nn.ReLU(inplace = True)
		self.sigmoid = torch.nn.Sigmoid()

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)

		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	# def forward(self, sub, rel):

	# 	sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
	# 	stk_inp				= self.concat(sub_emb, rel_emb)
	# 	x				= self.bn0(stk_inp)
	# 	x				= self.m_conv1(x)
	# 	x				= self.bn1(x)
	# 	x				= F.relu(x)
	# 	x				= self.feature_drop(x)
	# 	x				= x.view(-1, self.flat_sz)
	# 	x				= self.fc(x)
	# 	x				= self.hidden_drop2(x)
	# 	x				= self.bn2(x)
	# 	x				= F.relu(x)

	# 	x = torch.mm(x, all_ent.transpose(1,0))
	# 	x += self.bias.expand_as(x)

	# 	score = torch.sigmoid(x)
	# 	return score
	def forward(self, node_features, edge_features, edge_index):

		sub_emb, rel_emb, all_ent	= self.forward_base(node_features, edge_features, edge_index, self.hidden_drop, self.feature_drop)

		stk_inp				= self.concat(sub_emb, rel_emb)

		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= self.relu(x)
		# print(x.shape, all_ent.shape)
		# x = torch.mm(x, all_ent.transpose(1,0))
		# # x += self.bias.expand_as(x)

		# score = self.sigmoid(x)
		# return score
		return self.fc_1(x)
class CompGCN_Classify(CompGCNBase):
	def __init__(self, params = None):
		super().__init__(params)
		self.fc_1 = torch.nn.Linear(params.embed_dim, params.embed_dim)
		self.relu = torch.nn.ReLU(inplace = True)
		self.fc_2 = torch.nn.Linear(params.embed_dim, params.n_classes)
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
	def forward(self, node_features, edge_features, inverse_edge_features, edge_index, inverse_edge_index):
		sub_emb, rel_emb, all_ent	= self.forward_base(node_features, edge_features, inverse_edge_features, edge_index, inverse_edge_index, self.hidden_drop, self.feature_drop)
		out = self.fc_1(sub_emb)
		out = self.feature_drop(out)
		out = self.relu(out)
		out = self.fc_2(out)
		return out