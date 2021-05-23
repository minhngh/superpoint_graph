"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import math
import transforms3d
import torch
import ecc
import h5py
from sklearn.preprocessing import StandardScaler
import igraph



def spg_node_features(node_att):
    feats = np.hstack((node_att['xyz'], node_att['nlength'], node_att['volume'], node_att['surface']))
    return feats    

def scaler01(trainlist, testlist, transform_train=True, validlist = []):
    """ Scale edge features to 0 mean 1 stddev """
    node_feats = np.concatenate([ trainlist[i][2] for i in range(len(trainlist)) ], 0)
    scaler = StandardScaler().fit(node_feats)

    if transform_train:
        for i in range(len(trainlist)):
            scaler.transform(trainlist[i][2], copy=False)
    for i in range(len(testlist)):
        scaler.transform(testlist[i][2], copy=False)
    if len(validlist)>0:
        for i in range(len(validlist)):
            scaler.transform(validlist[i][2], copy=False)
    return trainlist, testlist, validlist, scaler

def spg_reader(args, fname, incl_dir_in_name=False):
    """ Loads a supergraph from H5 file. """
    f = h5py.File(fname,'r')

    if f['sp_labels'].size > 0:
        node_gt_size = f['sp_labels'][:].astype(np.int64) # column 0: no of unlabeled points, column 1+: no of labeled points per class
        node_gt = np.argmax(node_gt_size[:,1:], 1)[:,None]
        node_gt[node_gt_size[:,1:].sum(1)==0,:] = -100    # superpoints without labels are to be ignored in loss computation
    else:
        N = f['sp_point_count'].shape[0]
        node_gt_size = np.concatenate([f['sp_point_count'][:].astype(np.int64), np.zeros((N,8), dtype=np.int64)], 1)
        node_gt = np.zeros((N,1), dtype=np.int64)

    node_att = {}
    node_att['xyz'] = f['sp_centroids'][:]
    node_att['nlength'] = np.maximum(0, f['sp_length'][:])
    node_att['volume'] = np.maximum(0, f['sp_volume'][:] ** 2)
    node_att['surface'] = np.maximum(0, f['sp_surface'][:] ** 2)
    # node_att['size'] = f['sp_point_count'][:]
    edges = np.concatenate([ f['source'][:], f['target'][:] ], axis=1).astype(np.int64)

    node_feats = spg_node_features(node_att)

    name = os.path.basename(fname)[:-len('.h5')]
    if incl_dir_in_name: name = os.path.basename(os.path.dirname(fname)) + '/' + name

    return node_gt, node_gt_size, node_feats, edges, name


def spg_to_igraph(node_gt, node_gt_size, node_feats, edges, fname):
    """ Builds representation of superpoint graph as igraph. """
    targets = np.concatenate([node_gt, node_gt_size], axis=1)
    G = igraph.Graph(n = node_gt.shape[0], edges = edges.tolist(), directed = True,
                     vertex_attrs = {'v': list(range(node_gt.shape[0])),
                                   'f': node_feats,
                                   't': targets, 
                                   's':node_gt_size.sum(1)})
    return G, fname

def random_neighborhoods(G, num, order):
    """ Samples `num` random neighborhoods of size `order`.
        Graph nodes are then treated as set, i.e. after hardcutoff, neighborhoods may be broken (sort of data augmentation). """
    centers = random.sample(range(G.vcount()), k=num)
    neighb = G.neighborhood(centers, order)
    subset = [item for sublist in neighb for item in sublist]
    subset = sorted(set(subset))
    return G.subgraph(subset)

def k_big_enough(G, minpts, k):
    """ Returns a induced graph on maximum k superpoints of size >= minpts (smaller ones are not counted) """
    valid = np.array(G.vs['s']) >= minpts
    n = np.argwhere(np.cumsum(valid)<=k)[-1][0]+1
    return G.subgraph(range(n))


def loader(entry, train, args, db_path, test_seed_offset=0):
    """ Prepares a superpoint graph (potentially subsampled in training) and associated superpoints. """
    G, fname = entry
    # 1) subset (neighborhood) selection of (permuted) superpoint graph
    if train:
        if 0 < args.spg_augm_hardcutoff < G.vcount():
            perm = list(range(G.vcount())); random.shuffle(perm)
            G = G.permute_vertices(perm)

        if 0 < args.spg_augm_nneigh < G.vcount():
            G = random_neighborhoods(G, args.spg_augm_nneigh, args.spg_augm_order)

        if 0 < args.spg_augm_hardcutoff < G.vcount():
            G = k_big_enough(G, args.ptn_minpts, args.spg_augm_hardcutoff)

    # Only stores graph with edges
    if len(G.get_edgelist()) != 0:
        # 2) loading clouds for chosen superpoint graph nodes
        clouds_meta, clouds_flag = [], [] # meta: textual id of the superpoint; flag: 0/-1 if no cloud because too small
        clouds, clouds_global = [], [] # clouds: point cloud arrays; clouds_global: diameters before scaling

        for s in range(G.vcount()):
            cloud, diam = load_superpoint(args, db_path + '/parsed/' + fname + '.h5', G.vs[s]['v'], train, test_seed_offset)
            if cloud is not None:
                clouds_meta.append('{}.{:d}'.format(fname,G.vs[s]['v'])); clouds_flag.append(0)
                clouds.append(cloud.T)
                clouds_global.append(diam)
            else:
                clouds_meta.append('{}.{:d}'.format(fname,G.vs[s]['v'])); clouds_flag.append(-1)

        clouds_flag = np.array(clouds_flag)
        if len(clouds) != 0:
            clouds = np.stack(clouds)
        if len(clouds_global) != 0:
            clouds_global = np.concatenate(clouds_global)

        # return np.array(G.vs['t']), np.array(G.vs['f']), clouds_meta, clouds_flag, clouds, clouds_global
        # cloud_means = np.mean(clouds, axis = -1)
        # cloud_stds  = np.std(clouds, axis = -1)
        return np.array(G.vs['t']), np.array(G.vs['f']), G.get_edgelist(), clouds_meta, clouds_flag, clouds, clouds_global

    # Don't use the graph if it doesn't have edges.
    else:
        target, G, edgelist, clouds_meta, clouds_flag, clouds, clouds_global = None, None, None, None, None, None, None
        return target, G, edgelist, clouds_meta, clouds_flag, clouds, clouds_global


def cloud_edge_feats(edgeattrs):
    edgefeats = np.asarray(edgeattrs['f'])
    return torch.from_numpy(edgefeats), None

def eccpc_collate(batch):
    """ Collates a list of dataset samples into a single batch (adapted in ecc.graph_info_collate_classification())
    """
    targets, node_feats, edgelist, clouds_meta, clouds_flag, clouds, clouds_global = list(zip(*batch))

    targets = torch.cat([torch.from_numpy(t) for t in targets if t is not None], 0).long()
    feats = torch.cat([torch.from_numpy(f) for f in node_feats], dim = 0)

    if len(clouds_meta[0]) > 0:
        clouds = torch.cat([torch.from_numpy(f) for f in clouds if f is not None], 0)
        clouds_global = torch.cat([torch.from_numpy(f) for f in clouds_global if f is not None], 0)
        clouds_flag = torch.cat([torch.from_numpy(f) for f in clouds_flag if f is not None], 0)
        clouds_meta = [item for sublist in clouds_meta if sublist is not None for item in sublist]

    # edgelist = torch.cat([torch.Tensor(t) for t in edgelist], 0).long().T
    _edgelist = []
    acc_node_size = 0
    for i in range(len(edgelist)):
        edges = torch.Tensor(edgelist[i])
        edges += acc_node_size
        acc_node_size += node_feats[i].shape[0]
        _edgelist.append(edges)
    edgelist = torch.cat(_edgelist, dim = 0).long().T
        

    # cloud_feats = torch.cat((feats, cloud_means, cloud_stds, clouds_global.unsqueeze(1)), dim = 1)
    return targets, edgelist, feats, (clouds_meta, clouds_flag, clouds, clouds_global)


############### POINT CLOUD PROCESSING ##########

def load_superpoint(args, fname, id, train, test_seed_offset):
    """ """
    hf = h5py.File(fname,'r')
    P = hf['{:d}'.format(id)]
    N = P.shape[0]
    # if N < args.ptn_minpts: # skip if too few pts (this must be consistent at train and test time)
    #     return None, N
    P = P[:].astype(np.float32)

    rs = np.random.random.__self__ if train else np.random.RandomState(seed=id+test_seed_offset) # fix seed for test

    if N > args.ptn_npts: # need to subsample
        ii = rs.choice(N, args.ptn_npts)
        P = P[ii, ...]
    elif N < args.ptn_npts: # need to pad by duplication
        ii = rs.choice(N, args.ptn_npts - N)
        P = np.concatenate([P, P[ii,...]], 0)

    if args.pc_xyznormalize:
        # normalize xyz into unit ball, i.e. in [-0.5,0.5]
        diameter = np.max(np.max(P[:,:3],axis=0) - np.min(P[:,:3],axis=0))
        P[:,:3] = (P[:,:3] - np.mean(P[:,:3], axis=0, keepdims=True)) / (diameter + 1e-10)
    else:
        diameter = 0.0
        P[:,:3] = (P[:,:3] - np.mean(P[:,:3], axis=0, keepdims=True))

    if args.pc_attribs != '':
        columns = []
        if 'xyz' in args.pc_attribs: columns.append(P[:,:3])
        if 'rgb' in args.pc_attribs: columns.append(P[:,3:6])
        if 'e' in args.pc_attribs: columns.append(P[:,6,None])
        if 'lpsv' in args.pc_attribs: columns.append(P[:,7:11])
        if 'XYZ' in args.pc_attribs: columns.append(P[:,11:14])
        if 'd' in args.pc_attribs: columns.append(P[:,14])
        P = np.concatenate(columns, axis=1)

    if train:
        P = augment_cloud(P, args)
    return P, np.array([diameter], dtype=np.float32)


def augment_cloud(P, args):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if args.pc_augm_scale > 1:
        s = random.uniform(1/args.pc_augm_scale, args.pc_augm_scale)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if args.pc_augm_rot==1:
        angle = random.uniform(0, 2*math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0,0,1], angle), M) # z=upright assumption
    if args.pc_augm_mirror_prob > 0: # mirroring x&y, not z
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,1,0]), M)
    P[:,:3] = np.dot(P[:,:3], M.T)

    if args.pc_augm_jitter:
        sigma, clip= 0.01, 0.05 # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1*clip, clip).astype(np.float32)
    return P
    
def global_rotation(P, args):
    print("e")
