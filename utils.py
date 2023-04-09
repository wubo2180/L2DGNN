import math
import torch
# from torch_geometric.utils import negative_sampling,structured_negative_sampling,to_dense_adj,dense_to_sparse,to_torch_coo_tensor,to_torch_csr_tensor,to_torch_csc_tensor
import numpy as np
import random
def data_preprocessing(dataset,args):
    data_list = [] 
    for time, snapshot in enumerate(dataset):
        # print(negative_sampling(snapshot.edge_index))
        data_list.append(task_construct(snapshot,args))
    # print(len(data_list))
    return data_list,snapshot.x.shape[0],snapshot.x.shape[1]


def task_construct(data,args):
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    
    # sample support set and query set for each data/task/graph
    # num_sampled_edges = args.n_way * (args.k_spt + args.k_qry)
    num_sampled_edges = args.k_spt + args.k_qry
    perm = np.random.randint(num_edges, size=num_sampled_edges)
    pos_edges = data.edge_index[:, perm]

    # x = 1 - 1.1 * (data.edge_index.size(1) / (num_nodes * num_nodes) )
    
    # if x != 0:
    #     alpha = 1 / (1 - 1.1 * (data.edge_index.size(1) / (num_nodes * num_nodes) ))
    # else:
    #     alpha = 0
    # if alpha > 0:
    neg_edges = negative_sampling(data.edge_index, num_nodes, num_sampled_edges)
    # else:
    #     i, _, k = structured_negative_sampling(data.edge_index)
    #     neg_edges = torch.stack((i,k), 0)
    cur_num_neg = neg_edges.shape[1]
    if cur_num_neg != num_sampled_edges:
        perm = np.random.randint(cur_num_neg, size=num_sampled_edges)
        neg_edges = neg_edges[:, perm]

    data.pos_sup_edge_index = pos_edges[:, :args.k_spt]
    data.neg_sup_edge_index = neg_edges[:, :args.k_spt]
    data.pos_que_edge_index = pos_edges[:, args.k_qry:]
    data.neg_que_edge_index = neg_edges[:, args.k_qry:]
    
    num_sampled_nodes = args.k_spt + args.k_qry
    perm = np.random.randint(num_nodes, size=num_sampled_nodes)
    data.temporal_sup_index = perm[:args.k_spt]
    data.temporal_que_index = perm[args.k_qry:]
    # data.aug_feature = aug_random_mask(data.x)
    return data

def aug_random_mask(input_feature, drop_percent=0.2,dim_drop=0.5):
    
    node_num, dim = input_feature.shape
    # print(dim)

    mask_num = int(node_num * drop_percent)
    mask_dim = int(dim * dim_drop)
    
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    dim_idx = [i for i in range(dim)]
    # print(mask_idx,dim_idx,mask_dim)
    # aug_feature = copy.deepcopy(input_feature)
    aug_feature = input_feature
    zeros = torch.zeros_like(aug_feature[0][0])
    for i in mask_idx:
        # for j in mask_dim:
        # mask_dim = random.sample(dim_idx, mask_dim)
        # print(mask_dim)
        aug_feature[i][random.sample(dim_idx, mask_dim)] = 0
    return aug_feature

class edge_index_transform(object):
    def __init__(self, adj_mx, num_sampled_edges) -> None:
        self.adj_mx = adj_mx
        self.num_sampled_edges = num_sampled_edges
    def __call__(self, ):
        
        print('edge_index')
        self.adj_mx[self.adj_mx>0] = 1.0
        adj_mx_tensor = torch.from_numpy(self.adj_mx).long()
        edge_index, _ = dense_to_sparse(adj_mx_tensor)
        negative_edge_index = negative_sampling(edge_index)
        num_edges = edge_index.shape[1]
        perm = np.random.randint(num_edges, size=self.num_sampled_edges)
        pos_edge_index = edge_index[:,perm]
        neg_edge_index = negative_edge_index[:,perm]
        return pos_edge_index,neg_edge_index