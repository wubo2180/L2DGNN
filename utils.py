import math
import torch

def select_input_features(data: torch.Tensor,config) -> torch.Tensor:
    """Select input features.

    Args:
        data (torch.Tensor): input history data, shape [B, L, N, C]

    Returns:
        torch.Tensor: reshaped data
    """

    # select feature using self.forward_features

    data = data[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
    return data

def select_target_features( data: torch.Tensor,config) -> torch.Tensor:
    """Select target feature.

    Args:
        data (torch.Tensor): prediction of the model with arbitrary shape.

    Returns:
        torch.Tensor: reshaped data with shape [B, L, N, C]
    """

    # select feature using self.target_features
    data = data[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
    return data

def build_train_dataset(self, cfg: dict):
    """Build MNIST train dataset

    Args:
        cfg (dict): config

    Returns:
        train dataset (Dataset)
    """

    data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
    index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TRAIN"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

    # build dataset args
    dataset_args = cfg.get("DATASET_ARGS", {})
    # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
    dataset_args["data_file_path"] = data_file_path
    dataset_args["index_file_path"] = index_file_path
    dataset_args["mode"] = "train"

    dataset = cfg["DATASET_CLS"](**dataset_args)
    print("train len: {0}".format(len(dataset)))

    batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
    self.iter_per_epoch = math.ceil(len(dataset) / batch_size)

    return dataset

@staticmethod
def build_val_dataset(cfg: dict):
    """Build MNIST val dataset

    Args:
        cfg (dict): config

    Returns:
        validation dataset (Dataset)
    """
    data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
    index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["VAL"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

    # build dataset args
    dataset_args = cfg.get("DATASET_ARGS", {})
    # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
    dataset_args["data_file_path"] = data_file_path
    dataset_args["index_file_path"] = index_file_path
    dataset_args["mode"] = "valid"

    dataset = cfg["DATASET_CLS"](**dataset_args)
    print("val len: {0}".format(len(dataset)))

    return dataset

@staticmethod
def build_test_dataset(cfg: dict):
    """Build MNIST val dataset

    Args:
        cfg (dict): config

    Returns:
        train dataset (Dataset)
    """

    data_file_path = "{0}/data_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])
    index_file_path = "{0}/index_in{1}_out{2}.pkl".format(cfg["TEST"]["DATA"]["DIR"], cfg["DATASET_INPUT_LEN"], cfg["DATASET_OUTPUT_LEN"])

    # build dataset args
    dataset_args = cfg.get("DATASET_ARGS", {})
    # three necessary arguments, data file path, corresponding index file path, and mode (train, valid, or test)
    dataset_args["data_file_path"] = data_file_path
    dataset_args["index_file_path"] = index_file_path
    dataset_args["mode"] = "test"

    dataset = cfg["DATASET_CLS"](**dataset_args)
    print("test len: {0}".format(len(dataset)))

    return dataset

from torch_geometric.utils import negative_sampling,structured_negative_sampling
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import torch
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
