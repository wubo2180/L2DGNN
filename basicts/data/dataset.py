import os

import torch
from torch.utils.data import Dataset

from ..utils import load_pkl
from torch_geometric.utils import dense_to_sparse,negative_sampling
import random
import numpy as np
import math
class TimeSeriesForecastingDataset(Dataset):
    """Time series forecasting dataset."""

    def __init__(self, data_file_path: str, index_file_path: str, mode: str,support_set_size, query_set_size, adj_mx) -> None:
        super().__init__()
        assert mode in ["train", "valid", "test"], "error mode"
        self._check_if_file_exists(data_file_path, index_file_path)
        # read raw data (normalized)
        data = load_pkl(data_file_path)
        processed_data = data["processed_data"]
        self.data = torch.from_numpy(processed_data).float()
        # read index
        self.index = load_pkl(index_file_path)[mode]
        self.num_sampled_edges = support_set_size + query_set_size
        self.support_set_size = support_set_size
        self.query_set_size = query_set_size
        self.adj_mx = adj_mx
        # print(self.index[-1])
        # dd
        self.pos_sup_edge_index, self.neg_sup_edge_index, self.pos_que_edge_index, self.neg_que_edge_index = self.create_edge_index(self.index[-1][-1])
        # print('###')
        # print(self.index)
        # dd

    def _check_if_file_exists(self, data_file_path: str, index_file_path: str):
        """Check if data file and index file exist.

        Args:
            data_file_path (str): data file path
            index_file_path (str): index file path

        Raises:
            FileNotFoundError: no data file
            FileNotFoundError: no index file
        """

        if not os.path.isfile(data_file_path):
            raise FileNotFoundError("BasicTS can not find data file {0}".format(data_file_path))
        if not os.path.isfile(index_file_path):
            raise FileNotFoundError("BasicTS can not find index file {0}".format(index_file_path))
        

    def __getitem__(self, index: int) -> tuple:
        """Get a sample.

        Args:
            index (int): the iteration index (not the self.index)

        Returns:
            tuple: (future_data, history_data), where the shape of each is L x N x C.
        """

        idx = list(self.index[index])
        if isinstance(idx[0], int):
            # continuous index
            history_data = self.data[idx[0]:idx[1]]
            future_data = self.data[idx[1]:idx[2]]

        else:
            # discontinuous index or custom index
            # NOTE: current time $t$ should not included in the index[0]
            history_index = idx[0]    # list
            assert idx[1] not in history_index, "current time t should not included in the idx[0]"
            history_index.append(idx[1])
            history_data = self.data[history_index]
            future_data = self.data[idx[1], idx[2]]
        # 
        # print(idx)
        pos_sup_edge_index = self.pos_sup_edge_index[idx[0]:idx[1]]
        neg_sup_edge_index = self.neg_sup_edge_index[idx[0]:idx[1]]
        pos_que_edge_index = self.pos_que_edge_index[idx[0]:idx[1]]
        neg_que_edge_index = self.neg_que_edge_index[idx[0]:idx[1]]
        # print(pos_sup_edge_index.shape)
        # print(neg_sup_edge_index.shape)
        # print(pos_que_edge_index.shape)
        # print(neg_que_edge_index.shape)
        
        # print(history_data.shape)
        # return future_data, history_data, pos_edge_index, neg_edge_index
        return future_data, history_data, pos_sup_edge_index, neg_sup_edge_index, pos_que_edge_index, neg_que_edge_index

    def __len__(self):
        """Dataset length

        Returns:
            int: dataset length
        """
        return len(self.index)
    
    def create_edge_index(self, length):
        self.adj_mx[torch.abs(self.adj_mx)>0] = 1.0
        edge_index, _ = dense_to_sparse(self.adj_mx.long())
        negative_edge_index = negative_sampling(edge_index)
        num_edges = edge_index.shape[1]
        
        pos_sup_edge_index = []
        neg_sup_edge_index = []
        pos_que_edge_index = []
        neg_que_edge_index = []
        for i in range(length):
            perm = np.random.randint(num_edges, size=self.num_sampled_edges)
            pos_edge_index = edge_index[:,perm]
            neg_edge_index = negative_edge_index[:,perm]

            pos_sup_edge_index.append(pos_edge_index[:, :self.support_set_size])
            neg_sup_edge_index.append(neg_edge_index[:, :self.support_set_size])
            pos_que_edge_index.append(pos_edge_index[:, self.support_set_size:])
            neg_que_edge_index.append(neg_edge_index[:, self.support_set_size:])

        pos_sup_edge_index = torch.stack(pos_sup_edge_index, dim=0)
        neg_sup_edge_index = torch.stack(neg_sup_edge_index, dim=0)
        pos_que_edge_index = torch.stack(pos_que_edge_index, dim=0)
        neg_que_edge_index = torch.stack(neg_que_edge_index, dim=0)
        

        return pos_sup_edge_index, neg_sup_edge_index, pos_que_edge_index, neg_que_edge_index
        # return pos_edge_index,neg_edge_index