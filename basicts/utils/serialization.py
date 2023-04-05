import pickle

import torch
import numpy as np
import numpy.ma as ma
from .adjacent_matrix_norm import calculate_scaled_laplacian, calculate_symmetric_normalized_laplacian, calculate_symmetric_message_passing_adj, calculate_transition_matrix


def load_pkl(pickle_file: str) -> object:
    """Load pickle data.

    Args:
        pickle_file (str): file path

    Returns:
        object: loaded objected
    """

    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise

    return pickle_data


def dump_pkl(obj: object, file_path: str):
    """Dumplicate pickle data.

    Args:
        obj (object): object
        file_path (str): file path
    """

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def load_adj(file_path: str, adj_type: str):
    """load adjacency matrix.

    Args:
        file_path (str): file path
        adj_type (str): adjacency matrix type

    Returns:
        list of numpy.matrix: list of preproceesed adjacency matrices
        np.ndarray: raw adjacency matrix
    """

    try:
        # METR and PEMS_BAY
        a, b, adj_mx = load_pkl(file_path)
        # print(adj_mx)
        
        
#         np.savetxt('D:\\python_workspace\\L2DGNN\\1.txt', np.c_[adj_mx],
#  fmt='%d',delimiter='\t')

        # with open('D:\\python_workspace\\L2DGNN\\1.txt', 'w') as f:
        #     for i in range (len (adj_mx)): 
        #         f.write(str(adj_mx[i])+'\n')

        # dd
    except ValueError:
        # PEMS04
        adj_mx = load_pkl(file_path)
    # print(adj_mx.shape)
    # print(adj_mx)
    # print(np.sum(adj_mx>0))
    # adj_mx_flatten = adj_mx.flatten()
    # nums = adj_mx.shape[0]
    
    # mask = np.random.choice([0, 1], size=adj_mx_flatten.shape[0], p=[.2, .8])

    # # mask = np.random.randint(0,2,adj_mx_flatten.shape[0])
    # # print(mask)
    # # dd
    # adj_mx = (adj_mx_flatten*mask).reshape(nums,nums)
    # print(np.sum(adj_mx>0))
    # n = adj_mx.shape[0]
    # adj_mx[range(n), range(n)] = 0
    # print(adj_mx.shape)
    # print(adj_mx)
    # print(np.sum(adj_mx>0))
    # # dd
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [calculate_symmetric_message_passing_adj(
            adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == "original":
        adj = [adj_mx]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx


def load_node2vec_emb(file_path: str) -> torch.Tensor:
    """load node2vec embedding

    Args:
        file_path (str): file path

    Returns:
        torch.Tensor: node2vec embedding
    """

    # spatial embedding
    with open(file_path, mode="r") as f:
        lines = f.readlines()
        temp = lines[0].split(" ")
        num_vertex, dims = int(temp[0]), int(temp[1])
        spatial_embeddings = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(" ")
            index = int(temp[0])
            spatial_embeddings[index] = torch.Tensor([float(ch) for ch in temp[1:]])
    return spatial_embeddings
