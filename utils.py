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