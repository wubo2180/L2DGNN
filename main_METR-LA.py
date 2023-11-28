import argparse
import os
import time
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torchvision.transforms as transforms
import yaml
# import argparser
from argparse import ArgumentParser
from basicts.archs import STGCN
from basicts.archs import GraphWaveNet,Linear
from MLP_arch import MultiLayerPerceptron 
# from step.step_data import ForecastingDataset
# from step.step_data import TimeSeriesForecastingDataset
from basicts.data import TimeSeriesForecastingDataset
from torch.utils.data import Dataset, DataLoader
from basicts.utils import load_adj
# from basicts.losses import masked_mae as loss_masked_mae
from basicts.metrics import masked_mae,masked_mape,masked_rmse
from basicts.data import SCALER_REGISTRY
from basicts.utils import load_pkl
from tqdm import tqdm
from collections import OrderedDict
import learn2learn as l2l

from utils import edge_index_transform
from torch_geometric.utils import dense_to_sparse,negative_sampling,k_hop_subgraph,is_undirected,to_undirected,dropout_adj
def drop_edge(adj_mx):
    adj = adj_mx
    adj_mx[torch.abs(adj_mx)>0] = 1.0
    for i in range(adj_mx.shape[0]):
        adj_mx[i,i] = 1.0
    edge_index, _ = dense_to_sparse(adj_mx.long())
    edge_index, _ = dropout_adj(edge_index, p = 0.5)
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    row, col = edge_index
    adj[row, col] = 0.0
    return adj

def metric_forward(metric_func, args):
    """Computing metrics.

    Args:
        metric_func (function, functools.partial): metric function.
        args (list): arguments for metrics computation.
    """

    if isinstance(metric_func, functools.partial) and list(metric_func.keywords.keys()) == ["null_val"]:
        # support partial(metric_func, null_val = something)
        metric_item = metric_func(*args)
    elif callable(metric_func):
        # is a function
        metric_item = metric_func(*args, null_val=0.0)
    else:
        raise TypeError("Unknown metric type: {0}".format(type(metric_func)))
    return metric_item
def val(val_data_loader,model,config,scaler):
    model.eval()
    metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}

    prediction = []
    real_value = []
    with torch.no_grad():
        for data in tqdm(val_data_loader):
            future_data = data[0].to(config['GENERAL']['DEVICE'])
            history_data = data[1].to(config['GENERAL']['DEVICE'])
            preds = model(history_data,future_data,32,1,True)
            preds = preds[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
            labels = future_data[:, :, :, config["MODEL"]["TARGET_FEATURES"]]
            prediction.append(preds.detach().cpu())        # preds = forward_return[0]
            real_value.append(labels.detach().cpu())        # testy = forward_return[1]

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])
        # print(real_value_rescaled)
        # dd
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            metric_results[metric_name] = metric_item.item()
        print("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
def test(test_data_loader,model,config,scaler):
    """Evaluate the model.

    Args:
        train_epoch (int, optional): current epoch if in training process.
    """
    model.eval()
    # test loop
    metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}

    prediction = []
    real_value = []
    with torch.no_grad():
        for data in tqdm(test_data_loader):
            future_data = data[0].to(config['GENERAL']['DEVICE'])
            history_data = data[1].to(config['GENERAL']['DEVICE'])
            preds = model(history_data,future_data,32,1,True)
            preds = preds[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
            labels = future_data[:, :, :, config["MODEL"]["TARGET_FEATURES"]]
            prediction.append(preds.detach().cpu())        # preds = forward_return[0]
            real_value.append(labels.detach().cpu())        # testy = forward_return[1]s

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
        real_value = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])

        for i in range(config['TEST']['EVALUATION_HORIZONS']):
            # For horizon i, only calculate the metrics **at that time** slice here.
            pred = prediction[:, i, :, :]
            real = real_value[:, i, :, :]

            metric_results = {}
            for metric_name, metric_func in metrics.items():
                metric_item = metric_forward(metric_func, [pred, real])
                metric_results[metric_name] = metric_item.item()

            print("Evaluate best model on test data for horizon " + \
                "{:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}".format(i+1,metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction, real_value])
            metric_results[metric_name] = metric_item.item()
        print("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
def train(train_data_loader,model,config,scaler,optimizer,maml):
    model.train()
    metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}
    prediction = []
    real_value = []
    device = config['GENERAL']['DEVICE']

    num_nodes = config['GENERAL']['NUM_NODE']
    loss = 0.0
    for idx, data in enumerate(tqdm(train_data_loader)):

        
        # batch_size = data[0].shape[0]
        meta_train_loss = 0.0
        future_data = data[0].to(device)
        history_data = data[1].to(device)
        k_hop_index = data[4]
        batch_size = future_data.shape[0]
        
        # B L N C
        labels = future_data[:, :, :, config["MODEL"]["TARGET_FEATURES"]]
        
        real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
        # print(real_value_rescaled.shape)
        # anchor_nodes = random.sample(range(num_nodes), 10) # task random.sample(range(num_nodes), 10)
        for i in range(batch_size): # task per step
        # for i in range(num_nodes):
            learner = maml.clone()
            for j in range(config['META']['UPDATE_SAPCE_STEP']): #args.update_sapce_step
                preds = learner(history_data[i],future_data[i],batch_size,1,True)
                preds = preds[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
                prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
                support_loss = metric_forward(masked_mae, [prediction_rescaled[:,:,k_hop_index,:], real_value_rescaled[:,:,k_hop_index,:]])
                learner.adapt(support_loss)
            query_loss = metric_forward (masked_mae, [prediction_rescaled[:,:,:,:], real_value_rescaled[:,:,:,:]])
            meta_train_loss += query_loss
        # meta_train_loss /=num_nodes
        optimizer.zero_grad()
        meta_train_loss.backward()
        optimizer.step()
        loss += meta_train_loss.item()

        prediction.append(prediction_rescaled.detach().cpu())        # preds = forward_return[0]
        real_value.append(real_value_rescaled.detach().cpu())        # testy = forward_return[1]

    # prediction 
    prediction = torch.cat(prediction, dim=0) 
    real_value = torch.cat(real_value, dim=0)
    # re-scale data
    # compute train metrics
    metric_results = {}
    for metric_name, metric_func in metrics.items():
        metric_item = metric_forward(metric_func, [prediction, real_value])
        metric_results[metric_name] = metric_item.item()
    print("Evaluate train data" + \
                "train MAE: {:.4f}, train RMSE: {:.4f}, train MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
    print(loss/(idx+1))
    
def main(config):
    # adj_orig, _ = load_adj(config['GENERAL']['ADJ_DIR'], "normlap")
    adj_mx, _ = load_adj(config['GENERAL']['ADJ_DIR'], "normlap")
    
    adj_mx = torch.Tensor(adj_mx[0])
    adj_mx = drop_edge(adj_mx)
    config['GENERAL']['NUM_NODE'] = adj_mx.shape[0]
    # print(adj_mx)
    config['MODEL']['STGCN']['n_vertex'] = adj_mx.shape[0]
    scaler = load_pkl(config['GENERAL']['SCALER_DIR'])
    # print(dense_to_sparse(adj_mx))
    print('adj_mx',adj_mx.shape)
    config['MODEL']['STGCN']['gso'] = adj_mx.to(config['GENERAL']['DEVICE'])
    # 加载数据集d
    # num_sampled_edges = config['META']['SUPPORT_SET_SIZE'] + config['META']['QUERY_SET_SIZE']
    train_dataset = TimeSeriesForecastingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'train',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],1,adj_mx,config['GENERAL']['DEVICE'])
    val_dataset = TimeSeriesForecastingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'valid',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],1,adj_mx,config['GENERAL']['DEVICE'])
    test_dataset = TimeSeriesForecastingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'test',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],1,adj_mx,config['GENERAL']['DEVICE'])
    # meta_train_dataset = l2l.data.MetaDataset(train_dataset)
    # print(meta_train_dataset)
    # from learn2learn.data import TaskDataset
    # dataset = l2l.data.MetaDataset(train_dataset)
    # transforms = [
    # l2l.data.transforms.NWays(dataset, n=5),
    # l2l.data.transforms.KShots(dataset, k=1),
    # l2l.data.transforms.LoadData(dataset),
    #             ]
    # taskset = TaskDataset(dataset, transforms, num_tasks=20000)
    # for task in taskset:
    #     X, y = task
    # dd
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))

    train_data_loader = DataLoader(train_dataset, batch_size=config['TRAIN']['DATA_BATCH_SIZE'], shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config['VAL']['DATA_BATCH_SIZE'], shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=config['TEST']['DATA_BATCH_SIZE'], shuffle=False)

    model = STGCN(config['MODEL']['STGCN']['Ks'],config['MODEL']['STGCN']['Kt'],config['MODEL']['STGCN']['blocks'],
                config['MODEL']['STGCN']['T'],config['MODEL']['STGCN']['n_vertex'],config['MODEL']['STGCN']['act_func'],
                config['MODEL']['STGCN']['graph_conv_type'],config['MODEL']['STGCN']['gso'],config['MODEL']['STGCN']['bias'],
                config['MODEL']['STGCN']['droprate'])
    print(model)

    print(config['GENERAL']['DEVICE'])
    model = model.to(config['GENERAL']['DEVICE'])


    maml = l2l.algorithms.MAML(model, lr=config['OPTIM']['ADAPT_LR'], first_order=False, allow_unused=True)
    optimizer = optim.Adam(model.parameters(), config['OPTIM']['META_LR'], weight_decay=1.0e-5,eps=1.0e-8)
    # optimizer = optim.Adam(model.parameters(), lr=config['OPTIM']['LR'], weight_decay=1.0e-5,eps=1.0e-8)
    print(optimizer)
    # dd
    for epoch in range(config['TRAIN']['EPOCHS']):
        print('============ epoch {:d} ============'.format(epoch))
        train(train_data_loader,model,config,scaler,optimizer,maml)
        val(val_data_loader,model,config,scaler)
        test(test_data_loader,model,config,scaler)
        # path = config['GENERAL']['MODEL_SAVE_PATH']+str(epoch)
        # if not os.path.exists(path):
        #     os.mkdir(path)
        #     file = os.path.join(path,config['GENERAL']['MODEL_NAME']+'.pt')
        #     print(file)
        #     torch.save(model.state_dict(), file)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default='./parameter/METR-LA.yaml', type=str,
                        help='Path to the YAML config file')
    parser.add_argument('--device', default=0, type=int,
                        help='device')
    args = parser.parse_args()
    # 读取配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # 更新配置文件中的参数
    if args.device is not None:
        config['device'] = device

    # 输出更新后的配置
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    config['GENERAL']['DEVICE'] = device
    main(config)


