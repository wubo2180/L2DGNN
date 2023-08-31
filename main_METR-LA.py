import argparse
import os
import time
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
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
def compute_space_loss(embedding, pos_edge_index,neg_edge_index):
    # embedding shape B L N C 
    criterion_space = nn.BCEWithLogitsLoss()
    
    B, L, N, C = embedding.shape
    # embedding = embedding.transpose(1,2).reshape((B,N,-1))
    embedding = embedding[:,0,:].squeeze(1)
    B, _ , E = pos_edge_index.shape
    loss = 0.0
    for i in range(B):
        pos_src_node_index = pos_edge_index[i,0,:].flatten()
        pos_tar_node_index = pos_edge_index[i,1,:].flatten()
        neg_src_node_index = neg_edge_index[i,0,:].flatten()
        neg_tar_node_index = neg_edge_index[i,1,:].flatten()

        pos_score = torch.sum(embedding[i][pos_src_node_index] * embedding[i][pos_tar_node_index], dim=1)
        neg_score = torch.sum(embedding[i][neg_src_node_index] * embedding[i][neg_tar_node_index], dim=1)
            
        loss += criterion_space(pos_score, torch.ones_like(pos_score)) + \
                criterion_space(neg_score, torch.zeros_like(neg_score))
    return loss/B
def compute_temporal_loss(embedding, pos_edge_index,neg_edge_index,real_value_rescaled):
    embedding_list = []
    real_value_list = []
    B, L, _, E = pos_edge_index.shape
    B, L, N, C = embedding.shape
    loss = 0.0
    for i in range(B):
        for j in range(L):
            pos_src_node_index = pos_edge_index[i][j][0]

            embedding_list.append(embedding[i][j][pos_src_node_index])
            real_value_list.append(real_value_rescaled[i][j][pos_src_node_index])

    embedding_list = torch.stack(embedding_list).reshape((B, L, E, C))
    real_value_list = torch.stack(real_value_list).reshape((B, L, E, C))
    loss = metric_forward (masked_mae, [embedding_list, real_value_list])
    return loss
def compute_long_temporal_loss(prediction_rescaled, real_value_rescaled,index_list):
    timespan = range(20)
    # print(prediction_rescaled.shape)
    # print(prediction_rescaled)
    B, L, C  = prediction_rescaled.shape
    prediction_rescaled = prediction_rescaled.squeeze()
    prediction_rescaled = prediction_rescaled.mean(dim = 1)
    
    real_value_rescaled  = real_value_rescaled.squeeze()
    real_value_rescaled = real_value_rescaled.mean(dim = 1)

    loss = metric_forward (masked_mae, [prediction_rescaled[index_list], real_value_rescaled[index_list]])
    return loss
def compute_edge_loss(prediction_rescaled,real_value_rescaled, pos_edge_index,neg_edge_index,):
    N, L, C  = prediction_rescaled.shape
    # N,  E = pos_edge_index.shape

    loss = metric_forward (masked_mae, [prediction_rescaled[pos_edge_index], real_value_rescaled[pos_edge_index]])
    return loss

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

    # batch_size = config['TRAIN']['DATA_BATCH_SIZE']
    num_nodes = config['GENERAL']['NUM_NODE']
    loss = 0.0
    for idx, data in enumerate(tqdm(train_data_loader)):

        

        query_loss = 0.0
        
        future_data = data[0].to(device)
        history_data = data[1].to(device)

        
        values,indices  = torch.sort(data[7])
        
        k_hop_index = data[8]
        
        batch_size = future_data.shape[0]


        
        origin_weights = OrderedDict(model.named_parameters())
        # print(origin_weights)
        for i in range(num_nodes): # num_nodes = tasks
            # model_clone = copy.deepcopy(model)
            # model_clone_optimizer = optim.Adam(model_clone.parameters(), config['OPTIM']['ADAPT_LR'], weight_decay=1.0e-5,eps=1.0e-8)
            # model.update_parameter(origin_weights)
            learner = maml.clone()
            for j in range(config['META']['UPDATE_SAPCE_STEP']): #args.update_sapce_step
                preds = learner(history_data,future_data,32,1,True)

                preds = preds[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
                labels = future_data[:, :, :, config["MODEL"]["TARGET_FEATURES"]]
                prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
                real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
                
                support_loss = metric_forward (masked_mae, [prediction_rescaled[:,:,k_hop_index[i],:], real_value_rescaled[:,:,k_hop_index[i],:]])
                # model_clone_optimizer.zero_grad()
                # support_loss.backward(retain_graph= True)
                # model_clone_optimizer.step()
                # gradients = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)

                # fast_weights = OrderedDict(
                #     (name, param - config['OPTIM']['ADAPT_LR'] * grad)
                #     for ((name, param), grad) in zip(origin_weights.items(), gradients)
                # )
                # model.update_parameter(fast_weights)
                learner.adapt(support_loss)
            preds = learner(history_data,future_data,32,1,True)
            preds = preds[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
            labels = future_data[:, :, :, config["MODEL"]["TARGET_FEATURES"]]
            prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
                
            query_loss += metric_forward (masked_mae, [prediction_rescaled[:,:,i,:], real_value_rescaled[:,:,i,:]])
        query_loss = query_loss/num_nodes
        # model.update_parameter(origin_weights)
        optimizer.zero_grad()
        query_loss.backward()
        optimizer.step()
        loss += query_loss.item()

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
    train_dataset = TimeSeriesForecastingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'train',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],adj_mx,config['GENERAL']['DEVICE'])
    val_dataset = TimeSeriesForecastingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'valid',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],adj_mx,config['GENERAL']['DEVICE'])
    test_dataset = TimeSeriesForecastingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'test',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],adj_mx,config['GENERAL']['DEVICE'])

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    # print(train_dataset[0][2].shape)
    # dd
    train_data_loader = DataLoader(train_dataset, batch_size=config['TRAIN']['DATA_BATCH_SIZE'], shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config['VAL']['DATA_BATCH_SIZE'], shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=config['TEST']['DATA_BATCH_SIZE'], shuffle=False)

    model = STGCN(config['MODEL']['STGCN']['Ks'],config['MODEL']['STGCN']['Kt'],config['MODEL']['STGCN']['blocks'],
                config['MODEL']['STGCN']['T'],config['MODEL']['STGCN']['n_vertex'],config['MODEL']['STGCN']['act_func'],
                config['MODEL']['STGCN']['graph_conv_type'],config['MODEL']['STGCN']['gso'],config['MODEL']['STGCN']['bias'],
                config['MODEL']['STGCN']['droprate'])
    print(model)
    # print('ff')
    # dd
    # model = nn.Linear(10,10)
    # print(model.W.data)

    # dd
    # model = MultiLayerPerceptron(12,12,32)
    
    # model = GraphWaveNet(207,0.3,[torch.tensor(i) for i in adj_mx],True,True,None,3)
    # model = MultiLayerPerceptron(12,12,32)
    
    # print(net)
    # model.load_state_dict(torch.load(config['GENERAL']['MODEL_SAVE_PATH']+'5/STGCN.pt'))
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
    ###
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


