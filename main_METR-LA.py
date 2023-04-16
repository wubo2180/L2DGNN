import argparse
import os
import time
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
# import argparser
from argparse import ArgumentParser
from stgcn_arch import STGCN
from step.step_data import ForecastingDataset
from step.step_data import PretrainingDataset
from torch.utils.data import Dataset, DataLoader
from basicts.utils import load_adj
from metrics.mae import masked_mae
from metrics.mape import masked_mape
from metrics.rmse import masked_rmse
from basicts.data import SCALER_REGISTRY
from basicts.utils import load_pkl
from tqdm import tqdm
from MLP_arch import MultiLayerPerceptron
from gwnet_arch import GraphWaveNet
import learn2learn as l2l
from utils import edge_index_transform
from collections import OrderedDict
from torch_geometric_temporal.nn.attention import STConv
# from torch_geometric.utils import negative_sampling,structured_negative_sampling,to_dense_adj,dense_to_sparse,to_torch_coo_tensor,to_torch_csr_tensor,to_torch_csc_tensor
# def compute_space_loss(embedding, pos_edge_index,neg_edge_index):
#     # embedding shape B L N C 
    
    
#     B, L, N, C = embedding.shape
#     embedding = embedding.transpose(1,2).reshape((B,N,-1))
#     loss = 0.0
#     for i in range(B):
#         for j in range(L):
#             pos_src_node_index = pos_edge_index[i][j][0]
#             pos_tar_node_index = pos_edge_index[i][j][1]
#             neg_src_node_index = neg_edge_index[i][j][0]
#             neg_tar_node_index = neg_edge_index[i][j][1]
#             # print(pos_src_node_index.shape)
#             # print(embedding[i][j][pos_src_node_index].shape)
#             # print(embedding[i][j][pos_src_node_index].shape)
#             # dd
#             pos_score = torch.sum(embedding[i][j][pos_src_node_index] * embedding[i][j][pos_tar_node_index], dim=1)
#             neg_score = torch.sum(embedding[i][j][neg_src_node_index] * embedding[i][j][neg_tar_node_index], dim=1)
            
#             loss += criterion_space(pos_score, torch.ones_like(pos_score)) + \
#            criterion_space(neg_score, torch.zeros_like(neg_score))
            # print(pos_score.shape)
    #         print(loss)
    #         # dd
    
    # print(loss)
    # print(B)
    # print(L)
    # print(loss/(B * L))
    # dd
    # return loss/(B * L)
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
    # print(pos_src_node_index.shape)
    # print(embedding[i][j][pos_src_node_index].shape)
    # print(embedding[i][j][pos_src_node_index].shape)
    # dd
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
            # pos_tar_node_index = pos_edge_index[i][j][1]
            # neg_src_node_index = neg_edge_index[i][j][0]
            # neg_tar_node_index = neg_edge_index[i][j][1]
            embedding_list.append(embedding[i][j][pos_src_node_index])
            real_value_list.append(real_value_rescaled[i][j][pos_src_node_index])

    embedding_list = torch.stack(embedding_list).reshape((B, L, E, C))
    real_value_list = torch.stack(real_value_list).reshape((B, L, E, C))
    loss = metric_forward (masked_mae, [embedding_list, real_value_list])
    # print(loss.item())
    # dd
    return loss
def compute_long_temporal_loss(embedding, pos_edge_index,neg_edge_index,real_value_rescaled):
    embedding_list = []
    real_value_list = []
    B,  _, E = pos_edge_index.shape
    B, L, N, C = embedding.shape
    embedding = embedding.transpose(2,1).reshape((B,N,-1))
    # print(embedding.shape)
    # dd
    loss = 0.0
    print(pos_edge_index[0])
    dd
    for i in range(B):

        pos_src_node_index = pos_edge_index[i][0]
        # print(pos_src_node_index.shape)
        # pos_tar_node_index = pos_edge_index[i][j][1]
        # neg_src_node_index = neg_edge_index[i][j][0]
        # neg_tar_node_index = neg_edge_index[i][j][1]
        embedding_list.append(embedding[i][pos_src_node_index])
        real_value_list.append(real_value_rescaled[i][pos_src_node_index])
        print(embedding[i].shape)
        print(embedding[i][pos_src_node_index].shape)
        print(embedding[i][pos_src_node_index][:][:])
        
        dd
    # print(torch.stack(embedding_list).shape)
    # dd
    embedding_list = torch.stack(embedding_list).reshape((B,E,-1)).mean(dim=2)

    real_value_list = torch.stack(real_value_list).reshape((B, E,-1)).mean(dim=2)

    loss = metric_forward (masked_mae, [embedding_list, real_value_list])
    # print(loss.item())
    # dd
    print(embedding_list.shape)
    print(real_value_list.shape)
    print(loss.item())
    dd
    return loss/B
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
def train(train_data_loader,model,config,scaler,optimizer,maml):
    model.train()
    metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}
    prediction = []
    real_value = []
    batch_size = config['TRAIN']['DATA_BATCH_SIZE']
    loss = 0.0
    for idx, data in enumerate(tqdm(train_data_loader)):
        # if idx>10:
        #     break
        learner = maml.clone()


        query_space_loss = 0.0
        query_temporal_loss = 0.0
        future_data = data[0].to(config['GENERAL']['DEVICE'])
        history_data = data[1].to(config['GENERAL']['DEVICE'])
        pos_sup_edge_index = data[2].to(config['GENERAL']['DEVICE'])
        neg_sup_edge_index = data[3].to(config['GENERAL']['DEVICE'])
        pos_que_edge_index = data[4].to(config['GENERAL']['DEVICE'])
        neg_que_edge_index = data[5].to(config['GENERAL']['DEVICE'])
        # print(pos_sup_edge_index.shape)
        # print(neg_sup_edge_index.shape)
        # print(pos_que_edge_index.shape)
        # print(neg_que_edge_index.shape)
        preds = model(history_data,future_data,32,1,True)

        preds = preds[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
        labels = future_data[:, :, :, config["MODEL"]["TARGET_FEATURES"]]
        prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
        # print(prediction_rescaled.shape)
        # dd
        # for i in range(batch_size):
            
        for i in range(config['META']['UPDATE_SAPCE_STEP']): #args.update_sapce_step
            support_space_loss = compute_space_loss(prediction_rescaled, pos_sup_edge_index, neg_sup_edge_index)
            # print(support_space_loss.item())
            learner.adapt(support_space_loss)
            query_space_loss += compute_space_loss(prediction_rescaled, pos_que_edge_index, neg_que_edge_index)
        # for i in range(config['META']['UPDATE_SAPCE_STEP']): #args.update_sapce_step
        #     support_temporal_loss = compute_temporal_loss(prediction_rescaled, pos_sup_edge_index, neg_sup_edge_index,real_value_rescaled )
        #     # print(support_space_loss.item())
        #     learner.adapt(support_temporal_loss)
        #     query_temporal_loss += compute_temporal_loss(prediction_rescaled, pos_que_edge_index, neg_que_edge_index, real_value_rescaled)
        loss +=query_space_loss.item()
        # for _ in range(adapt_steps): # adaptation_steps
        #     support_preds = learner(x_support)
        #     support_loss=lossfn(support_preds, y_support)
        #     learner.adapt(support_loss)

        # print('pos_sup_edge_index')
        # print(pos_sup_edge_index[:][:][0].shape)
        # print(neg_sup_edge_index.shape)
        # print(pos_que_edge_index.shape)
        # print(neg_que_edge_index.shape)
        # dd
        # for i in range(batch_size):
        # fast_weights = OrderedDict(model.named_parameters())
        # print
        # for i in range(config['META']['UPDATE_SAPCE_STEP']): # args.update_sapce_step compute support loss
        #     support_space_loss = compute_temporal_loss(prediction_rescaled, pos_sup_edge_index, neg_sup_edge_index,real_value_rescaled )
        #     gradients = torch.autograd.grad(support_space_loss, fast_weights.values(), allow_unused=True, create_graph=True)
        #     print(gradients)
        #     dd
        #     fast_weights = OrderedDict(
        #             (name, param - config['OPTIM']['ADAPT_LR'] * grad)
        #             for ((name, param), grad) in zip(fast_weights.items(), gradients)
        #         )
        #     query_space_loss += compute_temporal_loss(prediction_rescaled, pos_que_edge_index, neg_que_edge_index, real_value_rescaled) # compute query loss

        # query_space_loss = query_space_loss
        # loss = metric_forward(masked_mae, [prediction_rescaled,real_value_rescaled])
        # update model parameters
        # print(query_space_loss.item())
        optimizer.zero_grad()
        query_space_loss.backward()
        optimizer.step()

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
    adj_mx, _ = load_adj(config['GENERAL']['ADJ_DIR'], "normlap")

    adj_mx = torch.Tensor(adj_mx[0])
    # print(adj_mx)
    config['MODEL']['STGCN']['n_vertex'] = adj_mx.shape[0]
    scaler = load_pkl(config['GENERAL']['SCALER_DIR'])
    # print(dense_to_sparse(adj_mx))
    print('adj_mx',adj_mx.shape)
    config['MODEL']['STGCN']['gso'] = adj_mx
    # 加载数据集d
    # num_sampled_edges = config['META']['SUPPORT_SET_SIZE'] + config['META']['QUERY_SET_SIZE']
    train_dataset = PretrainingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'train',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],adj_mx)
    
    
    # train_dataset[0]
    # dd
    val_dataset = PretrainingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'valid',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],adj_mx)
    test_dataset = PretrainingDataset(config['GENERAL']['DATASET_DIR'],config['GENERAL']['DATASET_INDEX_DIR'],'test',config['META']['SUPPORT_SET_SIZE'],config['META']['QUERY_SET_SIZE'],adj_mx)

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
    # model = MultiLayerPerceptron(12,12,32)
    # model = GraphWaveNet(config['MODEL']['STGCN']['n_vertex'],in_dim=3)
    
    # print(net)
    # model.load_state_dict(torch.load(config['GENERAL']['MODEL_SAVE_PATH']+'5/STGCN.pt'))
    print(config['GENERAL']['DEVICE'])
    model.to(config['GENERAL']['DEVICE'])
    # 定义优化器
    # torch.load()
    # model = SineModel(dim=hidden_dim)
    # maml = 1
    maml = l2l.algorithms.MAML(model, lr=config['OPTIM']['ADAPT_LR'], first_order=False, allow_unused=True)
    optimizer = optim.Adam(maml.parameters(), config['OPTIM']['META_LR'])
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
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate')
    parser.add_argument('--device', default=0, type=int,
                        help='device')
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 更新配置文件中的参数
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.lr is not None:
        config['lr'] = args.lr
    # 输出更新后的配置
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    config['GENERAL']['DEVICE'] = device
    # config['device'] = device
    # print(config['device'])
    # print(config['OPTIM']['LR'])
    # dd
    main(config)


