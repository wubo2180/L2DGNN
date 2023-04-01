import argparse
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
            future_data = data[0].to(device)
            history_data = data[1].to(device)
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
            future_data = data[0].to(device)
            history_data = data[1].to(device)
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
def train(train_data_loader,model,config,scaler,optimizer):
    model.train()
    metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape}
    prediction = []
    real_value = []
    
    for data in tqdm(train_data_loader):
        optimizer.zero_grad()
        future_data = data[0].to(device)
        history_data = data[1].to(device)

        preds = model(history_data,future_data,32,1,True)

        preds = preds[:, :, :, config["MODEL"]["FROWARD_FEATURES"]]
        labels = future_data[:, :, :, config["MODEL"]["TARGET_FEATURES"]]
        prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
        loss = metric_forward(masked_mae, [prediction_rescaled,real_value_rescaled])

        loss.backward()
        optimizer.step()

        prediction.append(prediction_rescaled.detach().cpu())        # preds = forward_return[0]
        real_value.append(real_value_rescaled.detach().cpu())        # testy = forward_return[1]

    prediction = torch.cat(prediction, dim=0)
    real_value = torch.cat(real_value, dim=0)
    # re-scale data

    metric_results = {}
    for metric_name, metric_func in metrics.items():
        metric_item = metric_forward(metric_func, [prediction, real_value])
        metric_results[metric_name] = metric_item.item()
    print("Evaluate train data" + \
                "train MAE: {:.4f}, train RMSE: {:.4f}, train MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))

def load_dataset():
    pass
def main(config):
    # 加载数据集
    # load_dataset()
    # datasets/METR-LA/data_in2016_out12.pkl datasets/METR-LA/data_in12_out12.pkl datasets/METR-LA/index_in12_out12.pkl
    # dataset = ForecastingDataset('datasets/METR-LA/data_in12_out12.pkl','datasets/METR-LA/index_in12_out12.pkl','train',2016)
    '../BasicTS/datasets/METR-LA/data_in12_out12.pkl' '../BasicTS/datasets/METR-LA/index_in12_out12.pkl'
    train_dataset = PretrainingDataset('../BasicTS/datasets/METR-LA/data_in12_out12.pkl','../BasicTS/datasets/METR-LA/index_in12_out12.pkl','train')
    val_dataset = PretrainingDataset('../BasicTS/datasets/METR-LA/data_in12_out12.pkl','../BasicTS/datasets/METR-LA/index_in12_out12.pkl','valid')
    test_dataset = PretrainingDataset('../BasicTS/datasets/METR-LA/data_in12_out12.pkl','../BasicTS/datasets/METR-LA/index_in12_out12.pkl','test')

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    # adj_mx = '/dataset/METR-LA/adj_mx.pkl'
    adj_mx, _ = load_adj("./datasets/METR-LA/adj_mx.pkl", "normlap")

    adj_mx = torch.Tensor(adj_mx[0])
    scaler = load_pkl("datasets/" + config['GENERAL']['DATASET_NAME'] + "/scaler_in{0}_out{1}.pkl".format(
                                                config['GENERAL']["DATASET_INPUT_LEN"], config['GENERAL']["DATASET_OUTPUT_LEN"]))

    print('adj_mx',adj_mx.shape)
    config['MODEL']['STGCN']['gso'] = adj_mx
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_data_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = STGCN(config['MODEL']['STGCN']['Ks'],config['MODEL']['STGCN']['Kt'],config['MODEL']['STGCN']['blocks'],
                config['MODEL']['STGCN']['T'],config['MODEL']['STGCN']['n_vertex'],config['MODEL']['STGCN']['act_func'],
                config['MODEL']['STGCN']['graph_conv_type'],config['MODEL']['STGCN']['gso'],config['MODEL']['STGCN']['bias'],
                config['MODEL']['STGCN']['droprate'])
    # model = MultiLayerPerceptron(12,12,32)
    # print(net)
    model.to(device)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['OPTIM']['LR'], weight_decay=1.0e-5,eps=1.0e-8)
    print(optimizer)
    # dd
    for epoch in range(config['TRAIN']['EPOCHS']):
        print('============ epoch {:d} ============'.format(epoch))
        train(train_data_loader,model,config,scaler,optimizer)
        val(val_data_loader,model,config,scaler)
        test(test_data_loader,model,config,scaler)
    

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
    config['device'] = device
    print(config['device'])
    print(config['OPTIM']['LR'])
    # dd
    main(config)


