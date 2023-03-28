import argparse
import time
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
# 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)


def load_dataset():
    pass
def main(config):
    # 加载数据集
    # load_dataset()
    # dataset = ForecastingDataset('datasets/METR-LA/data_in12_out12.pkl','datasets/METR-LA/index_in12_out12.pkl','train',2016)
    dataset = PretrainingDataset('datasets/METR-LA/data_in2016_out12.pkl','datasets/METR-LA/index_in2016_out12.pkl','train')
    
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    dd
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    net = STGCN()
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['OPTIM']['LR'], momentum=config['OPTIM']['MOMENTUM'])
    # 训练网络
    for epoch in range(config['TRAIN']['EPOCHS']):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # 测试网络
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

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
    # device = torch.device("cpu")
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    config['device'] = device
    print(config['device'])
    print(config['OPTIM']['LR'])
    # dd
    main(config)
    
# with open(yaml_path,'r') as file:
#     args = argparse.Namespace(**yaml.load(file.read(),Loader=yaml.FullLoader))
# # with open(yaml_path, "r") as f:
# #     args = yaml.safe_load(f)
# print(args.MODEL["STGCN"]["Ks"])
# args.data_tt = 'ds'
# print(args.data_tt)


