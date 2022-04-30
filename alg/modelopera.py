# coding=utf-8
import torch
from mymodels.CNN_1d import CNN


def get_fea(args):
    # if args.dataset == 'dg5':
    #     net = img_network.DTNBase()
    # elif args.net.startswith('res'):
    #     net = img_network.ResBase(args.net)
    # else:
    #     net = img_network.VGGBase(args.net)
    net = CNN()
    return net


def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)
            # p = network(x)

            if p.size(1) == 1:   # eval有几类/几个域，是否为1个
                correct += (p.gt(0).eq(y).float()).sum().item()   # p.gt(0): p的各元素值是否大于零
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()  # 计算预测准确的个数，argmax(1): 对行操作，每行最大概率的索引
            total += len(x)
    network.train()
    return correct / total


def accuracy_cnn(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x = x.cuda().float()
            y = y.cuda().long()
            p = network(x)

            if p.size(1) == 1:   # eval有几类/几个域，是否为1个
                correct += (p.gt(0).eq(y).float()).sum().item()   # p.gt(0): p的各元素值是否大于零
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()  # 计算预测准确的个数，argmax(1): 对行操作，每行最大概率的索引
            total += len(x)
    network.train()
    return correct / total