# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
from datautil.imgdata.imgdataload import DataGenerate
from datautil.mydataloader import InfiniteDataLoader
import os
from scipy.io import loadmat
from scipy import signal
import argparse
import torch
import random


def set_random_seed(seed=0):
    # seed setting
    # 初始化设置，保证每次运行结果一样
    # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)        # CPU 每次生成固定的随机数（生成的随机数是一样的），这就使得每次实验结果显示一致了
    torch.cuda.manual_seed(seed)   # GPU
    # 将这个flag置为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置Torch的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。
    torch.backends.cudnn.deterministic = True
    # True：如果网络结构不是动态变化的，网络的输入 (batch size，图像的大小，输入的通道) 是固定的，初始化时为每个卷积层选择最合适的卷积算法
    # False：不提前选择合适卷积算法，选用默认的
    torch.backends.cudnn.benchmark = False


signal_size = 1024
# train/source
datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data"]
normalname = ["97.mat", "98.mat", "99.mat", "100.mat"]
# For 12k Drive End Bearing Fault Data
cwru1 = ["97.mat", "105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat",
         "234.mat"]  # 1797rpm
cwru2 = ["98.mat", "106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat",
         "235.mat"]  # 1772rpm
cwru3 = ["99.mat", "107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat",
         "236.mat"]  # 1750rpm
cwru4 = ["100.mat", "108.mat", "121.mat", "133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat",
         "237.mat"]  # 1730rpm
# For 12k Fan End Bearing Fault Data
cwru5 = ["97.mat", "278.mat", "282.mat", "294.mat", "274.mat", "286.mat", "310.mat", "270.mat", "290.mat",
         "315.mat"]  # 1797rpm
cwru6 = ["98.mat", "279.mat", "283.mat", "295.mat", "275.mat", "287.mat", "309.mat", "271.mat", "291.mat",
         "316.mat"]  # 1772rpm
cwru7 = ["99.mat", "280.mat", "284.mat", "296.mat", "276.mat", "288.mat", "311.mat", "272.mat", "292.mat",
         "317.mat"]  # 1750rpm
cwru8 = ["100.mat", "281.mat", "285.mat", "297.mat", "277.mat", "289.mat", "312.mat", "273.mat", "293.mat",
         "318.mat"]  # 1730rpm
# For 48k Drive End Bearing Fault Data
cwru9 = ["97.mat", "109.mat", "122.mat", "135.mat", "173.mat", "189.mat", "201.mat", "213.mat", "226.mat",
         "238.mat"]  # 1797rpm
cwru10 = ["98.mat", "110.mat", "123.mat", "136.mat", "175.mat", "190.mat", "202.mat", "214.mat", "227.mat",
          "239.mat"]  # 1772rpm
cwru11 = ["99.mat", "111.mat", "124.mat", "137.mat", "176.mat", "191.mat", "203.mat", "215.mat", "28.mat",
          "240.mat"]  # 1750rpm
cwru12 = ["100.mat", "112.mat", "125.mat", "138.mat", "177.mat", "192.mat", "204.mat", "217.mat", "229.mat",
          "241.mat"]  # 1730rpm
# label
label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # The failure data is labeled 1-9
axis = ["_DE_time", "_FE_time", "_BA_time"]


class PU(object):
    def __init__(self, flags, task='[pu1, pu2]', condition='[0]', data_num=1):
        self.WC = eval(condition)
        self.frequency = 64000
        self.down_f = 16000
        self.root = os.path.join(flags.data_dir, 'PU')
        self.data = eval(task)
        self.d_num = data_num + 1


    def get_files(self):
        # data = {}
        # lab = {}
        data = []
        lab = []
        domain_num = 0

        for i in self.data:
            for state in self.WC:
                # data[str(domain_num)] = []
                # lab[str(domain_num)] = []
                for (k, v) in i.items():
                    for j in v:
                        for num in range(1, self.d_num):
                            name1 = WC[state] + "_" + j + "_" + '1'
                            path1 = os.path.join(self.root, j, name1 + ".mat")
                            data1, lab1 = self.data_load(path1, name=name1, label=int(k))
                            # data[str(domain_num)] += data1
                            # lab[str(domain_num)] += lab1
                            data += data1
                            lab += lab1

                # domain_num += 1
        return data, lab#, len(lab)

    def data_load(self, filename, name, label):
        fl = loadmat(filename)[name]
        fl = fl[0][0][2][0][6][2]  # Take out the data
        # fl = fl.reshape(-1,)
        fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=1)
        data = []
        lab = []
        start, end = 0, signal_size
        while end <= fl.shape[1]:
            x = fl[:, start:end]
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[:, range(int(x.shape[1] / 2))]
            # x = x.reshape(-1, 1)
            data.append(x)
            lab.append(label)
            start += signal_size
            end += signal_size

        return data, lab


class CWRUFew(object):
    def __init__(self, flags, task='cwru1', sub_root=0, shot=50, test=False):
        self.data = eval(task)
        self.sub_root = datasetname[sub_root]
        self.root = os.path.join(flags.data_dir, 'CWRU', self.sub_root)
        self.shot = shot
        self.test = test
        if sub_root == 1:   # 12k FE
            self.axis = axis[1]
        else:
            self.axis = axis[0]
        if sub_root == 2:   # 48k DE
            self.frequency = 48000
            self.down_f = 16000
        else:
            self.frequency = 12000
            self.down_f = 12000

    def get_files(self):
        data = []
        lab = []
        for i in range(len(self.data)):
            path2 = os.path.join(self.root, self.data[i])
            data1, lab1 = self.data_load(path2, self.data[i], label=label[i])
            data += data1
            lab += lab1
        return data, lab

    def data_load(self, filename, axisname, label):
        '''
        This function is mainly used to generate test data and training data.
        filename:Data location
        axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
        '''
        datanumber = axisname.split(".")
        if eval(datanumber[0]) < 100:
            realaxis = "X0" + datanumber[0] + axis[0]
        else:
            realaxis = "X" + datanumber[0] + axis[0]
        fl = loadmat(filename)[realaxis]
        # fl = fl.reshape(-1, )
        data = []
        lab = []
        fl_middle = int(fl.shape[0] / 2)
        if eval(datanumber[0]) <= 100:
            if self.test is False:
                start, end = fl_middle, (fl_middle + signal_size)
                while end <= fl.shape[0]:
                    x = fl[start:end]
                    x = np.fft.fft(x)
                    x = np.abs(x) / len(x)
                    x = x[range(int(x.shape[0] / 2))]
                    x = x.reshape(1, -1)
                    data.append(x)
                    lab.append(label)
                    start += signal_size
                    end += signal_size
            else:
                start, end = 0, signal_size
                while end <= fl_middle:
                    x = fl[start:end]
                    x = np.fft.fft(x)
                    x = np.abs(x) / len(x)
                    x = x[range(int(x.shape[0] / 2))]
                    x = x.reshape(1, -1)
                    data.append(x)
                    lab.append(label)
                    start += signal_size
                    end += signal_size
        else:
            start, end = 0, signal_size
            while end <= fl.shape[0]:
                x = fl[start:end]
                x = np.fft.fft(x)
                x = np.abs(x) / len(x)
                x = x[range(int(x.shape[0] / 2))]
                x = x.reshape(1, -1)
                data.append(x)
                lab.append(label)
                start += signal_size
                end += signal_size

        return data, lab


class MFPT(object):
    def __init__(self, flags):
        # self.WC = WC
        self.frequency = 48000   # 48828
        self.down_f = 16000
        self.root = os.path.join(flags.data_dir, 'MFPT')

    def get_files(self):
        '''
        This function is used to generate the final training set and test set.
        root:The location of the data set
        '''
        m = os.listdir(self.root)
        m.sort(key=lambda x1: int(x1.split('-')[0]))
        # datasetname = os.listdir(os.path.join(root, m[0]))
        # '1 - Three Baseline Conditions'
        # '2 - Three Outer Race Fault Conditions'
        # '3 - Seven More Outer Race Fault Conditions'
        # '4 - Seven Inner Race Fault Conditions'
        # '5 - Analyses',
        # '6 - Real World Examples
        # Generate a list of data
        dataset1 = os.listdir(os.path.join(self.root, m[0]))  # 'Three Baseline Conditions'
        dataset2 = os.listdir(os.path.join(self.root, m[2]))  # 'Seven More Outer Race Fault Conditions'
        dataset2.sort(key=lambda x1: int(x1.split('.')[0].split('_')[2]))
        dataset3 = os.listdir(os.path.join(self.root, m[3]))  # 'Seven Inner Race Fault Conditions'
        dataset3.sort(key=lambda x1: int(x1.split('.')[0].split('_')[2]))
        data_root1 = os.path.join(self.root, m[0])  # Path of Three Baseline Conditions
        data_root2 = os.path.join(self.root, m[2])  # Path of Seven More Outer Race Fault Conditions
        data_root3 = os.path.join(self.root, m[3])  # Path of Seven Inner Race Fault Conditions

        domain_num = 0
        # data = {}
        # lab = {}
        # data[str(domain_num)] = []
        # lab[str(domain_num)] = []
        data = []
        lab = []

        path1 = os.path.join(data_root1, dataset1[0])
        data0, lab0 = self.data_load(path1, label=0)  # The label for normal data is 0
        # data[str(domain_num)] += data0
        # lab[str(domain_num)] += lab0
        data += data0
        lab += lab0

        for i in range(len(dataset2)):
            path2 = os.path.join(data_root2, dataset2[i])
            data1, lab1 = self.data_load(path2, label=1)
            data += data1
            lab += lab1

        for j in range(len(dataset3)):
            path3 = os.path.join(data_root3, dataset3[j])

            data2, lab2 = self.data_load(path3, label=2)
            data += data2
            lab += lab2

        return data, lab#, len(lab)

    def data_load(self, filename, label):
        '''
        This function is mainly used to generate test data and training data.
        filename:Data location
        '''
        if label == 0:
            fl = (loadmat(filename)["bearing"][0][0][1])  # Take out the data
        else:
            fl = (loadmat(filename)["bearing"][0][0][2])  # Take out the data

        # fl = fl.reshape(-1, )
        fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=0)

        data = []
        lab = []
        start, end = 0, signal_size
        while end <= fl.shape[0]:
            x = fl[start:end]
            x = np.fft.fft(x)
            x = np.abs(x) / len(x)
            x = x[range(int(x.shape[0] / 2))]
            x = x.reshape(1, -1)
            data.append(x)
            lab.append(label)
            start += signal_size
            end += signal_size
        return data, lab


def get_data_loader(args):
    train_x = {}
    train_y = {}
    test_x = {}
    test_y = {}
    domain_a = CWRU(args, task='[cwru1]', sub_root=0)
    train_x['0'], train_y['0'] = domain_a.get_files()
    domain_b = CWRU(args, task='[cwru9]', sub_root=2)
    train_x['1'], train_y['1'] = domain_b.get_files()
    domain_c = PU(args, task='[pu1]', data_num=2, condition='[2]')
    train_x['2'], train_y['2'] = domain_c.get_files()
    domain_d = PU(args, task='[pu1]', data_num=2, condition='[1]')
    train_x['3'], train_y['3'] = domain_d.get_files()

    domain_e = CWRU(args, domain='B')  # MFPT(args)
    test_x['0'], test_y['0'] = domain_e.get_files()
    tr_num = 2
    te_num = 1
    # if args.train_data_name == 'PU':
    #     train_data = PU(args)
    #     train_x, train_y, tr_num = train_data.get_files()
    # else:
    #     train_data = CWRU-full-normal(args, sub_root=0, domain='A')
    #     train_x, train_y, tr_num = train_data.get_files()
    #
    # if args.test_data_name == 'PU':
    #     test_data = PU(args)
    #     test_x, test_y, te_num = test_data.get_files()
    # else:
    #     test_data = CWRU-full-normal(args, sub_root=1)
    #     test_x, test_y, te_num = test_data.get_files()

    # train_priori = CWRU-full-normal(args, test=False, sub_root=0, priori=True)
    # p_x, p_y, p_num = train_priori.get_files()

    rate = 0.2
    trdatalist, tedatalist, mtedatalist, valdatalist = [], [], [], []

    # valdatalist.append(DataGenerate(args=args, dir=args.test_data_dir, dataset=args.test_data_name, task=args.task,
    #                         domain_data=p_x['0'], labels=p_y['0']))

    for i in range(te_num):
        tedatalist.append(DataGenerate(args=args, domain_data=test_x[str(i)], labels=test_y[str(i)]))

    for i in range(tr_num):
        if i in args.test_envs:     # 划分测试domain image_test(): 用于初始化图像数据
            tedatalist.append(DataGenerate(args=args, domain_data=train_x[str(i)], labels=train_y[str(i)],))
        else:  # 划分训练domain，并在训练域中进一步划分训练集/测试集
            tmpdatay = DataGenerate(args=args, domain_data=train_x[str(i)], labels=train_y[str(i)],).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)   # 划分训练/验证集
                stsplit.get_n_splits(lslist, tmpdatay)   # 返回n_splits
                indextr, indexval = next(stsplit.split(lslist, tmpdatay))   # (数字索引，对应标签)，划分2组，交叉验证
                np.random.seed(args.seed)
                indexmte = np.random.permutation(indextr)

            trdatalist.append(DataGenerate(args=args, domain_data=train_x[str(i)], labels=train_y[str(i)], indices=indextr))
            mtedatalist.append(DataGenerate(args=args, domain_data=train_x[str(i)], labels=train_y[str(i)], indices=indexmte,))
            valdatalist.append(DataGenerate(args=args, domain_data=train_x[str(i)], labels=train_y[str(i)], indices=indexval,))

    train_loaders = [InfiniteDataLoader(   # 这是一个自编无限数据loader
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    meta_test_loaders = [InfiniteDataLoader(  # 这是一个自编无限数据loader
        dataset=env,
        weights=None,
        batch_size=args.batch_size_metatest,
        num_workers=args.N_WORKERS)
        for env in mtedatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=32,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in valdatalist]

    test_loaders = [DataLoader(
        dataset=env,
        batch_size=32,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]

    return train_loaders, meta_test_loaders, eval_loaders, test_loaders, tr_num


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='cnn_2d', help='the name of the model')
    parser.add_argument('--train_data_name', type=str, default='PU', help='the name of the data')
    parser.add_argument('--test_data_name', type=str, default='CWRU-full-normal', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default="/home/yfy/Desktop/Dataset/",
                        help='the directory of the training data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='R_NA',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./logs',
                        help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--batch_size_metatest', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--seed', type=int, default=0)

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')

    # others needed
    parser.add_argument('--test_envs', type=list, default=[], help='test domains')
    parser.add_argument('--task', type=str, default='wc3', help='unseen domains')
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--N_WORKERS', type=int, default=2)
    args = parser.parse_args()
    args.train_data_dir = os.path.join(args.data_dir, args.train_data_name)
    args.test_data_dir = os.path.join(args.data_dir, args.test_data_name)
    return args


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    few = CWRUFew(args, task='cwru1', sub_root=0)
    a_x, a_y = few.get_files()


    train_loaders, meta_test_loaders, eval_loaders, test_loaders, domain_num = get_data_loader(args)
    train_minibatches_iterator = zip(*train_loaders)
    # for i in range(100):
    #     minibatches_device = [data for data in next(train_minibatches_iterator)]
    for count, bat in enumerate(eval_loaders):
        for data in bat:
            x = data[0]
            y = data[1]

    for c, b in enumerate(test_loaders):
        for d in b:
            xt = d[0]
            yt = d[1]
