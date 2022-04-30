# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
from datautil.imgdata.imgdataload import DataGenerate
from datautil.mydataloader import InfiniteDataLoader
from datautil.imgdata.few_getloader import CWRUFew
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
WC = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
# health, OR, IR
pu1 = {'0': ['K001', "K002", 'K003'], '1': ['KA01', 'KA03', 'KA05'], '2': ['KI01', 'KI07', 'KI08']}
pu2 = {'0': ['K004', 'K005'], '1': ['KA06', 'KA07', 'KA08', 'KA09'], '2': ['KI03', 'KI05']}
pu3 = {'0': ['K004', 'K005', 'K006'], '1': ['KA04', 'KA16', 'KA30'], '2': ['KI14', 'KI17', 'KI18']}
pu4 = {'0': ['K006', 'K005', 'K006'],'1': ['KA15', 'KA22'], '2': ['KI04', 'KI16']}
domain_name = {'0': 'g1-WC1', '1': 'g1-WC2', '2': 'g1-WC3', '3': 'g1-WC4', '4': 'g2-WC1', '5': 'g2-WC2', '6': 'g2-WC3', '7': 'g2-WC4'}
# pu1 = {'0': ['K002'], '1': ['KA01', 'KA05', 'KA07'], '2': ['KI01', 'KI03', 'KI05', 'KI07']}
# pu2 = {'0': ['K001'], '1': ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'], '2': ['KI14', 'KI16', 'KI17', 'KI18', 'KI21']}
# CWRU-full-normal
axis = ["_DE_time", "_FE_time", "_BA_time"]
datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data"]
# For 12k Drive End Bearing Fault Data
cwru0 = {'0': ["97.mat"]}
cwru1 = {'0': ["97.mat"], '1': ["130.mat", "197.mat", "234.mat"], '2': ["105.mat", "169.mat", "209.mat"],
         # '3': ["118.mat", "185.mat", "222.mat"]
         }  # 1797rpm
cwru2 = {'0': ["98.mat"], '1': ["131.mat", "198.mat", "235.mat"], '2': ["106.mat", "170.mat", "210.mat"],
         # '3': ["119.mat", "186.mat", "223.mat"]
         }  # 1772rpm
cwru3 = {'0': ["99.mat"], '1': ["132.mat", "199.mat", "236.mat"], '2': ["107.mat", "171.mat", "211.mat"],
         # '3': ["120.mat", "187.mat", "224.mat"]
         }  # 1750rpm
cwru4 = {'0': ["100.mat"], '1': ["133.mat", "200.mat", "237.mat"], '2': ["108.mat", "172.mat", "212.mat"],
         # '3': ["121.mat", "188.mat", "225.mat"]
         }  # 1730rpm
# For 12k Fan End Bearing Fault Data      1的后两列数据不够统一，有些事@3的OR
cwru5 = {'0': ["97.mat"], '1': ["294.mat", "310.mat", "315.mat"], '2': ["278.mat", "274.mat", "270.mat"],
         # '3': ["282.mat", "286.mat", "290.mat"]
         }  # 1797rpm
cwru6 = {'0': ["98.mat"], '1': ["295.mat", "309.mat", "316.mat"], '2': ["279.mat", "275.mat", "271.mat"],
         # '3': ["283.mat", "287.mat", "291.mat"]
         }  # 1772rpm
cwru7 = {'0': ["99.mat"], '1': ["296.mat", "311.mat", "317.mat"], '2': ["280.mat", "276.mat", "272.mat"],
         # '3': ["284.mat", "288.mat", "292.mat"]
         }  # 1750rpm
cwru8 = {'0': ["100.mat"], '1': ["297.mat", "312.mat", "318.mat"], '2': ["281.mat", "277.mat", "273.mat"],
         # '3': ["285.mat", "289.mat", "293.mat"]
         }  # 1730rpm
# For 48k Drive End Bearing Fault Data                                               wrong?    217wrong?
cwru9 = {'0': ["97.mat"], '1': ["135.mat", "201.mat", "238.mat"], '2': ["109.mat", "173.mat", "213.mat"],
         # '3': ["122.mat", "189.mat", "226.mat"]
         }  # 1797rpm
cwru10 = {'0': ["98.mat"], '1': ["136.mat", "202.mat", "239.mat"], '2': ["110.mat", "175.mat", "214.mat"],
          # '3': ["123.mat", "190.mat", "227.mat"]
          }  # 1772rpm
cwru11 = {'0': ["99.mat"], '1': ["137.mat", "203.mat", "240.mat"], '2': ["111.mat", "176.mat", "215.mat"],
          # '3': ["124.mat", "191.mat", "228.mat"]
          }  # 1750rpm
cwru12 = {'0': ["100.mat"], '1': ["138.mat", "204.mat", "241.mat"], '2': ["112.mat", "177.mat", "217.mat"],
          # '3': ["125.mat", "192.mat", "229.mat"]
          }  # 1730rpm
domain_cwru = {
    'A': {'data': [cwru1, cwru2, cwru3, cwru4], 'subroot': 0, 'axis': 0, 'frequency': 12000, 'down_f': 12000},
    'B': {'data': [cwru5, cwru6, cwru7, cwru8], 'subroot': 1, 'axis': 1, 'frequency': 12000, 'down_f': 12000},
    'C': {'data': [cwru9, cwru10, cwru11, cwru12], 'subroot': 2, 'axis': 0, 'frequency': 48000, 'down_f': 16000}
}


class CWRUAll(object):
    def __init__(self, flags, domain='A', task=None, sub_root=None, test=False):
        self.test = test
        if task is None:
            self.domain = domain_cwru[domain]
            self.data = self.domain['data']
            self.root = os.path.join(flags.data_dir, 'CWRU', datasetname[self.domain['subroot']])
            self.frequency = self.domain['frequency']
            self.down_f = self.domain['down_f']
            self.axis = axis[self.domain['axis']]
        else:
            self.data = eval(task)
            self.sub_root = datasetname[sub_root]
            self.root = os.path.join(flags.data_dir, 'CWRU', self.sub_root)
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
        # data = {}
        # lab = {}
        data = []
        lab = []
        domain_num = 0
        # data[str(domain_num)] = []
        # lab[str(domain_num)] = []

        for i in self.data:
            # data[str(domain_num)] = []
            # lab[str(domain_num)] = []
            for (k, v) in i.items():
                for j in v:
                    path1 = os.path.join(self.root, j)
                    data1, lab1 = self.data_load(path1, j, label=int(k))
                    # data[str(domain_num)] += data1
                    # lab[str(domain_num)] += lab1
                    data += data1
                    lab += lab1

            # domain_num += 1
        return data, lab#, len(lab)

    def data_load(self, filename, axisname, label):
        datanumber = axisname.split(".")
        if eval(datanumber[0]) < 100:
            realaxis = "X0" + datanumber[0] + self.axis
        else:
            realaxis = "X" + datanumber[0] + self.axis
        fl = loadmat(filename)[realaxis]
        # fl = fl.reshape(1, -1)
        fl = signal.resample_poly(fl, self.down_f,self.frequency, axis=0)
        data = []
        lab = []
        fl_middle = int(fl.shape[0]/2)
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


class PUAll(object):
    def __init__(self, flags, task='[pu1, pu2]', condition='[0]', data_num=1, test=False):
        # self.WC = eval(condition)
        if test:
            self.WC = WC
        else:
            self.WC = WC
        self.frequency = 64000
        self.down_f = 16000
        self.root = os.path.join(flags.data_dir, 'PU')
        self.data = eval(task)
        self.d_num = data_num + 1
        # self.d_num = 3

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
                            name1 = state + "_" + j + "_" + '1'
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
        # fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=1)
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


class PU(object):
    def __init__(self, flags, task='[pu1, pu2]', condition='[0]', data_num=1, test=False):
        # self.WC = eval(condition)
        if test:
            self.WC = WC
        else:
            self.WC = WC
        self.test = test
        self.frequency = 64000
        self.down_f = 16000
        self.root = os.path.join(flags.data_dir, 'PU')
        self.data = eval(task)
        self.d_num = data_num + 1
        # self.d_num = 3

    def get_files(self):
        data = {}
        lab = {}
        # data = []
        # lab = []
        domain_num = 0

        for i in self.data:
            for state in self.WC:
                data[str(domain_num)] = []
                lab[str(domain_num)] = []
                for (k, v) in i.items():
                    for j in v:
                        for num in range(1, self.d_num):
                            name1 = state + "_" + j + "_" + '1'
                            path1 = os.path.join(self.root, j, name1 + ".mat")
                            data1, lab1 = self.data_load(path1, name=name1, label=int(k))
                            data[str(domain_num)] += data1
                            lab[str(domain_num)] += lab1
                            # data += data1
                            # lab += lab1

                domain_num += 1
        return data, lab, len(lab)

    def data_load(self, filename, name, label):
        fl = loadmat(filename)[name]
        fl = fl[0][0][2][0][6][2]  # Take out the data
        # fl = fl.reshape(-1,)
        # fl = signal.resample_poly(fl, self.down_f, self.frequency, axis=1)
        data = []
        lab = []
        # fl_middle = int(fl.shape[1] / 2)
        # if label == 0:
        #     if self.test:
        #         start, end = fl_middle, (fl_middle+signal_size)
        #         while end <= fl.shape[1]:
        #             x = fl[:, start:end]
        #             x = np.fft.fft(x)
        #             x = np.abs(x) / len(x)
        #             x = x[:, range(int(x.shape[1] / 2))]
        #             # x = x.reshape(-1, 1)
        #             data.append(x)
        #             lab.append(label)
        #             start += signal_size
        #             end += signal_size
        #     else:
        #         start, end = 0, signal_size
        #         while end <= fl_middle:
        #             x = fl[:, start:end]
        #             x = np.fft.fft(x)
        #             x = np.abs(x) / len(x)
        #             x = x[:, range(int(x.shape[1] / 2))]
        #             # x = x.reshape(-1, 1)
        #             data.append(x)
        #             lab.append(label)
        #             start += signal_size
        #             end += signal_size
        # else:
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


class CWRU(object):
    def __init__(self, flags, domain='A', task=None, sub_root=None, test=False):
        self.test = test
        if task is None:
            self.domain = domain_cwru[domain]
            self.data = self.domain['data']
            self.root = os.path.join(flags.data_dir, 'CWRU', datasetname[self.domain['subroot']])
            self.frequency = self.domain['frequency']
            self.down_f = self.domain['down_f']
            self.axis = axis[self.domain['axis']]
        else:
            self.data = eval(task)
            self.sub_root = datasetname[sub_root]
            self.root = os.path.join(flags.data_dir, 'CWRU', self.sub_root)
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
        data = {}
        lab = {}
        # data = []
        # lab = []
        domain_num = 0
        data[str(domain_num)] = []
        lab[str(domain_num)] = []

        for i in self.data:
            data[str(domain_num)] = []
            lab[str(domain_num)] = []
            for (k, v) in i.items():
                for j in v:
                    path1 = os.path.join(self.root, j)
                    data1, lab1 = self.data_load(path1, j, label=int(k))
                    data[str(domain_num)] += data1
                    lab[str(domain_num)] += lab1
                    # data += data1
                    # lab += lab1

            domain_num += 1
        return data, lab, len(lab)

    def data_load(self, filename, axisname, label):
        datanumber = axisname.split(".")
        if eval(datanumber[0]) < 100:
            realaxis = "X0" + datanumber[0] + self.axis
        else:
            realaxis = "X" + datanumber[0] + self.axis
        fl = loadmat(filename)[realaxis]
        # fl = fl.reshape(1, -1)
        fl = signal.resample_poly(fl, self.down_f,self.frequency, axis=0)
        data = []
        lab = []
        fl_middle = int(fl.shape[0]/2)
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

    domain_a = CWRUFew(args, task='cwru3', sub_root=0)
    train_x['0'], train_y['0'] = domain_a.get_files()
    domain_b = CWRUFew(args, task='cwru2', sub_root=0)
    train_x['1'], train_y['1'] = domain_b.get_files()
    # domain_c = CWRUFew(args, task='cwru3', sub_root=0)
    # train_x['2'], train_y['2'] = domain_c.get_files()
    # domain_d = CWRUFew(args, task='cwru4', sub_root=0)
    # train_x['3'], train_y['3'] = domain_d.get_files()
    domain_e = CWRUFew(args, task='cwru4', sub_root=0, test=True)
    test_x['0'], test_y['0'] = domain_e.get_files()

    # domain_a = CWRU(args, domain='A')
    # train_x['0'], train_y['0'] = domain_a.get_files()
    # # domain_b = CWRU(args, domain='C')  # task='[cwru2]', sub_root=0)
    # # train_x['1'], train_y['1'] = domain_b.get_files()
    # domain_e = CWRU(args, domain='B')  # MFPT(args)
    # test_x['0'], test_y['0'] = domain_e.get_files()

    # domain_e = CWRU(args, task='[cwru1]', sub_root=0)  # MFPT(args)
    # test_x['0'], test_y['0'] = domain_e.get_files()

    # domain_a = PU(args, task='[pu1]', data_num=1, condition='[0]')
    # train_x['0'], train_y['0'] = domain_a.get_files()
    # domain_b = PU(args, task='[pu1]', data_num=1, condition='[1]')
    # train_x['1'], train_y['1'] = domain_b.get_files()
    # domain_c = PU(args, task='[pu1]', data_num=1, condition='[2]')
    # train_x['2'], train_y['2'] = domain_c.get_files()
    # domain_d = PU(args, task='[pu1]', data_num=1, condition='[3]')
    # train_x['3'], train_y['3'] = domain_d.get_files()
    # domain_e = PU(args, task='[pu2]', data_num=1, condition='[0]')
    # test_x['0'], test_y['0'] = domain_e.get_files()

    tr_num = 2
    te_num = 1

# {} as input
#     train_data = PU(args, task='[pu1,pu2]')
#     train_x, train_y, tr_num = train_data.get_files()
#     test_data = PU(args, task='[pu3]', test=True)
#     test_x, test_y, te_num = test_data.get_files()
    # AGG
    # train_data = PUAll(args, task='[pu1,pu2]')
    # train_x['0'], train_y['0'] = train_data.get_files()
    # test_data = PU(args, task='[pu3]', test=True)
    # test_x, test_y, te_num = test_data.get_files()

    # train_data1 = CWRU(args, domain='B')   # task='[cwru1,cwru2,cwru3,cwru4]', sub_root=0)
    # train_x, train_y, tr_num = train_data1.get_files()
    # train_data2 = CWRU(args, domain='C')
    # train_x1, train_y1, tr_num1 = train_data2.get_files()
    # for i in range(tr_num1):
    #     train_x[str(tr_num+i)] = train_x1[str(i)]
    #     train_y[str(tr_num+i)] = train_y1[str(i)]
    #
    # tr_num += tr_num1

    # test_data = CWRU(args, domain='C', test=True)
    # test_x, test_y, te_num = test_data.get_files()
    # test_data = CWRU(args, task='[cwru7]', sub_root=1, test=True)
    # test_x, test_y, te_num = test_data.get_files()

    # # AGG
    # tr_data1 = CWRUAll(args, domain='A')
    # train_x['0'], train_y['0'] = tr_data1.get_files()
    # test_data = CWRU(args, domain='B', test=True)
    # test_x, test_y, te_num = test_data.get_files()
    # tr_num = 1

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

    # meta_test_loaders = [DataLoader(  # 这是一个自编无限数据loader
    #     dataset=env,
    #     batch_size=args.batch_size,
    #     num_workers=args.N_WORKERS,
    #     drop_last=False,
    #     shuffle=False)
    #     for env in trdatalist]

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
