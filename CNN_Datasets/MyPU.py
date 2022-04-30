import os
from scipy.io import loadmat
from datasets.sequence_aug import *
from datautil.getdataloader import get_data_loader

signal_size = 1024

# # 1 Undamaged (healthy) bearings(6X)
# HBdata = ['K001', "K002", 'K003', 'K004', 'K005', 'K006']
# # 2 Artificially damaged bearings(12X)
# ADBdata = ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09', 'KI01', 'KI03', 'KI05', 'KI07', 'KI08']
# # 3 Bearings with real damages caused by accelerated lifetime tests(14x)
# RDBdata = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30', 'KB23', 'KB24', 'KB27', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21']


# working condition
WC = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
# health, OR, IR
data_group1 = {'0': ['K001', "K002", 'K003'], '1': ['KA01', 'KA03', 'KA05'], '2': ['KI01', 'KI07', 'KI08']}
data_group2 = {'0': ['K004', 'K005', 'K006'], '1': ['KA04', 'KA16', 'KA30'], '2': ['KI14', 'KI17', 'KI18']}
domain_name = {'0': 'g1-WC1', '1': 'g1-WC2', '2': 'g1-WC3', '3': 'g1-WC4', '4': 'g2-WC1', '5': 'g2-WC2',
               '6': 'g2-WC3', '7': 'g2-WC4'}


# generate Training Dataset and Testing Dataset
def get_files(root, data_group):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = {}
    lab = {}
    domain_num = 0
    sample_num = 0

    for i in data_group:
        for state in WC:
            data[str(domain_num)] = []
            lab[str(domain_num)] = []
            for (k, v) in i.items():
                for j in v:
                    name1 = state+"_" + j + "_" + '1'
                    path1 = os.path.join(root, j, name1+".mat")
                    data1, lab1 = data_load(path1, name=name1, label=int(k))
                    data[str(domain_num)] += data1
                    lab[str(domain_num)] += lab1
                    sample_num += len(lab1)

            domain_num += 1
    return data, lab, sample_num


def data_load(filename, name, label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = loadmat(filename)[name]
    fl = fl[0][0][2][0][6][2]  # Take out the data
    fl = fl.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


# --------------------------------------------------------------------------------------------------------------------
class MyPU(object):
    num_classes = 3
    inputchannel = 1

    def __init__(self, args):
        self.data_dir = args.data_dir
        args.domain_name = domain_name
        args.source_domains = [data_group1, data_group2]
        self.normlizetype = args.normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, args):
        # get source train and val
        train_x, train_y, num_samples = get_files(self.data_dir, args.source_domains)
        train_loader, eval_loader = get_data_loader(train_x, train_y, num_samples, args)
        return train_loader, eval_loader
