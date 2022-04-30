import os
import pandas as pd
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import DataSet
from datasets.sequence_aug import *
from tqdm import tqdm

signal_size = 1024
HBdata = ['K001', "K002", 'K003', 'K004', 'K005', 'K006']
WC = ["N15_M07_F10"]


def get_files(root, test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab = []

    for k in tqdm(range(len(HBdata))):
        name3 = WC[0] + "_" + HBdata[k] + "_1"
        path3 = os.path.join(root, HBdata[k], name3 + ".mat")
        data3, lab3 = data_load(path3, name=name3, label=0)
        data += data3
        lab += lab3

    return [data, lab]


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


if __name__ == '__main__':
    # root = '/home/yfy/Desktop/Dataset/PU/'
    # name3 = WC[0] + "_" + HBdata[0] + "_1"
    # path3 = os.path.join(root, HBdata[0], name3 + ".mat")
    # fl = loadmat(path3)[name3]
    # fl = fl[0][0][2][0][6][2]     # 得到的是时间数据点，采样频率是64kHz
    # fl = fl.reshape(-1)
    # M = 4
    # downsampled_pu = signal.decimate(fl, M)
    # downsampled_pu1 = signal.resample_poly(fl, 16000, 64000, axis=0)  # axis = 1 第二维度
    # fl_f = 64000
    # fl_size = 1024

    root = "/home/yfy/Desktop/Dataset/CWRU-full-normal/12k Fan End Bearing Fault Data/309.mat"
    realaxis = "X" + '309' + "_DE_time"
    fl = loadmat(root)[realaxis]
    # fl = fl.reshape(-1, 1)
    fl_f = 12000
    M = 3
    down_f = int(fl_f / M)
    fl_size = 1024
    # downsampled_cwru = signal.decimate(fl, M)
    downsampled_cwru1 = signal.resample_poly(fl, down_f, fl_f, axis=0)  # axis = 1 第二维度
    freqs, times, Sxx = signal.spectrogram(fl, fs=fl_f, window='hanning',
                                           nperseg=fl_size, noverlap=512,
                                           detrend=False, scaling='spectrum')
    plt.figure(1)
    plt.pcolormesh(times, freqs, 20 * np.log10(Sxx/1e-06), cmap='inferno')
    # plt.clim(70, 150)
    # plt.ylim(0, 3000)
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()

    down_f = fl_f / M
    f1, t1, sxx1 = signal.spectrogram(downsampled_cwru, fs=down_f, window='hanning',
                                      nperseg=fl_size, noverlap=512,
                                      detrend=False, scaling='spectrum')
    f2, t2, sxx2 = signal.spectrogram(downsampled_cwru1, fs=down_f, window='hanning',
                                      nperseg=fl_size, noverlap=512,
                                      detrend=False, scaling='spectrum')
    plt.figure(2)
    plt.pcolormesh(t1, f1, 20 * np.log10(sxx1/1e-06), cmap='inferno')
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()

    plt.figure(3)
    plt.pcolormesh(t2, f2, 20 * np.log10(sxx2/1e-06), cmap='inferno')
    plt.colorbar()
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()
