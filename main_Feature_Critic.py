import argparse
import os
from model_PACS import Model_Feature_Critic_PACS
import sys
import torch
import warnings
from utils.train_utils import set_random_seed
import time

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
sys.setrecursionlimit(1000000)


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters   data_name/dataset
    parser.add_argument('--model_name', type=str, default='cnn_1d', help='the name of the model')
    parser.add_argument('--train_data_name', type=str, default='PU', help='the name of the data')
    parser.add_argument('--test_data_name', type=str, default='CWRU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default="/home/yfy/Desktop/Dataset/",
                        help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--batch_size_metatest', type=int, default=64, help='batchsize of the meta-test process')
    parser.add_argument('--N_WORKERS', type=int, default=0, help='the number of training process')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--method", type=str, default='Feature_Critic',
                        help='Feature_Critic')
    parser.add_argument("--num_classes", type=int, default=3,
                        help="number of classes")
    parser.add_argument("--iteration_size", type=int, default=600,
                        help="iteration of training domains")
    parser.add_argument("--meta_iteration_size", type=int, default=1,
                        help='iteration of test domains')
    parser.add_argument("--beta", type=float, default=1,
                        help='learning rate of the dg function')
    parser.add_argument('--lr', type=float, default=5e-4, help='the initial learning rate')
    parser.add_argument("--heldout_p", type=float, default=3,
                        help='learning rate of the heldout function')
    parser.add_argument("--omega", type=float, default=7e-3,  # 7e-3
                        help='learning rate of the omega function')
    parser.add_argument("--logs", type=str, default='logs/PU/Feature_Critic/',
                        help='logs folder to write log')
    parser.add_argument("--load_path", type=str, default='model_output/PU/Baseline/',
                        help='folder for loading baseline model')
    parser.add_argument("--model_path", type=str, default='model_output/PU/Feature_Critic/',
                        help='folder for saving model')
    parser.add_argument("--count_test", type=int, default=1,
                        help='the amount of episode for testing our method')
    parser.add_argument('--test_envs', type=list, default=[], help='test domains')
    parser.add_argument('--task', type=str, default='wc3', help='unseen domains')
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument("--if_train", type=bool, default=True,
                        help='if we need to train to get the model')
    parser.add_argument("--if_test", type=bool, default=True,
                        help='if we want to test on the target data')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # dg_function = {'1': 'MLP', '2': 'Flatten_FTF'}
    dg_function = {'1': 'MLP'}
    for nn in range(len(dg_function)):
        args.type = dg_function[str(nn + 1)]
        # args.type = dg_function['2']
        for i in range(args.count_test):
            print('Episode %d in type %s.' % (i, args.type))
            model_obj = Model_Feature_Critic_PACS(flags=args)
            if args.if_train:
                start_time1 = time.time()
                model_obj.train(flags=args)
                end_time1 = time.time()
                print('training time: %.4f seconds' % (end_time1 - start_time1))
                torch.cuda.empty_cache()
            if args.if_test:
                start_time2 = time.time()
                model_obj.heldout_test(flags=args)
                end_time2 = time.time()
                print('test time: %.4f seconds' % (end_time2 - start_time2))
