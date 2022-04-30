#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils import TrainUtils, set_random_seed

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='cnn_2d', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='MyPU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default="/home/yfy/Desktop/Dataset/PU/",
                        help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--processing_type', type=str, choices=['R_A', 'R_NA', 'O_A'], default='R_NA',
                        help='R_A: random split with data augmentation, R_NA: random split without data augmentation, O_A: order split with data augmentation')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./logs', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--batch_size_metatest', type=int, default=32, help='batchsize of the meta-test process')
    parser.add_argument('--N_WORKERS', type=int, default=2, help='the number of training process')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--method", type=str, default='Feature_Critic',
                        help='Feature_Critic')
    parser.add_argument("--num_classes", type=int, default=3,
                        help="number of classes")
    parser.add_argument("--iteration_size", type=int, default=45000,
                        help="iteration of training domains")
    parser.add_argument("--meta_iteration_size", type=int, default=1,
                        help='iteration of test domains')

    # optimization information
    # parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=5e-4, help='the initial learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    # parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    # parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix',
    #                     help='the learning rate schedule')
    # parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    # parser.add_argument('--steps', type=str, default='9', help='the learning rate decay for step and stepLR')

    parser.add_argument("--type", type=str, default='MLP',
                        help='the type of dg functions, {MLP: 1, Flatten_FTF: 2}')
    parser.add_argument("--heldout_p", type=float, default=1,
                        help='learning rate of the heldout function')
    parser.add_argument("--omega", type=float, default=1e-3,
                        help='learning rate of the omega function')
    # save, load and display information
    # parser.add_argument('--max_epoch', type=int, default=100, help='max number of epoch')
    # parser.add_argument('--print_step', type=int, default=100, help='the interval of log training information')
    parser.add_argument("--load_path", type=str, default='model_output/PACS/Baseline/Feature_Critic/',
                        help='folder for loading baseline model')
    parser.add_argument("--model_path", type=str, default='model_output/PACS/Feature_Critic/Feature_Critic/',
                        help='folder for saving model')
    parser.add_argument("--debug", type=bool, default=True,
                        help='whether for debug mode or not')
    parser.add_argument("--count_test", type=int, default=1,
                        help='the amount of episode for testing our method')
    parser.add_argument("--if_train", type=bool, default=True,
                        help='if we need to train to get the model')
    parser.add_argument("--if_test", type=bool, default=True,
                        help='if we want to test on the target data')
    # others needed  unseen_index/test_envs
    parser.add_argument('--test_envs', type=list, default=[2, 6], help='test domains')
    parser.add_argument('--task', type=str, default='wc3', help='unseen domains')
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    set_random_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + args.data_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'training.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = TrainUtils(args, save_dir)
    trainer.setup()
    trainer.train()
