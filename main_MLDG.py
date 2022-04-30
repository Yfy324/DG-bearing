# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse

from alg.opt import *
from alg import alg, modelopera
from utils.MLDGutil import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ
from datautil.getdataloader import get_data_loader


def get_args():
    parser = argparse.ArgumentParser(description='DG')
    # parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--algorithm', type=str, default="MLDG")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=1, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    # parser.add_argument('--dataset', type=str, default='office')
    parser.add_argument('--dataset', type=str, default='PACS')
    # parser.add_argument('--data_dir', type=str, default='', help='data dir')
    parser.add_argument('--data_dir', type=str, default='/home/yfy/Desktop/Dataset/', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=5e-4, help="learning rate used in MLDG")   # 1e-2
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=100, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1.5, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase")
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--N_WORKERS', type=int, default=0)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[], help='target domains')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()
    args.steps_per_epoch = 50
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))   # Tee是自定义对象，用于重定向print输出，这里是初始化两个文件
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)   # 初始化数据集相关的参数
    # print_environ()               # 输出实验环境的信息
    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)   # 初始化设置，固定随机数

    loss_list = alg_loss_dict(args)   # 返回不同算法的loss种类
    train_loaders, meta_test_loaders, eval_loaders, test_loaders, domain_num = get_data_loader(args)  # 划分训练集和测试集loader，测试集=unseen域+0.2其他域，巧用索引进行split
    # eval_name_dict = train_valid_target_eval_names(args)  # 在上一步的基础上进一步定义 指明训练、验证、测试集的 字典，用于后续训练
    algorithm_class = alg.get_algorithm_class(args.algorithm)  # 确认算法类型
    algorithm = algorithm_class(args).cuda()   # 类似于实例化了对象
    algorithm.train()    # 开启torch的训练模型
    opt = get_optimizer(algorithm, args)  # 得到网络模型，初始化学习率 -> 优化器
    sch = get_scheduler(opt, args)  # 初始化策略

    s = print_args(args, [])
    print('=======hyper-parameter used========')
    # print(s)
    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    print('===========start training===========')
    sss = time.time()
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):   # 一个epoch，按照设置的迭代次数运行/优化，保留最后一次的objective
            minibatches_device = [data
                                  for data in next(train_minibatches_iterator)]
            if args.algorithm == 'VREx' and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)
            step_vals = algorithm.update(minibatches_device, opt, sch)

        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):   # 手动减少学习率
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):   # 设置输出
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in loss_list:
                s += (item+'_loss:%.4f,' % step_vals[item])
            print(s[:-1])     # 输出loss
            s = ''
            # for item in acc_type_list:
            #     acc_record[item] = np.mean(np.array([modelopera.accuracy(
            #         algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))    # 按照 训练/验证/测试 的顺序和已经指定好的域，分别计算对应part的准确率
            #     s += (item+'_acc:%.4f,' % acc_record[item])
            acc_record['train'] = np.mean(np.array([modelopera.accuracy(
                    algorithm, trloader) for count, trloader in enumerate(meta_test_loaders)]))
            s += ('train' + '_acc:%.4f,' % acc_record['train'])
            acc_record['eval'] = np.mean(np.array([modelopera.accuracy(
                algorithm, eloader) for count, eloader in enumerate(eval_loaders)]))
            s += ('eval' + '_acc:%.4f,' % acc_record['eval'])
            acc_record['test'] = np.mean(np.array([modelopera.accuracy(
                algorithm, teloader) for count, teloader in enumerate(test_loaders)]))
            s += ('test' + '_acc:%.4f,' % acc_record['test'])
            print(s[:-1])
            if acc_record['eval'] > best_valid_acc:
                best_valid_acc = acc_record['eval']
                target_acc = acc_record['test']
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time()-sss))
            algorithm_dict = algorithm.state_dict()

    save_checkpoint('model.pkl', algorithm, args)

    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)

    with open(os.path.join(args.output, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc:%.4f' % (target_acc))
