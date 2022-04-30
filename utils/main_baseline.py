# coding=utf-8

from mymodels.CNN_1d import CNN
from datautil.getdataloader import *
from utils.train_utils import set_random_seed
from alg import modelopera
import time



def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters   data_name/dataset
    parser.add_argument('--data_dir', type=str, default="/home/yfy/Desktop/Dataset/",
                        help='the directory of the data')
    parser.add_argument('--normlizetype', type=str, choices=['0-1', '1-1', 'mean-std'], default='0-1',
                        help='data normalization methods')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--batch_size_metatest', type=int, default=64, help='batchsize of the meta-test process')
    parser.add_argument('--N_WORKERS', type=int, default=0, help='the number of training process')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=3,
                        help="number of classes")
    parser.add_argument("--meta_iteration_size", type=int, default=1,
                        help='iteration of test domains')
    parser.add_argument("--beta", type=float, default=1,
                        help='learning rate of the dg function')
    parser.add_argument('--lr', type=float, default=5e-4, help='the initial learning rate')
    parser.add_argument("--logs", type=str, default='logs/PU/Feature_Critic/',
                        help='logs folder to write log')
    parser.add_argument("--load_path", type=str, default='model_output/PU/Baseline/',
                        help='folder for loading baseline model')
    parser.add_argument("--model_path", type=str, default='model_output/PU/Feature_Critic/',
                        help='folder for saving model')
    parser.add_argument("--count_test", type=int, default=1,
                        help='the amount of episode for testing our method')
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    feat_cnn = CNN(out_channel=args.num_classes).cuda()
    feat_cnn.train()
    optimizer = torch.optim.SGD(feat_cnn.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    # tr_data = PUAll(args, task='[pu1, pu2]')
    # tr_x, tr_y = tr_data.get_files()
    # te_data = PUAll(args, task='[pu3]', condition='[0]', test=True)
    # te_x, te_y = te_data.get_files()

    tr_data1 = CWRUAll(args, domain='C')
    tr_x, tr_y = tr_data1.get_files()
    # tr_data2 = CWRUAll(args, domain='C')
    # tr_x2, tr_y2 = tr_data2.get_files()
    # tr_x, tr_y = tr_x1+tr_x2, tr_y1+tr_y2
    te_data = CWRUAll(args, task='[cwru6]', sub_root=1)
    te_x, te_y = te_data.get_files()

    # tr_x = []
    # tr_y = []
    # te_x = []

    # te_y = []
    # domain_a = CWRUFew(args, task='cwru1', sub_root=0)
    # tr_x, tr_y = domain_a.get_files()
    # domain_b = CWRUFew(args, task='cwru2', sub_root=0)
    # tr_x['1'], tr_y['1'] = domain_b.get_files()
    # domain_c = CWRUFew(args, task='cwru3', sub_root=0)
    # train_x['2'], train_y['2'] = domain_c.get_files()
    # domain_d = CWRUFew(args, task='cwru4', sub_root=0)
    # train_x['3'], train_y['3'] = domain_d.get_files()
    # domain_e = CWRUFew(args, task='cwru2', sub_root=0)
    # te_x, te_y = domain_e.get_files()

    trdatalist = DataGenerate(args=args, domain_data=tr_x, labels=tr_y)
    tedatalist = DataGenerate(args=args, domain_data=te_x, labels=te_y)

    train_loaders = DataLoader(dataset=trdatalist,
                               batch_size=args.batch_size,
                               num_workers=args.N_WORKERS,
                               drop_last=False,
                               shuffle=True)

    test_loaders = DataLoader(dataset=tedatalist,
                              batch_size=args.batch_size,
                              num_workers=args.N_WORKERS,
                              drop_last=False,
                              shuffle=False)

    start_time1 = time.time()
    for epoch in range(30):
        for step, (x, y) in enumerate(train_loaders):
            b_x = x.cuda().float()
            b_y = y.cuda().long()

            output = feat_cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if step % 50 == 0:
            #     accuracy = modelopera.accuracy_cnn(feat_cnn, test_loaders)
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(),
            #           '| test accuracy: %.2f' % accuracy)

    end_time1 = time.time()
    print('training time: %.4f seconds' % (end_time1 - start_time1))

    start_time2 = time.time()
    final_acc = modelopera.accuracy_cnn(feat_cnn, test_loaders)
    end_time2 = time.time()
    print('test time: %.4f seconds' % (end_time2 - start_time2))

# !!!!!!!! Change in here !!!!!!!!! #
# pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

# 如果需要使用numpy、可视化工具等，还需要将pred移至cpu
# pred_y = pred_y.cpu()

print('final accuracy: %.4f' % final_acc)
# print(test_y[:10], 'real number')


