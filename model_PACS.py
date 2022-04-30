import os

import torch.nn
from torch.autograd import Variable

import mymodels
from datasets.data_gen_PACS import *
from utils.FCutils import *
import datetime
from datautil.getdataloader import get_data_loader


class ModelBaseline_PACS:
    def __init__(self, flags):
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.configure(flags)   # 日志路径
        self.setup_path(flags)  # 数据，数据生成器
        self.init_network_parameter(flags)  # 网络初始化参数，预训练Alexnet
        
        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)
        if not os.path.exists(flags.model_path):
            os.mkdir(flags.model_path)

    def __del__(self):
        print('release source')

    def configure(self, flags):
        self.flags_log = os.path.join(flags.logs, '%s.txt'%(flags.method))
        self.activate_load_model = False

    def setup_path(self, flags):
        self.best_accuracy_val = -1
        self.train_loaders, self.meta_test_loaders, self.eval_loaders, self.test_loaders, self.num_domain = get_data_loader(flags)

        self.num_test_domain = 0
        self.num_train_domain = self.num_domain - self.num_test_domain

        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log('get data loaders', flags_log)

    def init_network_parameter(self, flags):

        self.weight_decay = 5e-5
        self.batch_size = flags.batch_size

        self.h = 64  # 64   1024pu 表现好
        self.hh = 16

        ######################################################
        # self.feature_extractor_network = alexnet(pretrained=True).cuda()
        # self.feature_extractor_network = mymodels.CNN_1d.CNN(in_channel=1).cuda()
        self.feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained, in_channel=1).cuda()

        # phi means the classifer network parameter, from h (the output feature layer of input data) to c (the number of classes).
        self.para_theta = self.feature_extractor_network.parameters()   # nn.parameters()将‘参数’转为可不断优化的参数
        self.phi = classifier_homo(self.h, flags.num_classes)
        self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr, weight_decay=self.weight_decay, momentum=0.9)
        self.opt_theta = torch.optim.SGD(self.para_theta, lr=flags.lr,
                                         weight_decay=self.weight_decay, momentum=0.9)
        # self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=flags.lr,
        #                             weight_decay=self.weight_decay)
        # self.opt_theta = torch.optim.Adam(self.para_theta, lr=flags.lr,
        #                                   weight_decay=self.weight_decay)

        self.ce_loss = torch.nn.CrossEntropyLoss()

    def load_state_dict(self, state_dict=''):

        tmp = torch.load(state_dict)
        pretrained_dict = tmp[0]
        # load the new state dict
        self.feature_extractor_network.load_state_dict(pretrained_dict)
        self.phi.load_state_dict(tmp[1])

    def heldout_test(self, flags):

        # load the best model on the validation data
        model_path = os.path.join(flags.model_path, 'best_model.tar')
        self.load_state_dict(state_dict=model_path)
        self.feature_extractor_network.eval()
        self.phi.eval()

        for count, bat in enumerate(self.test_loaders):
            test_preds = []
            test_labels = []
            for data in bat:
                x_data = data[0].cuda().float()
                y_label = data[1].view(len(data[1]), 1)
                y_data = torch.zeros(y_label.shape[0], flags.num_classes).scatter_(1, y_label, 1)
                y_data = y_data.numpy()
                classifier_out = self.phi(self.feature_extractor_network(x_data)).data.cpu().numpy()
                test_preds.append(classifier_out)
                test_labels.append(y_data)

            test_classifier_output = np.concatenate(test_preds)
            labels_test = np.concatenate(test_labels)
            torch.cuda.empty_cache()
            accuracy = compute_accuracy(predictions=test_classifier_output, labels=labels_test)
            # precision = np.mean(test_classifier_output == labels_test)
            print(accuracy)
            # print(precision)
            # accuracy
            # accuracy_info = '\n the test domain %s' % (flags.test_envs[count])
            accuracy_info = '\n the test task'
            flags_log = os.path.join(flags.logs, 'heldout_test_log.txt')
            write_log(accuracy_info, flags_log)
            write_log(accuracy, flags_log)
            # write_log(precision, flags_log)

    def train_newloss(self, flags):

        write_log(flags, self.flags_log)  # 输出配置信息
        # self.pre_train(flags)  # 路径准备
        # self.reinit_network_P(flags)  # 训练更新的参数
        time_start = datetime.datetime.now()
        train_minibatches_iterator = zip(*self.train_loaders)
        train_minibatches_iterator_m = zip(*self.meta_test_loaders)

        for _ in range(flags.iteration_size):
            self.iteration = _
            if _ == 15000:  # 自定义学习率调整
                self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr / 10, weight_decay=self.weight_decay,
                                               momentum=0.9)
                self.opt_theta = torch.optim.SGD(self.feature_extractor_network.parameters(), lr=flags.lr / 10,
                                                 weight_decay=self.weight_decay, momentum=0.9)
                # self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=flags.lr/10,
                #                             weight_decay=self.weight_decay)
                # self.opt_theta = torch.optim.Adam(self.feature_extractor_network.parameters(), lr=flags.lr/10,
                #                                   weight_decay=self.weight_decay)

            self.feature_extractor_network.train()  # 特征提取
            self.phi.train()  # 分类器

            index = np.random.permutation(self.num_train_domain)
            meta_train_idx = index[0:(self.num_domain-1)]
            meta_test_idx = index[(self.num_domain-1):]
            # meta_train_idx = index[0:]
            # meta_test_idx = index[0:]
            # meta_train_idx = index[0:5]
            # meta_test_idx = index[5:]
            write_log('-----------------iteration_%d--------------' % (_), self.flags_log)
            write_log(meta_train_idx, self.flags_log)
            write_log(meta_test_idx, self.flags_log)

            for itr in range(flags.meta_iteration_size):
                meta_train_loss_main = 0.0
                meta_train_loss_dg = 0.0
                meta_loss_held_out = 0.0

                for i in meta_train_idx:
                    minibatches_device = [data for data in next(train_minibatches_iterator)]
                    domain_a_x, domain_a_y = minibatches_device[i][0], minibatches_device[i][
                        1]  # .get_images_labels_batch()   # 类似于无限数据生成器
                    x_subset_a = domain_a_x.cuda().float()
                    y_subset_a = domain_a_y.cuda().long()

                    feat_a = self.feature_extractor_network(x_subset_a)  # 提取特征
                    pred_a = self.phi(feat_a)  # 预测类别

                    loss_main = self.ce_loss(pred_a, y_subset_a)  # 对训练数据，计算ce_loss eq.1
                    meta_train_loss_main += loss_main  # 特征提取器、分类的loss

                    self.omega = Critic_Network_MLP(self.h, self.hh).cuda()
                    loss_dg = self.beta * self.omega(feat_a)  # aux_loss eq.6/7

                    meta_train_loss_dg += loss_dg  # 辅助loss

                self.opt_phi.zero_grad()
                self.opt_theta.zero_grad()
                meta_train_loss_main.backward(retain_graph=True)

                grad_theta = {j: theta_i.grad for (j, theta_i) in
                              self.feature_extractor_network.named_parameters()}  # 记录网络参数的梯度
                theta_updated_old = {}  # 对应算法theta(old)的计算，相当于自己实现了opt.step()
                '''
                for (k, v), g in zip(self.feature_extractor_network.state_dict().items(),grad_theta):
                    theta_updated[k] = v - self.alpha * g
                '''
                # Todo: fix the new running_mean and running_var
                # Because Resnet18 network contains BatchNorm structure, there is no gradient in BatchNorm with running_mean and running_var.
                # Therefore, these two factors should be avoided in the calculation process of theta_old and theta_new.
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):
                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_old[k] = v
                        else:
                            theta_updated_old[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                meta_train_loss_dg.backward(create_graph=True)

                grad_theta = {m: theta_i.grad for (m, theta_i) in self.feature_extractor_network.named_parameters()}
                theta_updated_new = {}  # 对应算法theta(old)的计算
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):

                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_new[k] = v
                        else:
                            theta_updated_new[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                temp_new_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained,
                                                                                         in_channel=1).cuda()
                fix_nn(temp_new_feature_extractor_network, theta_updated_new)  # 把更新后的new网络参数保存下来
                temp_new_feature_extractor_network.train()

                temp_old_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained,
                                                                                         in_channel=1).cuda()
                fix_nn(temp_old_feature_extractor_network, theta_updated_old)
                # temp_old_feature_extractor_network.load_state_dict(theta_updated_old)   # 常规网络，故不用自编函数保存参数
                temp_old_feature_extractor_network.train()

                for i in meta_test_idx:
                    minibatches_device_m = [data_m for data_m in next(train_minibatches_iterator_m)]
                    domain_b_x, domain_b_y = minibatches_device_m[i][0], minibatches_device_m[i][1]
                    x_subset_b = domain_b_x.cuda().float()
                    y_subset_b = domain_b_y.cuda().long()

                    feat_b_old = temp_old_feature_extractor_network(x_subset_b).detach()
                    feat_b_new = temp_new_feature_extractor_network(x_subset_b).detach()

                    cls_b_old = self.phi(feat_b_old)
                    cls_b_new = self.phi(feat_b_new)

                    loss_main_old = self.ce_loss(cls_b_old, y_subset_b)
                    loss_main_new = self.ce_loss(cls_b_new, y_subset_b)
                    reward = loss_main_old - loss_main_new  # 新、旧ce_loss的差
                    # calculate the updating rule of omega, here is the max function of h.
                    utility = torch.tanh(reward)  # eq.5
                    # so, here is the min value transfering to the backpropogation.
                    loss_held_out = - utility.sum()  # eq.4
                    meta_loss_held_out += loss_held_out * self.heldout_p  # meta-test loss，用于更新feature-critic

                self.opt_theta.step()
                self.opt_phi.step()

                self.opt_omega.zero_grad()
                meta_loss_held_out.backward()
                self.opt_omega.step()
                torch.cuda.empty_cache()  # 释放GPU显存空间

                print('episode %d' % (_), meta_train_loss_main.data.cpu().numpy(),
                      meta_train_loss_dg.data.cpu().numpy(),
                      meta_loss_held_out.data.cpu().numpy(),
                      )

            if _ % 50 == 0 and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) % 500
                time_cost = epoch * (time_end - time_start).seconds / 60  # 转化为分钟
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (
                _, time_cost))
                torch.cuda.empty_cache()
                self.validate_workflow(self.eval_loaders, flags, _)
                torch.cuda.empty_cache()

    def validate_workflow(self, batImageGenVals, flags, ite):

        accuracies = []
        for count, batImageGenVal in enumerate(batImageGenVals):
            accuracy_val = self.test(batImageGenTest=batImageGenVal, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(count), count=count)

            accuracies.append(accuracy_val)

        mean_acc = np.mean(accuracies)

        if mean_acc > self.best_accuracy_val:
            self.best_accuracy_val = mean_acc

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val accuracy:{}\n'.format(ite, self.best_accuracy_val))
            f.close()

            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)

            outfile = os.path.join(flags.model_path, 'best_model.tar')
            if flags.method == 'baseline':
                torch.save((self.feature_extractor_network.state_dict(), self.phi.state_dict()), outfile)
            if flags.method == 'Feature_Critic':
                torch.save((self.feature_extractor_network.state_dict(), self.phi.state_dict(), self.omega.state_dict()), outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None, count=0):
        # args, iterations, words in logs,
        self.feature_extractor_network.eval()
        self.phi.eval()
        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', metatest=False, b_unfold_label=False)

        test_preds = []
        test_labels = []
        for data in batImageGenTest:
            x_data = data[0].cuda().float()
            y_label = data[1].view(len(data[1]), 1)
            y_data = torch.zeros(y_label.shape[0], flags.num_classes).scatter_(1, y_label, 1)
            y_data = y_data.numpy()
            # y_label = data[1].view(len(data[1]), 1).numpy()
            # y_data = OneHotEncoder().fit_transform(y_label).toarray()
            # y_data = torch.zeros(flags.batch_size_metatest, flags.num_classes).scatter_(1, y_label, 1)
            # y_data = y_data.cuda().long()
            classifier_out = self.phi(self.feature_extractor_network(x_data)).data.cpu().numpy()
            test_preds.append(classifier_out)
            test_labels.append(y_data)

        # concatenate the test predictions first
        predictions = np.concatenate(test_preds)
        labels = np.concatenate(test_labels)
        accuracy = compute_accuracy(predictions=predictions, labels=labels)
        print('----------accuracy test of domain----------:', accuracy)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, accuracy:{}'.format(ite, accuracy)), log_path=log_path)

        return accuracy


class Model_Feature_Critic_PACS(ModelBaseline_PACS):
    def __init__(self, flags):
        ModelBaseline_PACS.__init__(self, flags)
        self.init_dg_function(flags)

    def __del__(self):
        print('release source')

    def init_dg_function(self,flags):
        self.dg_function = {'MLP': 1, 'Flatten_FTF': 2}
        self.id_dg = self.dg_function[flags.type]

        if self.id_dg == 1:
            self.omega = Critic_Network_MLP(self.h, self.hh).cuda()
        if self.id_dg == 2:
            self.omega = Critic_Network_Flatten_FTF(self.h, self.hh).cuda()

    def train(self, flags):

        write_log(flags, self.flags_log)   # 输出配置信息
        self.pre_train(flags)    # 路径准备
        self.reinit_network_P(flags)  # 训练更新的参数
        time_start = datetime.datetime.now()
        train_minibatches_iterator = zip(*self.train_loaders)
        train_minibatches_iterator_m = zip(*self.meta_test_loaders)

        for _ in range(flags.iteration_size):
            self.iteration = _
            if _ == 15000:  # 自定义学习率调整
                self.opt_phi = torch.optim.SGD(self.phi.parameters(), lr=flags.lr/100, weight_decay=self.weight_decay,
                                               momentum=0.9)
                self.opt_theta = torch.optim.SGD(self.feature_extractor_network.parameters(), lr=flags.lr/100,
                                                 weight_decay=self.weight_decay, momentum=0.9)
                # self.opt_phi = torch.optim.Adam(self.phi.parameters(), lr=flags.lr/10,
                #                             weight_decay=self.weight_decay)
                # self.opt_theta = torch.optim.Adam(self.feature_extractor_network.parameters(), lr=flags.lr/10,
                #                                   weight_decay=self.weight_decay)

            self.feature_extractor_network.train()    # 特征提取
            self.phi.train()                          # 分类器

            index = np.random.permutation(self.num_train_domain)
            meta_train_idx = index[0:(self.num_domain-1)]
            meta_test_idx = index[(self.num_domain-1):]
            # meta_train_idx = index[0:]
            # meta_test_idx = index[0:]
            # meta_train_idx = index[0:5]
            # meta_test_idx = index[5:]
            write_log('-----------------iteration_%d--------------'%(_), self.flags_log)
            write_log(meta_train_idx, self.flags_log)
            write_log(meta_test_idx, self.flags_log)

            for itr in range(flags.meta_iteration_size):
                meta_train_loss_main = 0.0
                meta_train_loss_dg = 0.0
                meta_loss_held_out = 0.0

                for i in meta_train_idx:
                    minibatches_device = [data for data in next(train_minibatches_iterator)]
                    domain_a_x, domain_a_y = minibatches_device[i][0], minibatches_device[i][1]  # .get_images_labels_batch()   # 类似于无限数据生成器
                    x_subset_a = domain_a_x.cuda().float()
                    y_subset_a = domain_a_y.cuda().long()

                    feat_a = self.feature_extractor_network(x_subset_a)   # 提取特征
                    pred_a = self.phi(feat_a)   # 预测类别

                    loss_main = self.ce_loss(pred_a, y_subset_a)  # 对训练数据，计算ce_loss eq.1
                    meta_train_loss_main += loss_main    # 特征提取器、分类的loss
                    if self.id_dg == 1:
                        loss_dg = self.beta * self.omega(feat_a)  # aux_loss eq.6/7
                    if self.id_dg == 2:
                        loss_dg = self.beta * self.omega(torch.matmul(torch.transpose(feat_a, 0, 1), feat_a).view(1, -1))

                    meta_train_loss_dg += loss_dg  # 辅助loss

                self.opt_phi.zero_grad()
                self.opt_theta.zero_grad()
                meta_train_loss_main.backward(retain_graph=True)

                grad_theta = {j: theta_i.grad for (j, theta_i) in self.feature_extractor_network.named_parameters()}   # 记录网络参数的梯度
                theta_updated_old = {}   # 对应算法theta(old)的计算，相当于自己实现了opt.step()
                '''
                for (k, v), g in zip(self.feature_extractor_network.state_dict().items(),grad_theta):
                    theta_updated[k] = v - self.alpha * g
                '''
                # Todo: fix the new running_mean and running_var
                # Because Resnet18 network contains BatchNorm structure, there is no gradient in BatchNorm with running_mean and running_var.
                # Therefore, these two factors should be avoided in the calculation process of theta_old and theta_new.
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):
                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_old[k] = v
                        else:
                            theta_updated_old[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                meta_train_loss_dg.backward(create_graph=True)

                grad_theta = {m: theta_i.grad for (m, theta_i) in self.feature_extractor_network.named_parameters()}
                theta_updated_new = {}   # 对应算法theta(old)的计算
                # num_grad = 0
                for i, (k, v) in enumerate(self.feature_extractor_network.state_dict().items()):

                    if k in grad_theta:
                        if grad_theta[k] is None:
                            # num_grad +=1
                            theta_updated_new[k] = v
                        else:
                            theta_updated_new[k] = v - self.alpha * grad_theta[k]
                            # num_grad += 1

                temp_new_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained, in_channel=1).cuda()
                fix_nn(temp_new_feature_extractor_network, theta_updated_new)   # 把更新后的new网络参数保存下来
                temp_new_feature_extractor_network.train()

                temp_old_feature_extractor_network = getattr(mymodels, flags.model_name)(pretrained=flags.pretrained, in_channel=1).cuda()
                fix_nn(temp_old_feature_extractor_network, theta_updated_old)
                # temp_old_feature_extractor_network.load_state_dict(theta_updated_old)   # 常规网络，故不用自编函数保存参数
                temp_old_feature_extractor_network.train()

                for i in meta_test_idx:
                    minibatches_device_m = [data_m for data_m in next(train_minibatches_iterator_m)]
                    domain_b_x, domain_b_y = minibatches_device_m[i][0], minibatches_device_m[i][1]
                    x_subset_b = domain_b_x.cuda().float()
                    y_subset_b = domain_b_y.cuda().long()

                    feat_b_old = temp_old_feature_extractor_network(x_subset_b).detach()
                    feat_b_new = temp_new_feature_extractor_network(x_subset_b).detach()

                    cls_b_old = self.phi(feat_b_old)
                    cls_b_new = self.phi(feat_b_new)

                    loss_main_old = self.ce_loss(cls_b_old, y_subset_b)
                    loss_main_new = self.ce_loss(cls_b_new, y_subset_b)
                    reward = loss_main_old - loss_main_new   # 新、旧ce_loss的差
                    # calculate the updating rule of omega, here is the max function of h.
                    utility = torch.tanh(reward)   # eq.5
                    # so, here is the min value transfering to the backpropogation.
                    loss_held_out = - utility.sum()  # eq.4
                    meta_loss_held_out += loss_held_out*self.heldout_p  # meta-test loss，用于更新feature-critic

                self.opt_theta.step()
                self.opt_phi.step()

                self.opt_omega.zero_grad()
                meta_loss_held_out.backward()
                self.opt_omega.step()
                torch.cuda.empty_cache()   # 释放GPU显存空间

                print('episode %d' % (_), meta_train_loss_main.data.cpu().numpy(),
                          meta_train_loss_dg.data.cpu().numpy(),
                          meta_loss_held_out.data.cpu().numpy(),
                          )

            if _ % 50 == 0 and _ != 0:
                time_end = datetime.datetime.now()
                epoch = (flags.iteration_size - int(_)) % 500
                time_cost = epoch * (time_end - time_start).seconds / 60    # 转化为分钟
                time_start = time_end
                torch.cuda.empty_cache()
                print('the number of iteration %d, and it is expected to take another %d minutes to complete..' % (_, time_cost))
                torch.cuda.empty_cache()
                self.validate_workflow(self.eval_loaders, flags, _)
                torch.cuda.empty_cache()

    def pre_train(self, flags):
        model_path = os.path.join(flags.load_path, 'best_model.tar')
        if os.path.exists(model_path):
            self.load_state_dict(state_dict=model_path)

    def reinit_network_P(self,flags):
        self.beta = flags.beta
        self.alpha = flags.lr
        self.eta = flags.lr
        self.omega_para = flags.omega
        self.heldout_p = flags.heldout_p
        self.opt_omega = torch.optim.SGD(self.omega.parameters(), lr=self.omega_para, weight_decay=self.weight_decay, momentum=0.9)
        # self.opt_omega = torch.optim.Adam(self.omega.parameters(), lr=self.omega_para,
        #                                   weight_decay=self.weight_decay)
