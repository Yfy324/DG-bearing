# coding=utf-8
import torch


class Algorithm(torch.nn.Module):

    def __init__(self, args):
        super(Algorithm, self).__init__()

    def update(self, minibatches, opt, sch):
        raise NotImplementedError  # 在面向对象编程中，可以在父类中先预留一个方法接口不实现，在其子类中实现。如果要求其子类一定要实现，不实现的时候会导致问题，那么采用raise的方式就很好。

    def predict(self, x):
        raise NotImplementedError
