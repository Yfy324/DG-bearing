# coding=utf-8
from datasets.sequence_aug import *


class DataGenerate(object):
    num_classes = 3
    inputchannel = 1

    def __init__(self, args, domain_data, labels=None,
                 transform=None, target_transform=None,
                 indices=None):
        self.domain_num = 0
        self.labels = np.array(labels)
        self.x = domain_data
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(labels))
        else:
            self.indices = indices
        self.normlizetype = args.normlizetype
        self.transforms = Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                # Retype(),
            ])
        self.target_transforms = Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
            ])

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img = self.input_trans(self.x[index])
        ctarget = self.target_trans(self.labels[index])
        return img, ctarget

    def __len__(self):
        return len(self.indices)
