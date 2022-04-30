from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils.FCutils import unfold_label, shuffle_data

transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.75),
    # transforms.RandomCrop(224, padding=4),
    # transforms.CenterCrop(224),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    # transforms.CenterCrop(224),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_domain_name():
    return {'0': 'photo', '1': 'art_painting', '2': 'cartoon', '3': 'sketch'}


def get_data_folder():
    data_folder = '/home/yfy/Desktop/Dataset/PACS/'
    train_data = ['/home/yfy/Desktop/Dataset/PACS/pacs_label/photo_train_kfold.txt',
                  '/home/yfy/Desktop/Dataset/PACS/pacs_label/art_painting_train_kfold.txt',
                  '/home/yfy/Desktop/Dataset/PACS/pacs_label/cartoon_train_kfold.txt',
                  '/home/yfy/Desktop/Dataset/PACS/pacs_label/sketch_train_kfold.txt']

    val_data = ['/home/yfy/Desktop/Dataset/PACS/pacs_label/photo_crossval_kfold.txt',
                '/home/yfy/Desktop/Dataset/PACS/pacs_label/art_painting_crossval_kfold.txt',
                '/home/yfy/Desktop/Dataset/PACS/pacs_label/cartoon_crossval_kfold.txt',
                '/home/yfy/Desktop/Dataset/PACS/pacs_label/sketch_crossval_kfold.txt']

    test_data = ['/home/yfy/Desktop/Dataset/PACS/pacs_label/photo_test_kfold.txt',
                 '/home/yfy/Desktop/Dataset/PACS/pacs_label/art_painting_test_kfold.txt',
                 '/home/yfy/Desktop/Dataset/PACS/pacs_label/cartoon_test_kfold.txt',
                 '/home/yfy/Desktop/Dataset/PACS/pacs_label/sketch_test_kfold.txt']
    return data_folder, train_data, val_data, test_data


class BatchImageGenerator:
    def __init__(self, flags, stage, file_path, metatest, b_unfold_label):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(flags, stage, file_path, metatest)
        self.load_data(b_unfold_label)  # 打乱顺序的样本、标签

    def configuration(self, flags, stage, file_path, metatest):  # meta-train/meta-test 主要就是batch_size的区别
        if metatest == False:
            self.batch_size = flags.batch_size
        if metatest == True:
            self.batch_size = flags.batch_size_metatest
        self.current_index = -1
        self.file_path = file_path
        self.stage = stage
        self.shuffled = False

    def load_data(self, b_unfold_label):
        file_path = self.file_path
        images = []
        labels = []
        with open(file_path, 'r') as file_to_read:  # 读取文件内容，划分data和labels
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                    pass
                image, label = [i for i in lines.split()]
                images.append(image)
                labels.append(int(label) - 1)
                pass
        if b_unfold_label:
            labels = unfold_label(labels=labels, classes=len(np.unique(labels)))
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.file_num_train = len(self.labels)  # 训练样本数

        if self.stage is 'train':
            self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)

    def get_images_labels_batch(self):

        images = []
        labels = []
        for index in range(self.batch_size):
            self.current_index += 1  # 因为是无限迭代的，所以需要保存一下在该域中已经提取了多少张图片/省去了infinite generator
            # void over flow
            if self.current_index > self.file_num_train - 1:  # 已经提取了k轮，需要重新打乱照片
                self.current_index %= self.file_num_train
                self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)
            # img = Image.open('data/PACS/pacs_data/'+self.images[self.current_index])
            img = Image.open('/home/yfy/Desktop/Dataset/PACS/' + self.images[self.current_index])
            img = img.convert('RGB')
            img = transform_train(img)
            img = np.array(img)
            images.append(img)
            labels.append(self.labels[self.current_index])

        return np.array(images), np.array(labels)


def get_image(images):
    images_data = []
    for img in images:
        img = Image.open('/home/yfy/Desktop/Dataset/PACS/' + img)
        img = transform_train(img)
        img = np.array(img)
        images_data.append(img)
    return np.array(images_data)
