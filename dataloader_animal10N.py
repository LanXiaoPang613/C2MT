from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch
import os
import matplotlib

def unpickle(file):
    fo = open(file, 'rb').read()
    size = 64 * 64 * 3 + 1
    for i in range(50000):
        arr = np.fromstring(fo[i * size:(i + 1) * size], dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3, 64, 64)).transpose((1, 2, 0))
    return img, lab

class animal_dataset(Dataset):
    def __init__(self, root, transform, mode, pred=[], path=[], probability=[], num_class=10):

        self.root = root
        self.transform = transform
        self.mode = mode

        self.train_dir = root + '/training/'
        self.test_dir = root + '/testing/'
        train_imgs = os.listdir(self.train_dir)
        test_imgs = os.listdir(self.test_dir)
        self.test_data = []
        self.test_labels = []
        noise_file1 = './training_batch.json'
        noise_file2 = './testing_batch.json'
        if mode == 'test':
            if os.path.exists(noise_file2):
                dict = json.load(open(noise_file2, "r"))
                self.test_labels = dict['data']
                self.test_data = dict['label']
            else:
                for img in test_imgs:
                    self.test_data.append(self.test_dir+img)
                    self.test_labels.append(int(img[0]))
                dicts = {}
                dicts['data'] = self.test_data
                dicts['label'] = self.test_labels
                # json.dump(dicts, open(noise_file2, "w"))
        else:
            if os.path.exists(noise_file1):
                dict = json.load(open(noise_file1, "r"))
                train_data = dict['data']
                train_labels = dict['label']
            else:
                train_data = []
                train_labels = {}
                for img in train_imgs:
                    img_path = self.train_dir+img
                    train_data.append(img_path)
                    train_labels[img_path] = (int(img[0]))
                dicts = {}
                dicts['data'] = train_data
                dicts['label'] = train_labels
                # json.dump(dicts, open(noise_file1, "w"))
            if self.mode == "all":
                self.train_data = train_data
                self.train_labels = train_labels
            elif self.mode == "labeled":
                pred_idx = pred.nonzero()[0]
                train_img = path
                self.train_data = [train_img[i] for i in pred_idx]
                self.probability = probability[pred_idx]
                # self.train_labels = train_labels[pred_idx]
                self.train_labels = train_labels
                print("%s data has a size of %d" % (self.mode, len(self.train_data)))
            elif self.mode == "unlabeled":
                pred_idx = (1 - pred).nonzero()[0]
                train_img = path
                self.train_data = [train_img[i] for i in pred_idx]
                self.probability = probability[pred_idx]
                # self.train_labels = train_labels[pred_idx]
                print("%s data has a size of %d" % (self.mode, len(self.train_data)))
                self.train_labels = train_labels

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_data[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, target, prob
        elif self.mode == 'unlabeled':
            img_path = self.train_data[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2
        elif self.mode == 'all':
            img_path = self.train_data[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target,img_path
        elif self.mode == 'test':
            img_path = self.test_data[index]
            target = self.test_labels[index]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.test_data)
        else:
            return len(self.train_data)


class animal_dataloader():
    def __init__(self, root='E:/2_Dataset_All/Animal-10N', batch_size=32, num_workers=0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root

        self.transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        self.transform_test = transforms.Compose([
            # transforms.Resize(64),
            # transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])

    def run(self, mode, pred=[], prob=[], paths=[]):
        if mode == 'warmup':
            warmup_dataset = animal_dataset(self.root, transform=self.transform_train, mode='all')
            warmup_loader = DataLoader(
                dataset=warmup_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return warmup_loader
        elif mode == 'train':
            labeled_dataset = animal_dataset(self.root, transform=self.transform_train, mode='labeled', pred=pred, path=paths,
                                             probability=prob)
            labeled_loader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            unlabeled_dataset = animal_dataset(self.root, transform=self.transform_train, mode='unlabeled', pred=pred,path=paths,
                                               probability=prob)
            unlabeled_loader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=int(self.batch_size),
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_loader, unlabeled_loader
        elif mode == 'eval_train':
            eval_dataset = animal_dataset(self.root, transform=self.transform_test, mode='all')
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader
        elif mode == 'test':
            test_dataset = animal_dataset(self.root, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=1000,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

# if __name__ == '__main__':
#     loader = animal_dataloader()
#     train_loader = loader.run('warmup')
#     import matplotlib.pyplot as plt
#     for batch_idx, (inputs, labels, idx, img_path) in enumerate(train_loader):
#         print(img_path[0])
#         plt.figure(dpi=300)
#         # plt.imshow(inputs[0])
#         plt.imshow(inputs[0].reshape(64, 64, 3))
#         plt.show()
#         plt.close()
#         print(inputs.shape())
#         print(idx)
#         print(labels, len(labels))
#     # print(train_loader.dataset.__len__())