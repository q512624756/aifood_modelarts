import PIL.Image as Image
import os
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms


class DatasetLoader(data.Dataset):
    def __init__(self, train=True, label=None, path=None, train_transform=None, test_transform=None):
        self.train = train
        self.train_transform = train_transform
        self.test_transform = test_transform
        with open(label) as label_txt:
            lines = label_txt.readlines()
            self.img_name = [os.path.join(path, line.split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[1]) for line in lines]

    def __getitem__(self, item):
        img = Image.open(self.img_name[item])
        if self.train is True:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        label = self.img_label[item]
        return img, label

    def __len__(self):
        return len(self.img_name)


def dataprocess(train_label_path=None, data_dirtory=None, test_label_path=None,batch_size=None):
    train_transform = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    train_data = DatasetLoader(train=True, label=train_label_path,
                               path=data_dirtory,
                               train_transform=train_transform,
                               test_transform=test_transform)
    test_data = DatasetLoader(train=False, label=test_label_path,
                              path=data_dirtory,
                              train_transform=train_transform,
                              test_transform=test_transform)
    train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=4)
    test_iterator = data.DataLoader(test_data, shuffle=False, batch_size=batch_size)
    return train_iterator, test_iterator
