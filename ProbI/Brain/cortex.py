#!/bin/python
# https://drive.google.com/drive/folders/1bCLTcvi6siddiHClBgJn0aRlBx6eZyI8

from alive_progress import alive_bar

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import adjust_contrast
import torchvision.transforms.v2 as T
from torchvision.io import read_image, ImageReadMode

device = 'cuda'

class ImgDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, augment=None, target_transform=None, eqn=False):
        self.img_labels = pd.read_csv(annotations_file, skipinitialspace=True, dtype={'id': 'string', 'class': 'int8'}).to_numpy()
        if eqn == True:
            tr = []
            fl = []
            for x in self.img_labels:
                if x[1] == 1:
                    tr.append(x)
                else:
                    fl.append(x)

            print(len(tr))
            print(len(fl))
            self.img_labels = np.append(fl, [tr, tr, tr])
            self.img_labels = self.img_labels.reshape(len(self.img_labels) // 2, 2)

        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.target_transform = target_transform

    def weights(self):
        return compute_class_weight(class_weight='balanced', classes=np.unique(self.img_labels[:,1]), y=self.img_labels[:,1])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels[idx][0]}.png')
        image = read_image(img_path, mode=ImageReadMode.GRAY).to(device)
        ct = torch.tensor(self.img_labels[idx][1], dtype=torch.int64)
        label = F.one_hot(ct, num_classes=2).type(torch.float32)
        # label = torch.tensor(self.img_labels[idx][1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        image = adjust_contrast(image, 1.8)

        if self.augment:
            image = self.augment(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def rescal(t):
    return t / 255.0
    # return t / 127.5 - 1.0

imgtrans = nn.Sequential(
        T.ToDtype(torch.float32, scale=False),
        T.Lambda(rescal),
        T.Normalize([0.0], [1.0]))

augment = torch.nn.Sequential(
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=6.0, translate=(0.1, 0.1), scale=(0.90, 1.0), fill=-1.0),
    T.Normalize([0.0], [1.0]))

bsize = 64
dpath = '/home/arch/.datasets/brain'
trainds = ImgDataset(f'{dpath}/train_labels.txt', f'{dpath}/data', imgtrans, augment, eqn=True)
traindl = DataLoader(trainds, batch_size=bsize, shuffle=True, num_workers=0, generator=torch.Generator(device=device))

valds = ImgDataset(f'{dpath}/validation_labels.txt', f'{dpath}/data', imgtrans, augment)
valdl = DataLoader(valds, batch_size=bsize, shuffle=True, num_workers=0, generator=torch.Generator(device=device))


def imshow():
    train_features, train_labels = next(iter(traindl))
    img = train_features[0]
    label = train_labels[0]
    fig = plt.figure()
    columns = 3
    rows = 3
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img.squeeze(), cmap="gray", vmin=-0.0, vmax=1.0)
    for i in range(2, columns * rows +1):
        fig.add_subplot(rows, columns, i)
        # cimg = augment(img)
        # cimg = adjust_contrast(img, 1.5)
        cimg = img
        plt.imshow(cimg.squeeze(), cmap="gray", vmin=-0.0, vmax=1.0)
    plt.show()

# imshow()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = F.relu
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.conv3 = nn.Conv2d(8, 16, 5)
        self.apool = nn.AdaptiveAvgPool2d(5)
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        # self.fc1 = nn.Linear(16 * 24 * 24, 256)
        self.fc2 = nn.Linear(32, 10)
        self.fc3 = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = self.pool(self.activation(self.conv3(x)))
        x = self.apool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.dropout(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def compAcc(outputs, labels):
    r = np.zeros((4), dtype=np.int32)
    for idx, o in enumerate(outputs):
        r[0 if labels[idx][1] >= 0.5 else 2] += 1
        if torch.argmax(o) == (1 if (labels[idx][1] >= 0.5) else 0):
            r[1 if torch.argmax(o) == 1 else 3] += 1
    return r


def compAcc(outputs, labels):
    r = np.zeros((4), dtype=np.int32)
    for idx, o in enumerate(outputs):
        r[0 if labels[idx][1] >= 0.5 else 2] += 1
        if torch.argmax(o) == (1 if (labels[idx][1] >= 0.5) else 0):
            r[1 if torch.argmax(o) == 1 else 3] += 1
    return r

def getStats(s):
    tp = s[1]
    tn = s[3]
    fp = s[0] - tp
    fn = s[2] - tn
    tot = s[0] + s[2]
    acc = (tp + tn) / tot * 100
    p = tp / (tp + fp)
    if tp + fn == 0:
        tp += 1
    r = tp / (tp + fn)
    if p + r == 0:
        r = 1
    f1 = 2 * p * r / (p + r)
    return f'acc: {acc:.2f} f1: {f1:.2f} {tp} {fp} {tn} {fn}'

'''
def compAcc(outputs, labels):
    r = np.zeros((4), dtype=np.int32)
    for idx, o in enumerate(outputs):
        r[0 if labels[idx] >= 0.5 else 2] += 1
        if (o >= 0.5) == (labels[idx] >= 0.5):
            r[1 if o >= 0.5 else 3] += 1
    return r
'''

def getStats(s):
    try:
        tp = s[1]
        tn = s[3]
        fp = s[0] - tp
        fn = s[2] - tn
        tot = s[0] + s[2]
        acc = (tp + tn) / tot * 100
        p = tp / (tp + fp)
        if tp + fn == 0:
            r = 1
        else:
            r = tp / (tp + fn)
        if p + r == 0:
            r = 1
        f1 = 2 * p * r / (p + r)
        return f'acc: {acc:.2f} f1: {f1:.2f} {tp} {fp} {tn} {fn}'
    except:
        return ''


def train():
    net = Net().to(device)
    # net.load_state_dict(torch.load("model.pth"))
    # 2238 12762

    weights = trainds.weights()
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor(weights))
    # criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(net.parameters())

    epochs = 10
    for epoch in range(epochs):
        print(f'\nNew epoch: {epoch}')
        def train(bar=None):
            print('Start train')
            stats = np.zeros(4, dtype=np.int32)
            for i, data in enumerate(traindl, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                # o = torch.zeros([64, 2])
                # l = torch.zeros([64, 2])
                print(labels)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                stats = np.add(stats, compAcc(outputs, labels))

                if bar != None:
                    bar.text(getStats(stats))
                    bar()
            print(f'Train {getStats(stats)}')

        def val(bar=None):
            stats = np.zeros(4, dtype=np.int32)
            with torch.no_grad():
                for i, data in enumerate(valdl, 0):
                    inputs, labels = data
                    outputs = net(inputs)

                    stats = np.add(stats, compAcc(outputs, labels))

                    if bar != None:
                        bar.text(getStats(stats))
                        bar()
            print(f'Eval {getStats(stats)}')

        try:
            if False:
                train()
                val()
            else:
                with alive_bar(len(traindl), title='Train', max_cols=15) as bar:
                    train(bar)
                with alive_bar(len(valdl), title='Eval', max_cols=15) as bar:
                    val(bar)
        except KeyboardInterrupt:
            print('Saving model to unfinished.pth')
            torch.save(net.state_dict(), f"unfinished.pth")
            exit()
            
        print(f'Saving model to model-{epoch}.pth')
        torch.save(net.state_dict(), f"model-{epoch}.pth")

    return net

def get():
    net = Net().to(device)
    net.load_state_dict(torch.load("model.pth"))
    return net

def test(net):
    with alive_bar(len(valdl), title='Test', max_cols=15) as bar:
        stats = np.zeros(4, dtype=np.int32)
        for i, data in enumerate(valdl, 0):
            inputs, labels = data
            outputs = net(inputs)

            stats = np.add(stats, compAcc(outputs, labels))

            if bar != None:
                bar.text(getStats(stats))
                bar()
        print(f'Test {getStats(stats)}')

if __name__ == '__main__':
    torch.set_default_device(device)
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.manual_seed(123)
    np.random.seed(123)

    net = train()
    test(net)
