#!/bin/python
# https://drive.google.com/drive/folders/1bCLTcvi6siddiHClBgJn0aRlBx6eZyI8

from alive_progress import alive_bar

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision as tv
from torchvision.transforms import ToTensor
import torchvision.transforms.v2 as T
from torchvision.io import read_image, ImageReadMode


device = 'cpu'
torch.set_default_device(device)
torch.manual_seed(123)
np.random.seed(123)

class ImgDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, augment=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, skipinitialspace=True, dtype={'id': 'string', 'class': 'int8'})
        self.img_dir = img_dir
        self.transform = transform
        self.augment = augment
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.png')
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        label = torch.tensor(self.img_labels.iloc[idx, 1], device=device, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.augment:
            image = self.augment(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

imgtrans = nn.Sequential(
        T.ToDtype(torch.float32, scale=False),
        T.Lambda(lambda t: (t / 127.5) - 1.0),
        T.Normalize([0.0], [1.0]))

augment = torch.nn.Sequential(
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=6.0, translate=(0.1, 0.1), scale=(0.90, 1.0), fill=-1.0),
    T.Normalize([0.0], [1.0]))

dpath = '/home/arch/.datasets/brain'
trainds = ImgDataset(f'{dpath}/train_labels.txt', f'{dpath}/data', imgtrans, augment)
traindl = DataLoader(trainds, batch_size=64, shuffle=True, num_workers=8)

valds = ImgDataset(f'{dpath}/validation_labels.txt', f'{dpath}/data', imgtrans, augment)
valdl = DataLoader(valds, batch_size=64, shuffle=True, num_workers=8)

testds = ImgDataset(f'{dpath}/sample_submission.txt', f'{dpath}/data', imgtrans)
testdl = DataLoader(testds, batch_size=64, shuffle=True, num_workers=8)
'''
train_features, train_labels = next(iter(traindl))
img = train_features[0]
label = train_labels[0]
fig = plt.figure()
columns = 3
rows = 3
fig.add_subplot(rows, columns, 1)
plt.imshow(img.squeeze(), cmap="gray", vmin=-1.0, vmax=1.0)
for i in range(2, columns * rows +1):
    fig.add_subplot(rows, columns, i)
    cimg = augment(img)
    plt.imshow(cimg.squeeze(), cmap="gray", vmin=-1.0, vmax=1.0)
plt.show()
'''

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = F.relu
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.Linear(8 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        aconv = self.activation(self.conv1(x))
        x = self.pool(aconv)
        aconv = self.activation(self.conv2(x))
        x = self.pool(aconv)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.flatten(F.sigmoid(self.fc3(x)))
        return x

def train():
    net = Net()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = 10
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f'\nNew epoch: {epoch}')
        def train(bar=None):
            tot = 0
            correct = 0
            for i, data in enumerate(traindl, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                tot += len(outputs)
                for idx, o in enumerate(outputs):
                    if (o >= 0.5) == (labels[idx] >= 0.5):
                        correct += 1

                if bar != None:
                    bar.text(f'acc {(correct / tot) * 100:.2f}')
                    bar()
            print(f'Train acc {(correct / tot) * 100:.2f}')

        def val(bar=None):
            tot = 0
            correct = 0
            for i, data in enumerate(valdl, 0):
                inputs, labels = data
                outputs = net(inputs)

                if bar != None:
                    tot += len(outputs)
                    for idx, o in enumerate(outputs):
                        if (o >= 0.5) == (labels[idx] >= 0.5):
                            correct += 1
                    bar.text(f'acc {(correct / tot) * 100:.2f}')
                    bar()
            print(f'Eval acc {(correct / tot) * 100:.2f}')

        if False:
            train()
            val()
        else:
            with alive_bar(len(traindl), title='Train', max_cols=24) as bar:
                train(bar)
            with alive_bar(len(valdl), title='Eval', max_cols=24) as bar:
                val(bar)

    torch.save(net.state_dict(), "model.pth")
    return net

def get():
    net = Net().to(device)
    net.load_state_dict(torch.load("model.pth"))
    return net

def test(net):
    with alive_bar(len(valdl), title='Test', max_cols=24) as bar:
        tot = 0
        correct = 0
        o1 = 0
        o2 = 0
        t1 = 0
        t2 = 0
        for i, data in enumerate(valdl, 0):
            inputs, labels = data
            outputs = net(inputs)

            tot += len(outputs)
            for idx, o in enumerate(outputs):
                if (labels[idx] >= 0.5):
                    t1 += 1
                else:
                    t2 += 1
                if (o >= 0.5) == (labels[idx] >= 0.5):
                    if (o >= 0.5):
                        o1 += 1
                    else:
                        o2 += 1
                    # print(f'{o} vs {labels[idx]}')
                    # correct += 1

            if bar != None:
                bar.text(f'{(o1 / t1) * 100:.2f} {(o2 / t2) * 100:.2f}')
                bar()
        print(f'{(o1 / t1) * 100:.2f} {(o2 / t2) * 100:.2f}')
        # print(f'Test acc {(correct / tot) * 100:.2f}')

net = get()
test(net)

