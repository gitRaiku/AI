#!/bin/python

from alive_progress import alive_bar

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda'

class Treap: # This is not actually a treap
    def __init__(self, file):
        text = pd.read_csv(file + '_samples.txt', delimiter='\t').to_numpy()[:,1].flatten()
        words = [w for t in text for w in t.split(' ')]
        uwords = np.unique(words)
        self.converter = {}
        for idx, w in enumerate(words):
            self.converter[w] = idx

    def c(self, w):
        if w in self.converter:
            return self.converter(w)
        else:
            print(f'Word {w} not found!')
            return len(self.converter + 1)
    def cc(self, seq):
        s = np.zeros(len(seq))
        for idx, e in enumerate(seq):
            s[idx] = c(e)
        return s

class AlientDataset(Dataset):
    def __init__(self, floc, treap, eqn=False):
        text = pd.read_csv(floc + '_samples.txt', delimiter='\t').to_numpy()[:,1]
        ids = [treap.cc(i) for i in text]
        labels = pd.read_csv(floc + '_labels.txt', delimiter='\t').to_numpy()[:,1]
        self.data = np.stack((ids, labels), axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def getClass(self, cnum):
        idx = np.where(self.data[:,1] == cnum)
        return self.data[idx]

class Alienter(nn.Module):
    def __init__(self, dembed, dhidden, vocabsize, tagsize):
        super(Alienter, self).__init__()
        self.dhidden = dhidden
        self.word_embeddings = nn.Embedding(vocabsize, dembed)
        self.lstm = nn.LSTM(dembed, dhidden)
        self.hidden2tag = nn.Linear(dhidden, tagsize)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def compAcc(outputs, labels):
    r = np.zeros((3, 3), dtype=np.int32)
    for i in len(outputs):
        co = outputs[i]
        cp = np.argmax(co)
        cl = labels[i]
        r[cl][cp] += 1

def getStats(s):
    return 'NO'
    # return f'acc: {acc:.2f} f1: {f1:.2f} {tp} {fp} {tn} {fn}'


def train(trainds):
    embed_dim = 10
    hidden_dim = 10
    vocabsize = 10
    tagsize = 3
    model = Alienter(embed_dim, hidden_dim, vocabsize, tagsize).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
    optimizer = optim.Adam(net.parameters())

    epochs = 10
    for ie in range(epochs):
        print(f'\nNew epoch: {epoch}')
        def tr(bar=None):
            print('Start train')
            stats = np.zeros(4, dtype=np.int32)
            for i, data in enumerate(traindl, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                stats = np.add(stats, compAcc(outputs, labels))

                if bar != None:
                    bar.text(getStats(stats))
                    bar()
            print(f'Train {getStats(stats)}')
        


if __name__ == '__main__':
    torch.set_default_device(device)
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.manual_seed(123)
    np.random.seed(123)

    dpath = '/home/arch/.datasets/butant/'
    treap = Treap(dpath + 'train')

    bsize = 64
    trainds = AlientDataset(dpath + 'train', treap)
    traindl = DataLoader(trainds, batch_size=bsize, shuffle=True, num_workers=0, generator=torch.Generator(device=device))

    prepare()

    net = train(trainds)
    test(net)


