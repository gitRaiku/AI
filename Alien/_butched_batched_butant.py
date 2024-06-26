#!/bin/python

# I want to fucking kill myself

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
        for idx, w in enumerate(uwords):
            self.converter[w] = idx + 20
        print(f'wl {len(words)} : uw {len(uwords)} : conv {len(self.converter)}')

    def c(self, w):
        if w in self.converter:
            return self.converter[w]
        else:
            print(f'Word {w} not found!')
            return 0

    def cc(self, seq):
        s = np.zeros(2000, dtype=np.int32)
        # s = np.zeros(len(seq), dtype=np.int32)
        for idx, e in enumerate(seq):
            s[idx] = self.c(e)
        return s

class AlientDataset(Dataset):
    def __init__(self, floc, treap, eqn=False):
        text = pd.read_csv(floc + '_samples.txt', delimiter='\t').to_numpy()[:,1]
        self.ids = [treap.cc(i.split(' ')) for i in text]
        self.labels = pd.read_csv(floc + '_labels.txt', delimiter='\t').to_numpy()[:,1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]

def collate_fn(batch):
    # a = 
    words = np.array([np.array(i[0]) for i in batch])
    a = torch.tensor(words, dtype=torch.int32)
    b = torch.tensor(np.array([i[1] for i in batch], dtype=np.int64))
    return [a, b]

class Alienter(nn.Module):
    def __init__(self, dembed, dhidden, vocabsize, tagsize):
        super(Alienter, self).__init__()
        self.dhidden = dhidden
        self.word_embeddings = nn.Embedding(vocabsize, dembed)
        self.lstm = nn.LSTM(dembed, dhidden)
        self.hidden2tag = nn.Linear(dhidden, tagsize)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        pembed = nn.utils.rnn.pack_sequence(embeds)
        lstm_outp, _ = self.lstm(pembed)
        lstm_out = nn.utils.rnn.unpack_sequence(lstm_outp)
        lstm_lastp = [out[-1] for out in lstm_out]
        lstm_pred = torch.cat(lstm_lastp).view([len(lstm_lastp), len(lstm_lastp[0])])
        tags = self.hidden2tag(lstm_pred)
        tag_scores = F.log_softmax(tags, dim=1)
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


def train(traindl, vocabsize):
    print(f'Vocabsize {vocabsize}')
    embed_dim = 10
    hidden_dim = 20
    tagsize = 3
    model = Alienter(embed_dim, hidden_dim, vocabsize, tagsize).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    epochs = 10
    for epoch in range(epochs):
        print(f'\nNew epoch: {epoch}')
        def train(bar=None):
            print('Start train')
            stats = np.zeros((3, 3), dtype=np.int32)
            for i, data in enumerate(traindl, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                stats = np.add(stats, compAcc(outputs, labels))
                print(stats)

                if bar != None:
                    bar.text(getStats(stats))
                    bar()
            print(f'Train {getStats(stats)}')

        def val(bar=None):
            stats = np.zeros(4, dtype=np.int32)
            with torch.no_grad():
                for i, data in enumerate(valdl, 0):
                    inputs, labels = data
                    outputs = model(inputs)

                    stats = np.add(stats, compAcc(outputs, labels))

                    if bar != None:
                        bar.text(getStats(stats))
                        bar()
            print(f'Eval {getStats(stats)}')

        try:
            if False:
                with alive_bar(len(traindl), title='Train', max_cols=15) as bar:
                    train(bar)
                with alive_bar(len(valdl), title='Eval', max_cols=15) as bar:
                    val(bar)
            else:
                train(None)
                val(None)
        except KeyboardInterrupt:
            print('Saving model to unfinished.pth')
            torch.save(model.state_dict(), 'unfinished.pth')
            exit()
        


if __name__ == '__main__':
    torch.set_default_device(device)
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.manual_seed(123)
    np.random.seed(123)

    dpath = '/home/arch/.datasets/butant/'
    treap = Treap(dpath + 'train')

    bsize = 64
    trainds = AlientDataset(dpath + 'train', treap)
    traindl = DataLoader(trainds, batch_size=bsize, collate_fn=collate_fn, shuffle=False, num_workers=0, generator=torch.Generator(device=device))

    train(traindl, len(treap.converter) + 21)


