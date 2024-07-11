#!/bin/python

# I want to fucking kill myself
# Like fr fr on god no cap

from alive_progress import alive_bar

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
'''

device = 'cuda'

dpath = '/home/arch/.datasets/butant/'

floc = dpath + 'train'
text = pd.read_csv(floc + '_samples.txt', delimiter='\t').to_numpy()[:,1]
labels = pd.read_csv(floc + '_labels.txt', delimiter='\t').to_numpy()[:,1] - 1

floc = dpath + 'validation'
valtext = pd.read_csv(floc + '_samples.txt', delimiter='\t').to_numpy()[:,1]
vallabels = pd.read_csv(floc + '_labels.txt', delimiter='\t').to_numpy()[:,1] - 1

vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
X = vectorizer.fit_transform(text)
print(X)



'''
class Treap: # This is not actually a treap
    def __init__(self, file):
        text = pd.read_csv(file + '_samples.txt', delimiter='\t').to_numpy()[:,1].flatten()
        words = [w for t in text for w in t]
        uwords = np.unique(words)
        self.converter = {}
        for idx, w in enumerate(uwords):
            self.converter[w] = idx + 20
        print(f'wl {len(words)} : uw {len(uwords)} : conv {len(self.converter)}')
        self.unknown = 0

    def c(self, w):
        if w in self.converter:
            return self.converter[w]
        else:
            self.unknown += 1
            # print(f'Unkown {w}')
            return 0

    def cc(self, seq):
        s = np.zeros(len(seq), dtype=np.int32)
        for idx, e in enumerate(seq):
            s[idx] = self.c(e)
        return s

class AlientDataset(Dataset):
    def __init__(self, floc, treap, eqn=False):
        text = pd.read_csv(floc + '_samples.txt', delimiter='\t').to_numpy()[:,1]
        self.ids = [treap.cc(i) for i in text]
        self.labels = pd.read_csv(floc + '_labels.txt', delimiter='\t').to_numpy()[:,1] - 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], torch.tensor([self.labels[idx] for _ in range(len(self.ids[idx]))], dtype=torch.int64)

class Alienter(nn.Module):
    def __init__(self, dembed, dhidden, vocabsize, tagsize):
        super(Alienter, self).__init__()
        self.dhidden = dhidden
        self.word_embeddings = nn.Embedding(vocabsize, dembed)
        self.lstm = nn.LSTM(dembed, dhidden)
        self.hidden2tag = nn.Linear(dhidden, tagsize)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tags = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tags, dim=-1)
        return tag_scores

def compAcc(output, label):
    r = np.zeros((3, 3), dtype=np.int32)
    for idx, o in enumerate(output[0]):
        r[label[0][idx]][torch.argmax(o)] += 1
    return r

def getStats(s):
    t1 = np.sum(s[0])
    c1 = s[0][0]
    t2 = np.sum(s[1])
    c2 = s[1][1]
    t3 = np.sum(s[2])
    c3 = s[2][2]
    acc = (c1 + c2 + c3) / (t1 + t2 + t3)
    return f'acc: {acc:.2f} {s[0][0]} {s[0][1]} {s[0][2]} | {s[1][0]} {s[1][1]} {s[1][2]} | {s[2][0]} {s[2][1]} {s[2][2]} '


def train(traindl, valdl, treap, vocabsize):
    print(f'Vocabsize {vocabsize}')
    embed_dim = 5
    hidden_dim = 10
    tagsize = 3
    model = Alienter(embed_dim, hidden_dim, vocabsize, tagsize).to(device)

    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 20, 20], dtype=torch.float32))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    epochs = 10
    for epoch in range(epochs):
        print(f'\nNew epoch: {epoch}')
        with torch.no_grad():
            for c in treap.converter:
                print(model.word_embeddings(torch.tensor(treap.converter[c])))
                break

        def train(bar=None):
            print('Start train')
            stats = np.zeros((3, 3), dtype=np.int32)
            for i, data in enumerate(traindl, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                # print(outputs.shape)
                # print(labels.shape)
                loss = criterion(outputs.view([len(outputs[0]), 3]), labels.view([len(outputs[0])]))
                loss.backward()
                optimizer.step()

                # stats[labels][torch.argmax(outputs)] += 1
                # stats += compAcc(outputs, labels)
                # print(stats)

                if bar != None:
                    bar.text(f'loss: {loss:.2f} {getStats(stats)}')
                    bar()
            print(f'Train {getStats(stats)}')

        def val(bar=None):
            stats = np.zeros((3, 3), dtype=np.int32)
            with torch.no_grad():
                for i, data in enumerate(valdl, 0):
                    inputs, labels = data
                    outputs = model(inputs)

                    # stats = np.add(stats, compAcc(outputs, labels))
                    # stats[labels][torch.argmax(outputs)] += 1

                    if bar != None:
                        bar.text(getStats(stats))
                        bar()
            print(f'Eval {getStats(stats)}')

        try:
            if True:
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

    bsize = 1
    trainds = AlientDataset(dpath + 'train', treap)
    traindl = DataLoader(trainds, batch_size=bsize, shuffle=True, num_workers=0, generator=torch.Generator(device=device))

    valds = AlientDataset(dpath + 'validation', treap)
    valdl = DataLoader(valds, batch_size=bsize, shuffle=False, num_workers=0, generator=torch.Generator(device=device))
    print(f'Unknown words {treap.unknown}')

    train(traindl, valdl, treap, len(treap.converter) + 21)
'''

