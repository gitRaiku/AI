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

class AlientDataset(Dataset):
    def __init__(self, floc, eqn=False):
        text = pd.read_csv(floc + '_samples.txt', delimiter='\t').to_numpy()[:,1]
        labels = pd.read_csv(floc + '_labels.txt', delimiter='\t').to_numpy()[:,1]
        self.data = np.stack((text, labels), axis=1)
        # print(self.data)

    # def weights(self):
        # return compute_class_weight(class_weight='balanced', classes=np.unique(self.img_labels[:,1]), y=self.img_labels[:,1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def getClass(self, cnum):
        idx = np.where(self.data[:,1] == cnum)
        return self.data[idx]

dpath = '/home/arch/.datasets/butant/'
trainds = AlientDataset(dpath + 'train')

def ad(d, k):
    try:
        d[k] += 1
    except:
        d[k] = 1

def srt(d):
    sd = sorted(d, key=d.__getitem__, reverse=True)
    r = []
    for i in sd:
        r.append([i, d[i]])
    return np.array(r)

def analyze(text):
    l = len(text)
    chars = {}
    words = {}
    wlen = {}
    for t in text:
        for c in t:
            if c == ' ':
                continue
            ad(chars, c)

        for word in t.split(' '):
            ad(wlen, len(word))
            ad(words, word)
            

    # print(l)
    schars = srt(chars)
    swords = srt(words)
    swlen = srt(wlen)
    return schars, swords, swlen

c1 = trainds.getClass(1)[:,0]
c2 = trainds.getClass(2)[:,0]
c3 = trainds.getClass(3)[:,0]
with open('c1', 'wb') as f:
    s = pickle.dumps(c1)
    f.write(s)
with open('c2', 'wb') as f:
    s = pickle.dumps(c2)
    f.write(s)
with open('c3', 'wb') as f:
    s = pickle.dumps(c3)
    f.write(s)

with open('c1', 'w') as f:
    c1 = pickle.loads(f)
with open('c2', 'w') as f:
    c2 = pickle.loads(f)
with open('c3', 'w') as f:
    c3 = pickle.loads(f)

w1 = analyze(c1)
w2 = analyze(c2)
w3 = analyze(c3)

def gal(c1, c2, c3):
    common_alphabet = np.intersect1d(np.intersect1d(c1[0], c2[0]), c3[0])
    full_alphabet = np.union1d(np.union1d(c1[0], c2[0]), c3[0])
    unique = np.setdiff1d(full_alphabet, c1[0])
    print(c1[np.where(c1[:,0] in unique)])


alphabet = gal(w1, w2, w3)
'''
cwords = 0
cw12 = {}
for w in w1:
    if w in w2:
        ad(cw12, w)
cw123 = {}
for w in cw12:
    if w in w3:
        ad(cw123, w)
'''

# valds = ImgDataset(dpath + 'validation')
