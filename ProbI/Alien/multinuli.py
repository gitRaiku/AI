#!/bin/python
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda
https://www.kaggle.com/code/dxtvzw/steam-games-dataset-eda

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.naive_bayes import MultinomialNB

# %matplotlib inline
floc = '/home/arch/.datasets/butant/'
ttext = pd.read_csv(floc + 'train' + '_samples.txt', delimiter='\t').to_numpy()[:,1]
tlabels = pd.read_csv(floc + 'train' + '_labels.txt', delimiter='\t').to_numpy()[:,1] - 1
vtext = pd.read_csv(floc + 'validation' + '_samples.txt', delimiter='\t').to_numpy()[:,1]
vlabels = pd.read_csv(floc + 'validation' + '_labels.txt', delimiter='\t').to_numpy()[:,1] - 1

def eval(ch, mi, ma):
    vec2 = CountVectorizer(analyzer='char' if ch == 1 else 'char_wb', ngram_range=(mi, ma))
    tt = vec2.fit_transform(ttext)
    vt = vec2.transform(vtext)
    clf = MultinomialNB()
    clf.fit(tt, tlabels)
    sc = clf.score(vt, vlabels)
    print(f'{"char" if ch == 1 else "char_wb"} : ({mi},{ma}) : {sc}')

for ch in range(2):
    for mi in range(2, 6):
        for ma in range(mi, 6):
            eval(ch, mi, ma)
