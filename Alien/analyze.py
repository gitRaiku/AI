#!/bin/python

import pickle
import numpy as np
import matplotlib.pyplot as plt

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

def g(s):
    with open(s, 'rb') as f:
        return pickle.loads(f.read())

cc1 = g('c1')
cc2 = g('c2')
cc3 = g('c3')

w1 = analyze(cc1)
w2 = analyze(cc2)
w3 = analyze(cc3)

def gal():
    a1 = w1[0][:,0]
    a2 = w2[0][:,0]
    a3 = w3[0][:,0]

    all_alphabet = np.union1d(np.union1d(a1, a2), a3)
    return all_alphabet
    common_alphabet = np.intersect1d(np.intersect1d(a1, a2), a3)
    a1d = np.setdiff1d(a3, common_alphabet)
    print(a1d)
    for c in a1d:
        for w in w3[0]:
            if w[0] == c:
                print(f'{c} - {w[1]}')
                break
    # print(w1[0][np.where(w1[0][0] == a1d[1])])

alphabet = gal()


def pltw(alpha, words):
    pairs = {}
    turn = {}
    for idx, a in enumerate(alpha):
        turn[a] = idx
    for wi in range(len(words)):
        for i in range(len(words[wi][0]) - 1):
            pair = (words[wi][0][i], words[wi][0][i + 1])
            if pair in pairs:
                pairs[pair] += int(words[wi][1])
            else:
                pairs[pair] = int(words[wi][1])
    mat = np.zeros((len(alpha), len(alpha)))
    for p in pairs:
        mat[turn[p[0]]][turn[p[1]]] += pairs[p]
    return mat[0:100,0:100]

mats = [pltw(alphabet, w1[1]), pltw(alphabet, w2[1]), pltw(alphabet, w3[1])]

fig = plt.figure()
for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    plt.matshow(mats[i], fignum=False)
    # plt.plot([1, 2], [2, 3])
plt.show()

