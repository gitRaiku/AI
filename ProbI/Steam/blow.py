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

class ValveDataset(Dataset):
    def __init__(self, floc, eqn=False):
        self.data = pd.read_csv(floc).to_numpy()
        data = self.data

        print(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    torch.set_default_device(device)
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.manual_seed(123)
    np.random.seed(123)

    file = '/home/arch/.datasets/games/games.csv'
    ds = ValveDataset(file)

    trainds, valds = torch.utils.data.random_split(ds, [4 * (len(ds) // 5), len(ds) - 4 * (len(ds) // 5)])
