#!/bin/python

import torch

from torch import nn
from torch.utils.data import DataLoader

file_name = '/home/raiku/.datasets/IMDB.csv'
df = pd.read_csv(file_name)
df.head()

