#!/bin/python

from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt


device = 'cpu'
torch.set_default_device(device)

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 2 # Plus EOS marker
def findFiles(path): return glob.glob(path)
def unicodeToAscii(s): return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' and c in all_letters)
def readLines(filename): 
    with open(filename, encoding='utf-8') as some_file: 
        return [unicodeToAscii(line.strip()) for line in some_file]

category_lines = {}
all_categories = []
for filename in findFiles('/home/arch/.datasets/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h1 = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.h12h = nn.Linear(hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.h12h(self.i2h1(input_combined))
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self): return torch.zeros(1, self.hidden_size)
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

def inputTensor(line):
    tensor = torch.zeros(len(line) + 1, 1, n_letters)
    tensor[0][0][n_letters - 1] = 1
    for li in range(len(line)):
        letter = line[li]
        tensor[li + 1][0][all_letters.find(letter)] = 1
    return tensor

def train():
    def randomChoice(l): return l[random.randint(0, len(l) - 1)]
    def randomTrainingPair():
        category = randomChoice(all_categories)
        line = randomChoice(category_lines[category])
        return category, line

    def targetTensor(line):
        letter_indexes = [all_letters.find(line[li]) for li in range(0, len(line))]
        letter_indexes.append(n_letters - 2) # EOS
        return torch.tensor(letter_indexes, device=device, dtype=torch.int64)

    def randomTrainingExample():
        category, line = randomTrainingPair()
        category_tensor = categoryTensor(category)
        input_line_tensor = inputTensor(line)
        target_line_tensor = targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor


    criterion = nn.NLLLoss()
    learning_rate = 0.0005
    rnn = RNN(n_letters, 128, n_letters)

    def train(category_tensor, input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1)
        hidden = rnn.initHidden()

        rnn.zero_grad()

        loss = torch.tensor([0], device=device, dtype=torch.float32)
        def toLetter(l):
            if l == n_letters - 1:
                return '^'
            elif l == n_letters - 2:
                return '$'
            else:
                return all_letters[l]

        '''
        print('Input: ', end='')
        for letter in input_line_tensor:
            for idx, l in enumerate(letter[0]):
                if l > 0.01:
                    print(toLetter(idx), end='')
        print('\nTarget: ', end='')
        for letter in target_line_tensor:
            print(toLetter(letter[0]), end='')
        print('')
        '''
        for i in range(input_line_tensor.size(0)):
            output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
            if i > 0:
                l = criterion(output, target_line_tensor[i])
                loss += l

        loss.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item() / input_line_tensor.size(0)

    def timeSince(since): now = time.time(); s = now - since; m = math.floor(s / 60); s -= m * 60; return '%dm %ds' % (m, s)


    n_iters = 100000
    # n_iters = 15000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every ``plot_every`` ``iters``

    start = time.time()

    print('Start train')
    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    torch.save(rnn.state_dict(), "rnn.pth")
    plt.figure()
    plt.plot(all_losses)
    return rnn

def load():
    rnn = RNN(n_letters, 128, n_letters).to(device)
    rnn.load_state_dict(torch.load("rnn.pth"))
    return rnn

rnn = train()

max_length = 20

# Sample from a category and starting letter
def sample(category):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category)
        input = inputTensor('')
        hidden = rnn.initHidden()

        output_name = ''

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 2:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        print(output_name)

# Get multiple samples from one category and multiple starting letters

sample('German')
sample('Spanish')
sample('Chinese')
