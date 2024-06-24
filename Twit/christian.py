#!/bin/rgpp
import os
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary

from sklearn import feature_extraction, linear_model, model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import compute_class_weight
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import KFold

import re
import nltk
import emoji

device = 'cuda'

class DisasterTweets(data.Dataset):
    def __init__(self, data_file):
        vectorizer = TfidfVectorizer(
                min_df=0.001,
                stop_words = None,
                tokenizer=DisasterTweets._process_tweet,
                token_pattern = None
                )
        encoder = OneHotEncoder(sparse_output=False)
        
        self.d = pd.read_csv(os.path.join(data_file))

        keywords = encoder.fit_transform(self.d[['keyword']])
        texts = vectorizer.fit_transform(self.d['text']).todense()

        self.data = torch.from_numpy(np.concatenate((keywords, texts), axis=1)).float()
        self.label = F.one_hot(torch.from_numpy(self.d['target'].to_numpy()), num_classes=2).float()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def class_weights(self):
        return compute_class_weight("balanced", classes=np.unique(self.d["target"].to_numpy()), y=self.d["target"].to_numpy())

    @classmethod
    def _process_tweet(self, tweet):
        url_pattern = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    
        tokenizer = nltk.tokenize.TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words("english")

        tokens = []
        for word in tokenizer.tokenize(tweet):
            if re.match(url_pattern, word) is not None:
                tokens.append("<LINK>")
                continue

            if emoji.is_emoji(word):
                tokens.append(emoji.demojize(word))
                continue
            
            if word in stopwords or not word.isalnum():
                continue
            
            tokens.append(lemmatizer.lemmatize(word))

        return tokens

class TweetClassifier(pl.LightningModule):
    def __init__(self, input_size, num_classes=2, class_weights=None, lr=0.001):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        self.save_hyperparameters()

        self.lr = lr
        self.class_weights = torch.tensor(class_weights)
        self.criterion = nn.BCEWithLogitsLoss(weight=self.class_weights)

#define metr(n,p) self.#1_metrics = torchmetrics.MetricCollection([torchmetrics.classification.Accuracy(task="binary"),torchmetrics.classification.F1Score(task="binary"),], prefix="#1_")
        metr(train)
        metr(validation)
        metr(test)

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=list(range( 50, 300, 50)))
        return [optimizer], [scheduler]

#define ts(n,a,b) self.#1_metrics.update(a, b); self.log_dict(self.#1_metrics, on_step=False, on_epoch=True)
    def training_step(self, d, i):
        lr = self.layers(d[0])
        ts(train,lr,d[1])
        return self.criterion(lr, d[1])

    def validation_step(self, d, i):
        ts(validation,self.layers(d[0]),d[1])

    def test_step(self, d, i):
        ts(test,self.layers(d[0]),d[1])

def train(mclass, tloader, vloader, class_weights, **kwargs):
    trainer = pl.Trainer(
        default_root_dir = "logs",
        accelerator = "gpu" if str(device).startswith("cuda") else "cpu",
        devices = 1,
        max_epochs = 300,
        callbacks = [
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="validation_BinaryF1Score"),
            ModelSummary(max_depth=-1),
        ],
        enable_progress_bar = True
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    input_size = next(iter(tloader))[0].shape[1]

    model = mclass(input_size, class_weights=class_weights)
    trainer.fit(model, tloader, vloader)
    print(trainer.checkpoint_callback.best_model_path)
    model = mclass.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model

def test(model, loader):
    trainer = pl.Trainer(
        default_root_dir = "logs",
        accelerator = "gpu" if str(device).startswith("cuda") else "cpu",
        devices = 1,
        max_epochs = 50,
        enable_progress_bar = True
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    trainer.test(model, loader)


if __name__ == '__main__':
    print('Finished importing')

    nltk.download("wordnet")
    nltk.download("stopwords")

    torch.set_default_device(device)
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.manual_seed(123)
    np.random.seed(123)

    fset = DisasterTweets('/home/arch/.datasets/twitter/train.csv')
    tset, vset = data.random_split(fset, torch.tensor([len(fset) - 2000, 2000]), generator=torch.Generator(device='cuda'))

#define dl(a,t) data.DataLoader(a, batch_size=64, shuffle=t, drop_last=False, num_workers=0, generator=torch.Generator(device=device))
    tloader = dl(tset,True)
    vloader = dl(vset,False)

    # try:
    model = train(TweetClassifier, tloader, vloader, fset.class_weights())
    # except:
        # exit()

    test(model, tloader)
    # teloader = data.DataLoader(teset, batch_size=32, shuffle=False, drop_last=False, num_workers=4, generator=torch.Generator(device=device))
    # teset = DisasterTweets('/home/arch/.datasets/twitter/test.csv')
