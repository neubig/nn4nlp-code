from collections import defaultdict
import time
import random
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from model import DeepCBoW


# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# initialize the model
EMB_SIZE = 64
HID_SIZE = 64
NLAYERS = 2
model = DeepCBoW(nwords, ntags, NLAYERS, EMB_SIZE, HID_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()


for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for words, tag in train:
        words = torch.tensor(words).type(type)
        tag = torch.tensor([tag]).type(type)
        scores = model(words)
        loss = criterion(scores, tag)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (
                ITER, train_loss/len(train), time.time()-start))
    # Perform testing
    test_correct = 0.0
    for words, tag in dev:
        words = torch.tensor(words).type(type)
        scores = model(words)[0].detach().cpu().numpy()
        predict = np.argmax(scores)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct/len(dev)))





