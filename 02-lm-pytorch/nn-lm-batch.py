from collections import defaultdict
import math
import time
import random
import os, sys

import torch
import torch.nn as nn
from torch.autograd import Variable

# Feed-forward Neural Network Language Model
class FNN_LM(nn.Module):
  def __init__(self, nwords, emb_size, hid_size, num_hist, dropout):
    super(FNN_LM, self).__init__()
    self.embedding = nn.Embedding(nwords, emb_size)
    self.fnn = nn.Sequential(
      nn.Linear(num_hist*emb_size, hid_size), nn.Tanh(),
      nn.Dropout(dropout),
      nn.Linear(hid_size, nwords)
    )

  def forward(self, words):
    emb = self.embedding(words)      # 3D Tensor of size [batch_size x num_hist x emb_size]
    feat = emb.view(emb.size(0), -1) # 2D Tensor of size [batch_size x (num_hist*emb_size)]
    logit = self.fnn(feat)           # 2D Tensor of size [batch_size x nwords]

    return logit

N = 2 # The length of the n-gram
EMB_SIZE = 128 # The size of the embedding
HID_SIZE = 128 # The size of the hidden layer

USE_CUDA = torch.cuda.is_available()

# Functions to read in the corpus
# NOTE: We are using data from the Penn Treebank, which is already converted
#       into an easy-to-use format with "<unk>" symbols. If we were using other
#       data we would have to do pre-processing and consider how to choose
#       unknown words, etc.
w2i = defaultdict(lambda: len(w2i))
S = w2i["<s>"]
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      yield [w2i[x] for x in line.strip().split(" ")]

# Read in the data
train = list(read_dataset("../data/ptb/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/ptb/valid.txt"))
i2w = {v: k for k, v in w2i.items()}
nwords = len(w2i)

# Initialize the model and the optimizer
model = FNN_LM(nwords=nwords, emb_size=EMB_SIZE, hid_size=HID_SIZE, num_hist=N, dropout=0.2)
if USE_CUDA:
  model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# convert a (nested) list of int into a pytorch Variable
def convert_to_variable(words):
  var = Variable(torch.LongTensor(words))
  if USE_CUDA:
    var = var.cuda()

  return var

# A function to calculate scores for one value
def calc_score_of_histories(words):
  # This will change from a list of histories, to a pytorch Variable whose data type is LongTensor
  words_var = convert_to_variable(words)
  logits = model(words_var)
  return logits

# Calculate the loss value for the entire sentence
def calc_sent_loss(sent):
  # The initial history is equal to end of sentence symbols
  hist = [S] * N
  # Step through the sentence, including the end of sentence token
  all_histories = []
  all_targets = []
  for next_word in sent + [S]:
    all_histories.append(list(hist))
    all_targets.append(next_word)
    hist = hist[1:] + [next_word]

  logits = calc_score_of_histories(all_histories)
  loss = nn.functional.cross_entropy(logits, convert_to_variable(all_targets), size_average=False)

  return loss

MAX_LEN = 100
# Generate a sentence
def generate_sent():
  hist = [S] * N
  sent = []
  while True:
    logits = calc_score_of_histories([hist])
    prob = nn.functional.softmax(logits)
    next_word = prob.multinomial().data[0,0]
    if next_word == S or len(sent) == MAX_LEN:
      break
    sent.append(next_word)
    hist = hist[1:] + [next_word]
  return sent

last_dev = 1e20
best_dev = 1e20

for ITER in range(5):
  # Perform training
  random.shuffle(train)
  # set the model to training mode
  model.train()
  train_words, train_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(train):
    my_loss = calc_sent_loss(sent)
    train_loss += my_loss.data[0]
    train_words += len(sent)
    optimizer.zero_grad()
    my_loss.backward()
    optimizer.step()
    if (sent_id+1) % 5000 == 0:
      print("--finished %r sentences (word/sec=%.2f)" % (sent_id+1, train_words/(time.time()-start)))
  print("iter %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), train_words/(time.time()-start)))
  
  # Evaluate on dev set
  # set the model to evaluation mode
  model.eval()
  dev_words, dev_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(dev):
    my_loss = calc_sent_loss(sent)
    dev_loss += my_loss.data[0]
    dev_words += len(sent)

  # Keep track of the development accuracy and reduce the learning rate if it got worse
  if last_dev < dev_loss:
    optimizer.learning_rate /= 2
  last_dev = dev_loss
  
  # Keep track of the best development accuracy, and save the model only if it's the best one
  if best_dev > dev_loss:
    torch.save(model, "model.pt")
    best_dev = dev_loss
  
  # Save the model
  print("iter %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (ITER, dev_loss/dev_words, math.exp(dev_loss/dev_words), dev_words/(time.time()-start)))
  
  # Generate a few sentences
  for _ in range(5):
    sent = generate_sent()
    print(" ".join([i2w[x] for x in sent]))
