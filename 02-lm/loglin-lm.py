from collections import defaultdict
import math
import time
import random
import dynet as dy
import numpy as np

# The length of the n-gram
N = 2

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

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.SimpleSGDTrainer(model, learning_rate=0.1)

# Define the model
W_sm = [model.add_lookup_parameters((nwords, nwords)) for _ in range(N)] # Word weights at each position
b_sm = model.add_parameters((nwords))                # Softmax bias

# A function to calculate scores for one value
def calc_score_of_history(words):
  # Create a list of things to sum up with only the bias vector at first
  score_vecs = [dy.parameter(b_sm)]
  for word_id, lookup_param in zip(words, W_sm): 
    score_vecs.append(lookup_param[word_id])
  return dy.esum(score_vecs)

# Calculate the loss value for the entire sentence
def calc_sent_loss(sent):
  # Create a computation graph
  dy.renew_cg()
  # The initial history is equal to end of sentence symbols
  hist = [S] * N
  # Step through the sentence, including the end of sentence token
  all_losses = []
  for next_word in sent + [S]:
    s = calc_score_of_history(hist)
    all_losses.append(dy.pickneglogsoftmax(s, next_word))
    hist = hist[1:] + [next_word]
  return dy.esum(all_losses)

MAX_LEN = 100
# Generate a sentence
def generate_sent():
  dy.renew_cg()
  hist = [S] * N
  sent = []
  while True:
    p = dy.softmax(calc_score_of_history(hist)).npvalue()
    next_word = np.random.choice(nwords, p=p/p.sum())
    if next_word == S or len(sent) == MAX_LEN:
      break
    sent.append(next_word)
    hist = hist[1:] + [next_word]
  return sent

for ITER in range(100):
  # Perform training
  random.shuffle(train)
  train_words, train_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(train):
    my_loss = calc_sent_loss(sent)
    train_loss += my_loss.value()
    train_words += len(sent)
    my_loss.backward()
    trainer.update()
    if (sent_id+1) % 5000 == 0:
      print("--finished %r sentences" % (sent_id+1))
  print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), time.time()-start))
  # Evaluate on dev set
  dev_words, dev_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(dev):
    my_loss = calc_sent_loss(sent)
    dev_loss += my_loss.value()
    dev_words += len(sent)
    trainer.update()
  print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (ITER, dev_loss/dev_words, math.exp(dev_loss/dev_words), time.time()-start))
  # Generate a few sentences
  for _ in range(5):
    sent = generate_sent()
    print(" ".join([i2w[x] for x in sent]))
