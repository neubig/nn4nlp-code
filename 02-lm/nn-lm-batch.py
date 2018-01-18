from collections import defaultdict
import math
import time
import random
import dynet as dy
import numpy as np

N = 2 # The length of the n-gram
EMB_SIZE = 128 # The size of the embedding
HID_SIZE = 128 # The size of the hidden layer

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
trainer = dy.AdamTrainer(model, alpha=0.001)

# Define the model
W_emb = model.add_lookup_parameters((nwords, EMB_SIZE)) # Word weights at each position
W_h_p = model.add_parameters((HID_SIZE, EMB_SIZE * N))    # Weights of the softmax
b_h_p = model.add_parameters((HID_SIZE))                  # Weights of the softmax
W_sm_p = model.add_parameters((nwords, HID_SIZE))         # Weights of the softmax
b_sm_p = model.add_parameters((nwords))                   # Softmax bias

# A function to calculate scores for one value
def calc_score_of_histories(words, dropout=0.0):
  # This will change from a list of histories, to a list of words in each history position
  words = np.transpose(words)
  # Lookup the embeddings and concatenate them
  emb = dy.concatenate([dy.lookup_batch(W_emb, x) for x in words])
  # Create the hidden layer
  W_h = dy.parameter(W_h_p)
  b_h = dy.parameter(b_h_p)
  h = dy.tanh(dy.affine_transform([b_h, W_h, emb]))
  # Perform dropout
  if dropout != 0.0:
    h = dy.dropout(h, dropout)
  # Calculate the score and return
  W_sm = dy.parameter(W_sm_p)
  b_sm = dy.parameter(b_sm_p)
  return dy.affine_transform([b_sm, W_sm, h])

# Calculate the loss value for the entire sentence
def calc_sent_loss(sent, dropout=0.0):
  # Create a computation graph
  dy.renew_cg()
  # The initial history is equal to end of sentence symbols
  hist = [S] * N
  # Step through the sentence, including the end of sentence token
  all_histories = []
  all_targets = []
  for next_word in sent + [S]:
    all_histories.append(list(hist))
    all_targets.append(next_word)
    hist = hist[1:] + [next_word]
  s = calc_score_of_histories(all_histories, dropout=dropout)
  return dy.sum_batches(dy.pickneglogsoftmax_batch(s, all_targets))

MAX_LEN = 100
# Generate a sentence
def generate_sent():
  dy.renew_cg()
  hist = [S] * N
  sent = []
  while True:
    p = dy.softmax(calc_score_of_histories([hist])).npvalue()
    next_word = np.random.choice(nwords, p=p/p.sum())
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
  train_words, train_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(train):
    my_loss = calc_sent_loss(sent, dropout=0.2)
    train_loss += my_loss.value()
    train_words += len(sent)
    my_loss.backward()
    trainer.update()
    if (sent_id+1) % 5000 == 0:
      print("--finished %r sentences (word/sec=%.2f)" % (sent_id+1, train_words/(time.time()-start)))
  print("iter %r: train loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (ITER, train_loss/train_words, math.exp(train_loss/train_words), train_words/(time.time()-start)))
  # Evaluate on dev set
  dev_words, dev_loss = 0, 0.0
  start = time.time()
  for sent_id, sent in enumerate(dev):
    my_loss = calc_sent_loss(sent)
    dev_loss += my_loss.value()
    dev_words += len(sent)
    trainer.update()
  # Keep track of the development accuracy and reduce the learning rate if it got worse
  if last_dev < dev_loss:
    trainer.learning_rate /= 2
  last_dev = dev_loss
  # Keep track of the best development accuracy, and save the model only if it's the best one
  if best_dev > dev_loss:
    model.save("model.txt")
    best_dev = dev_loss
  # Save the model
  print("iter %r: dev loss/word=%.4f, ppl=%.4f (word/sec=%.2f)" % (ITER, dev_loss/dev_words, math.exp(dev_loss/dev_words), dev_words/(time.time()-start)))
  # Generate a few sentences
  for _ in range(5):
    sent = generate_sent()
    print(" ".join([i2w[x] for x in sent]))
