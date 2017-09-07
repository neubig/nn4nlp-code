from collections import defaultdict
import math
import time
import random
import dynet as dy
import numpy as np
import pdb

K=3 #number of negative samples
N=2 #length of window on each side (so N=2 gives a total window size of 5, as in t-2 t-1 t t+1 t+2)
EMB_SIZE = 128 # The size of the embedding

embeddings_location = "embeddings.txt" #the file to write the word embeddings to
labels_location = "labels.txt" #the file to write the labels to

# We reuse the data reading from the language modeling class
w2i = defaultdict(lambda: len(w2i))

#word counts for negative sampling
word_counts = defaultdict(int)

S = w2i["<s>"]
UNK = w2i["<unk>"]
def read_dataset(filename):
  with open(filename, "r") as f:
    for line in f:
      line = line.strip().split(" ")
      for word in line:
        word_counts[w2i[word]] += 1
      yield [w2i[x] for x in line]


# Read in the data
train = list(read_dataset("../data/ptb/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/ptb/valid.txt"))
i2w = {v: k for k, v in w2i.items()}
nwords = len(w2i)


# take the word counts to the 3/4, normalize
counts =  np.array([list(x) for x in word_counts.items()])[:,1]**.75
normalizing_constant = sum(counts)
word_probabilities = np.zeros(nwords)
for word_id in word_counts:
  word_probabilities[word_id] = word_counts[word_id]**.75/normalizing_constant

with open(labels_location, 'w') as labels_file:
  for i in range(nwords):
    labels_file.write(i2w[i] + '\n')

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.SimpleSGDTrainer(model, learning_rate=0.1)

# Define the model
W_c_p = model.add_lookup_parameters((nwords, EMB_SIZE)) # Context word weights
W_w_p = model.add_lookup_parameters((nwords, EMB_SIZE)) # Word weights

# Calculate the loss value for the entire sentence
def calc_sent_loss(sent):
  # Create a computation graph
  dy.renew_cg()
  
  # Get embeddings for the sentence
  emb = [W_w_p[x] for x in sent]

  # Sample K negative words for each predicted word at each position
  all_neg_words = np.random.choice(nwords, size=2*N*K*len(emb), replace=True, p=word_probabilities)

  # W_w = dy.parameter(W_w_p)
  # Step through the sentence and calculate the negative and positive losses
  all_losses = [] 
  for i, my_emb in enumerate(emb):
    neg_words = all_neg_words[i*K*2*N:(i+1)*K*2*N]
    pos_words = ([sent[x] if x >= 0 else S for x in range(i-N,i)] +
                 [sent[x] if x < len(sent) else S for x in range(i+1,i+N+1)])
    neg_loss = -dy.log(dy.logistic(-dy.dot_product(my_emb, dy.lookup_batch(W_c_p, neg_words))))
    pos_loss = -dy.log(dy.logistic(dy.dot_product(my_emb, dy.lookup_batch(W_c_p, pos_words))))
    all_losses.append(dy.sum_batches(neg_loss) + dy.sum_batches(pos_loss))
  return dy.esum(all_losses)

MAX_LEN = 100

for ITER in range(100):
  print("started iter %r" % ITER)
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

  print("saving embedding files")
  with open(embeddings_location, 'w') as embeddings_file:
    W_w_np = W_w_p.as_array()
    for i in range(nwords):
      ith_embedding = '\t'.join(map(str, W_w_np[i]))
      embeddings_file.write(ith_embedding + '\n')
