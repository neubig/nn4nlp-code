from collections import defaultdict
import math
import numpy as np
import time
import random
import torch
import torch.nn.functional as F


class WordEmbSkip(torch.nn.Module):
    def __init__(self, nwords, emb_size):
        super(WordEmbSkip, self).__init__()

        """ word embeddings """
        self.word_embedding = torch.nn.Embedding(nwords, emb_size, sparse=True)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        torch.nn.init.xavier_uniform_(self.word_embedding.weight)
        """ context embeddings"""
        self.context_embedding = torch.nn.Embedding(nwords, emb_size, sparse=True)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        torch.nn.init.xavier_uniform_(self.context_embedding.weight)

    # useful ref: https://arxiv.org/abs/1402.3722
    def forward(self, word_pos, context_positions, negative_sample=False):
        embed_word = self.word_embedding(word_pos)    # 1 * emb_size
        embed_context = self.context_embedding(context_positions)  # n * emb_size
        score = torch.matmul(embed_context, embed_word.transpose(dim0=1, dim1=0)) #score = n * 1

        # following is an example of something you can only do in a framework that allows
        # dynamic graph creation 
        if negative_sample:
              score = -1*score
        obj = -1 * torch.sum(F.logsigmoid(score))
        return obj

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

# initialize the model
model = WordEmbSkip(nwords, EMB_SIZE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()


# Calculate the loss value for the entire sentence
def calc_sent_loss(sent):
    # add padding to the sentence equal to the size of the window
    # as we need to predict the eos as well, the future window at that point is N past it
    all_neg_words = np.random.choice(nwords, size=2*N*K*len(sent), replace=True, p=word_probabilities)

    # Step through the sentence
    losses = []
    for i, word in enumerate(sent):
        pos_words = [sent[x] if x >= 0 else S for x in range(i-N,i)] + \
                     [sent[x] if x < len(sent) else S for x in range(i+1,i+N+1)]
        pos_words_tensor = torch.tensor(pos_words).type(type)
        neg_words = all_neg_words[i*K*2*N:(i+1)*K*2*N]
        neg_words_tensor = torch.tensor(neg_words).type(type)
        target_word_tensor = torch.tensor([word]).type(type)

        #NOTE: technically, one should ensure that the neg words don't contain the 
        #      the context (i.e. positive) words, but it is very unlikely, so we can ignore that

        pos_loss = model(target_word_tensor, pos_words_tensor)
        neg_loss = model(target_word_tensor, neg_words_tensor, negative_sample=True)

        losses.append(pos_loss + neg_loss)

    return torch.stack(losses).sum()


MAX_LEN = 100

for ITER in range(100):
    print("started iter %r" % ITER)
    # Perform training
    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    start = time.time()
    model.train()
    for sent_id, sent in enumerate(train):
        my_loss = calc_sent_loss(sent)
        train_loss += my_loss.item()
        train_words += len(sent)
        # Back prop while training
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
        if (sent_id + 1) % 50 == 0:
            print("--finished %r sentences" % (sent_id + 1))
            train_ppl = float('inf') if train_loss / train_words > 709 else math.exp(train_loss / train_words)
            print("after sentences %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
			sent_id + 1, train_loss / train_words, train_ppl, time.time() - start))
    train_ppl = float('inf') if train_loss / train_words > 709 else math.exp(train_loss / train_words)
    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, train_loss / train_words, train_ppl, time.time() - start))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    model.eval()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_sent_loss(sent)
        dev_loss += my_loss.item()
        dev_words += len(sent)
    dev_ppl = float('inf') if dev_loss / dev_words > 709 else math.exp(dev_loss / dev_words)
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, dev_loss / dev_words, dev_ppl, time.time() - start))

    print("saving embedding files")
    with open(embeddings_location, 'w') as embeddings_file:
        W_w_np = model.word_embedding.weight.data.cpu().numpy()
        for i in range(nwords):
            ith_embedding = '\t'.join(map(str, W_w_np[i]))
            embeddings_file.write(ith_embedding + '\n')
