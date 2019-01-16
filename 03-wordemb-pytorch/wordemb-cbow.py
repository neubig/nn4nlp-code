from collections import defaultdict
import math
import time
import random
import torch


class CBoW(torch.nn.Module):
    def __init__(self, nwords, emb_size):
        super(CBoW, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        """ projection layer for taking softmax over vocabulary words"""
        self.projection = torch.nn.Linear(emb_size, nwords)

    def forward(self, words):
        emb = self.embedding(words)
        emb_sum = torch.sum(emb, dim=0)  # size(emb_sum) = emb_size
        emb_sum = emb_sum.view(1, -1)  # size(emb_sum) = 1 x emb_size
        out = self.projection(emb_sum)  # size(out) = 1 x nwords
        return out


N = 2  # length of window on each side (so N=2 gives a total window size of 5, as in t-2 t-1 t t+1 t+2)
EMB_SIZE = 128  # The size of the embedding

embeddings_location = "embeddings.txt"  # the file to write the word embeddings to
labels_location = "labels.txt"  # the file to write the labels to

# We reuse the data reading from the language modeling class
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

with open(labels_location, 'w') as labels_file:
    for i in range(nwords):
        labels_file.write(i2w[i] + '\n')

# initialize the model
model = CBoW(nwords, EMB_SIZE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()


# Calculate the loss value for the entire sentence
def calc_sent_loss(sent, inference=False):

    # add padding to the sentence equal to the size of the window
    # as we need to predict the eos as well, the future window at that point is N past it
    padded_sent = [S] * N + sent + [S] * N

    # Step through the sentence
    total_loss = 0
    for i in range(N, len(sent) + N):
        # c is the context vector
        c = torch.tensor(padded_sent[i - N:i] + padded_sent[i + 1:i + N + 1]).type(type)
        t = torch.tensor([padded_sent[i]]).type(type) # This is the target vector
        log_prob = model(c)
        loss = criterion(log_prob, t)   # loss for predicting target from context vector
        if not inference:
            # Back prop while training only
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.data.cpu().item()
    return total_loss


MAX_LEN = 100

for ITER in range(100):
    print("started iter %r" % ITER)
    # Perform training
    random.shuffle(train)
    train_words, train_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(train):
        my_loss = calc_sent_loss(sent)
        train_loss += my_loss
        train_words += len(sent)
        if (sent_id + 1) % 5000 == 0:
            print("--finished %r sentences" % (sent_id + 1))
    print("iter %r: train loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, train_loss / train_words, math.exp(train_loss / train_words), time.time() - start))
    # Evaluate on dev set
    dev_words, dev_loss = 0, 0.0
    start = time.time()
    for sent_id, sent in enumerate(dev):
        my_loss = calc_sent_loss(sent, inference=True)
        dev_loss += my_loss
        dev_words += len(sent)
    print("iter %r: dev loss/word=%.4f, ppl=%.4f, time=%.2fs" % (
    ITER, dev_loss / dev_words, math.exp(dev_loss / dev_words), time.time() - start))

    print("saving embedding files")
    with open(embeddings_location, 'w') as embeddings_file:
        W_w_np = model.embedding.weight.data.cpu().numpy()
        for i in range(nwords):
            ith_embedding = '\t'.join(map(str, W_w_np[i]))
            embeddings_file.write(ith_embedding + '\n')
