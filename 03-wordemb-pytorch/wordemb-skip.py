from collections import defaultdict
import math
import time
import random
import torch
import torch.nn.functional as F


class Skip(torch.nn.Module):
    def __init__(self, nwords, emb_size):
        super(Skip, self).__init__()

        """ word embeddings """
        self.word_embedding = torch.nn.Embedding(nwords, emb_size)
        """ context embeddings"""
        self.context_embedding = torch.nn.Embedding(nwords, emb_size)

    def forward(self, word_pos, context_pos):
        embed_word = self.word_embedding(word_pos)    # 1 * emb_size
        embed_context = self.context_embedding(context_pos)  # 1 * emb_size
        score = torch.mul(embed_word, embed_context)
        score = torch.sum(score, dim=1)
        log_target = -1 * F.logsigmoid(score).squeeze()
        return log_target


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
model = Skip(nwords, EMB_SIZE)
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

    # Step through the sentence
    total_loss = 0
    for i, word in enumerate(sent):
        c = torch.tensor([word]).type(type)     # This is tensor for center word
        for j in range(1, N + 1):
            for direction in [-1, 1]:
                context_id = sent[i + direction * j] if 0 <= i + direction * j < len(sent) else S
                context = torch.tensor([context_id]).type(type)   # Tensor for context word
                loss = model(c, context)
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
        W_w_np = model.word_embedding.weight.data.cpu().numpy()
        for i in range(nwords):
            ith_embedding = '\t'.join(map(str, W_w_np[i]))
            embeddings_file.write(ith_embedding + '\n')
