from collections import defaultdict
import math
import time
import random
import torch


class WordEmbSkip(torch.nn.Module):
    def __init__(self, nwords, emb_size):
        super(WordEmbSkip, self).__init__()

        """ word embeddings """
        self.word_embedding = torch.nn.Embedding(nwords, emb_size)
        # uniform initialization
        torch.nn.init.uniform_(self.word_embedding.weight, -0.25, 0.25)
        """ context embeddings"""
        self.context_embedding = torch.nn.Parameter(torch.randn(emb_size, nwords))

    def forward(self, word):
        embed_word = self.word_embedding(word)    # 1 * emb_size
        out = torch.mm(embed_word, self.context_embedding)  # 1 * nwords
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
model = WordEmbSkip(nwords, EMB_SIZE)
criterion = torch.nn.CrossEntropyLoss()
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

    # Step through the sentence
    losses = []
    for i, word in enumerate(sent):
        for j in range(1, N + 1):
            for direction in [-1, 1]:
                c = torch.tensor([word]).type(type)  # This is tensor for center word
                context_id = sent[i + direction * j] if 0 <= i + direction * j < len(sent) else S
                context = torch.tensor([context_id]).type(type)   # Tensor for context word
                logits = model(c)
                loss = criterion(logits, context)
                losses.append(loss)
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
        if (sent_id + 1) % 5000 == 0:
            print("--finished %r sentences" % (sent_id + 1))
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
