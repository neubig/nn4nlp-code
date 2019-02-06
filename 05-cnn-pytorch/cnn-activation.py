from collections import defaultdict
import time
import random
import torch
import numpy as np


class CNNclass(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_size, ntags):
        super(CNNclass, self).__init__()

        """ layers """
        self.embedding = torch.nn.Embedding(nwords, emb_size)
        # uniform initialization
        torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
        # Conv 1d
        self.conv_1d = torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_size,
                                       stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.relu = torch.nn.ReLU()
        self.projection_layer = torch.nn.Linear(in_features=num_filters, out_features=ntags, bias=True)
        # Initializing the projection layer
        torch.nn.init.xavier_uniform_(self.projection_layer.weight)

    def forward(self, words, return_activations=False):
        emb = self.embedding(words)                 # nwords x emb_size
        emb = emb.unsqueeze(0).permute(0, 2, 1)     # 1 x emb_size x nwords
        h = self.conv_1d(emb)                       # 1 x num_filters x nwords
        activations = h.squeeze(0).max(dim=1)[1]     # argmax along length of the sentence
        # Do max pooling
        h = h.max(dim=2)[0]                         # 1 x num_filters
        h = self.relu(h)
        features = h.squeeze(0)
        out = self.projection_layer(h)              # size(out) = 1 x ntags
        if return_activations:
            return out, activations.data.cpu().numpy(), features.data.cpu().numpy()
        return out


np.set_printoptions(linewidth=np.nan, threshold=np.nan)

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = words.split(" ")
            yield (words, [w2i[x] for x in words], int(tag))


# Read in the data
train = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = 5

# Define the model
EMB_SIZE = 10
WIN_SIZE = 3
FILTER_SIZE = 8

# initialize the model
model = CNNclass(nwords, EMB_SIZE, FILTER_SIZE, WIN_SIZE, ntags)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

type = torch.LongTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    model.cuda()


def calc_predict_and_activations(wids, tag, words):
    if len(wids) < WIN_SIZE:
        wids += [0] * (WIN_SIZE-len(wids))
    words_tensor = torch.tensor(wids).type(type)
    scores, activations, features = model(words_tensor, return_activations=True)
    scores = scores.squeeze().cpu().data.numpy()
    print('%d ||| %s' % (tag, ' '.join(words)))
    predict = np.argmax(scores)
    print(display_activations(words, activations))
    W = model.projection_layer.weight.data.cpu().numpy()
    bias = model.projection_layer.bias.data.cpu().numpy()
    print('scores=%s, predict: %d' % (scores, predict))
    print('  bias=%s' % bias)
    contributions = W * features
    print(' very bad (%.4f): %s' % (scores[0], contributions[0]))
    print('      bad (%.4f): %s' % (scores[1], contributions[1]))
    print('  neutral (%.4f): %s' % (scores[2], contributions[2]))
    print('     good (%.4f): %s' % (scores[3], contributions[3]))
    print('very good (%.4f): %s' % (scores[4], contributions[4]))


def display_activations(words, activations):
    pad_begin = (WIN_SIZE - 1) / 2
    pad_end = WIN_SIZE - 1 - pad_begin
    words_padded = ['pad' for _ in range(int(pad_begin))] + words + ['pad' for _ in range(int(pad_end))]

    ngrams = []
    for act in activations:
        ngrams.append('[' + ', '.join(words_padded[act:act+WIN_SIZE]) + ']')

    return ngrams


for ITER in range(10):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for _, wids, tag in train:
        # Padding (can be done in the conv layer as well)
        if len(wids) < WIN_SIZE:
            wids += [0] * (WIN_SIZE - len(wids))
        words_tensor = torch.tensor(wids).type(type)
        tag_tensor = torch.tensor([tag]).type(type)
        scores = model(words_tensor)
        predict = scores[0].argmax().item()
        if predict == tag:
            train_correct += 1

        my_loss = criterion(scores, tag_tensor)
        train_loss += my_loss.item()
        # Do back-prop
        optimizer.zero_grad()
        my_loss.backward()
        optimizer.step()
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (ITER, train_loss/len(train), train_correct/len(train), time.time()-start))
    # Perform testing
    test_correct = 0.0
    for _, wids, tag in dev:
        # Padding (can be done in the conv layer as well)
        if len(wids) < WIN_SIZE:
            wids += [0] * (WIN_SIZE - len(wids))
        words_tensor = torch.tensor(wids).type(type)
        scores = model(words_tensor)
        predict = scores[0].argmax().item()
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct/len(dev)))


for words, wids, tag in dev:
    calc_predict_and_activations(wids, tag, words)
    input()
