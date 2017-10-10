from __future__ import print_function
import time
import random

start = time.time()

from collections import Counter, defaultdict
from biaffine import DeepBiaffineAttentionDecoder

import dynet as dy
import numpy as np

# format of files: each line is "word1/tag2 word2/tag2 ..."
train_file = "../data/parsing/graph/ptb_train.txt"
test_file = "../data/parsing//graph/ptb_dev.txt"

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

def read(fname):
    with open(fname, "r") as fh:
        for line in fh:
            tokens = line.strip().split()
            num_tokens = len(tokens)
            assert num_tokens % 3 == 0
            sent = []
            labels = []
            heads = []
            for i in range(num_tokens / 3):
                sent.append(w2i[tokens[3 * i]])
                labels.append(t2i[tokens[3 * i + 1]])
                heads.append(int(tokens[3 * i + 2]))
            yield (sent, labels, heads)


train = list(read(train_file))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read(test_file))
nwords = len(w2i)
ntags = len(t2i)

# DyNet Starts

model = dy.Model()
trainer = dy.AdamTrainer(model)

# Lookup parameters for word embeddings
EMB_SIZE = 32
HID_SIZE = 64
W_emb = model.add_lookup_parameters((nwords, EMB_SIZE))  # Word embeddings
fwdLSTM = dy.SimpleRNNBuilder(1, EMB_SIZE, HID_SIZE, model)  # Forward LSTM
bwdLSTM = dy.SimpleRNNBuilder(1, EMB_SIZE, HID_SIZE, model)  # Backward LSTM

biaffineParser = DeepBiaffineAttentionDecoder(model, ntags, src_ctx_dim=HID_SIZE * 2,
                                              n_arc_mlp_units=64, n_label_mlp_units=32)

def calc_loss(words, labels, heads):
    dy.renew_cg()
    word_embs = [dy.lookup(W_emb, x) for x in words]
    fwd_init = fwdLSTM.initial_state()
    fwd_embs = fwd_init.transduce(word_embs)
    bwd_init = bwdLSTM.initial_state()
    bwd_embs = bwd_init.transduce(reversed(word_embs))
    src_encodings = [dy.reshape(dy.concatenate([f, b]), (HID_SIZE * 2, 1)) for f, b in zip(fwd_embs, reversed(bwd_embs))]
    return biaffineParser.decode_loss(src_encodings, ([heads], [labels]))


def calc_acc(words, labels, heads):
    dy.renew_cg()
    word_embs = [dy.lookup(W_emb, x) for x in words]
    fwd_init = fwdLSTM.initial_state()
    fwd_embs = fwd_init.transduce(word_embs)
    bwd_init = bwdLSTM.initial_state()
    bwd_embs = bwd_init.transduce(reversed(word_embs))
    src_encodings = [dy.reshape(dy.concatenate([f, b]), (HID_SIZE * 2, 1)) for f, b in zip(fwd_embs, reversed(bwd_embs))]
    pred_heads, pred_labels = biaffineParser.decoding(src_encodings)
    return biaffineParser.cal_accuracy(pred_heads, pred_labels, heads, labels)

for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for words, labels, heads in train:
        loss = calc_loss(words, labels, heads)
        train_loss += loss.value()
        loss.backward()
        trainer.update()

    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))

    correct_heads = 0.
    correct_labels = 0.
    total = 0.
    for words, labels, heads in dev:
        head_acc, label_acc = calc_acc(words, labels, heads)
        correct_heads += head_acc * len(words)
        correct_labels += label_acc * len(words)
        total += len(words)
    print("iter %r: test head_acc=%.4f, label_acc=%.4f" % (ITER, correct_heads * 100 / total,
                                                           correct_labels * 100 / total))

