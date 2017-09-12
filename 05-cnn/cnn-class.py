from collections import defaultdict
import time
import random
import dynet as dy
import numpy as np

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, "r") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            yield ([w2i[x] for x in words.split(" ")], t2i[tag])

# Read in the data
train = list(read_dataset("../data/classes/train.txt"))
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("../data/classes/test.txt"))
nwords = len(w2i)
ntags = len(t2i)

# Start DyNet and define trainer
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Define the model
EMB_SIZE = 64
W_emb = model.add_lookup_parameters((nwords, 1, 1, EMB_SIZE)) # Word embeddings
WIN_SIZE = 3
FILTER_SIZE = 64
W_cnn = model.add_parameters((1, WIN_SIZE, EMB_SIZE, FILTER_SIZE)) # cnn weights
b_cnn = model.add_parameters((FILTER_SIZE)) # cnn bias

W_sm = model.add_parameters((ntags, FILTER_SIZE))          # Softmax weights
b_sm = model.add_parameters((ntags))                      # Softmax bias

def calc_scores(words):
    dy.renew_cg()
    W_cnn_express = dy.parameter(W_cnn)
    b_cnn_express = dy.parameter(b_cnn)
    W_sm_express = dy.parameter(W_sm)
    b_sm_express = dy.parameter(b_sm)
    if len(words) < WIN_SIZE:
      words += [0] * (WIN_SIZE-len(words))

    cnn_in = dy.concatenate([dy.lookup(W_emb, x) for x in words], d=1)
    cnn_out = dy.conv2d_bias(cnn_in, W_cnn_express, b_cnn_express, stride=(1, 1), is_valid=False)
    pool_out = dy.max_dim(cnn_out, d=1)
    pool_out = dy.reshape(pool_out, (FILTER_SIZE,))
    pool_out = dy.rectify(pool_out)
    return W_sm_express * pool_out + b_sm_express

for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for words, tag in train:
        scores = calc_scores(words)
        predict = np.argmax(scores.npvalue())
        if predict == tag:
            train_correct += 1

        my_loss = dy.pickneglogsoftmax(scores, tag)
        train_loss += my_loss.value()
        my_loss.backward()
        trainer.update()
    print("iter %r: train loss/sent=%.4f, acc=%.4f, time=%.2fs" % (ITER, train_loss/len(train), train_correct/len(train), time.time()-start))
    # Perform testing
    test_correct = 0.0
    for words, tag in dev:
        scores = calc_scores(words).npvalue()
        predict = np.argmax(scores)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct/len(dev)))

