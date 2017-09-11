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
            words = words.split(" ")
            yield (words, [w2i[x] for x in words], t2i[tag])

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
W_emb = model.add_lookup_parameters((nwords, EMB_SIZE)) # Word embeddings
WIN_SIZE = 3
FILTER_SIZE = 64
W_cnn = model.add_parameters((WIN_SIZE, 1, EMB_SIZE, FILTER_SIZE)) # cnn weights
b_cnn = model.add_parameters((FILTER_SIZE)) # cnn bias

W_sm = model.add_parameters((ntags, FILTER_SIZE))          # Softmax weights
b_sm = model.add_parameters((ntags))                      # Softmax bias

def calc_scores(wids):
    dy.renew_cg()
    W_cnn_express = dy.parameter(W_cnn)
    b_cnn_express = dy.parameter(b_cnn)
    W_sm_express = dy.parameter(W_sm)
    b_sm_express = dy.parameter(b_sm)

    cnn_in = dy.concatenate([dy.lookup(W_emb, x) for x in wids], d=1)
    cnn_in = dy.reshape(cnn_in, (len(wids), 1, EMB_SIZE))
    cnn_out = dy.conv2d_bias(cnn_in, W_cnn_express, b_cnn_express, stride=(1, 1), is_valid=False)
    pool_out = dy.maxpooling2d(cnn_out, (len(wids), 1), (1, 1), is_valid=True)
    pool_out = dy.reshape(pool_out, (FILTER_SIZE,))
    return W_sm_express * pool_out + b_sm_express

def calc_predict_and_activations(wids):
    dy.renew_cg()
    W_cnn_express = dy.parameter(W_cnn)
    b_cnn_express = dy.parameter(b_cnn)
    W_sm_express = dy.parameter(W_sm)
    b_sm_express = dy.parameter(b_sm)

    cnn_in = dy.concatenate([dy.lookup(W_emb, x) for x in wids], d=1)
    cnn_in = dy.reshape(cnn_in, (len(wids), 1, EMB_SIZE))
    cnn_out = dy.conv2d_bias(cnn_in, W_cnn_express, b_cnn_express, stride=(1, 1), is_valid=False)
    filters = (dy.reshape(cnn_out, (len(wids), FILTER_SIZE))).npvalue()
    activations = filters.argmax(axis=0)

    pool_out = dy.maxpooling2d(cnn_out, (len(wids), 1), (1, 1), is_valid=True)
    pool_out = dy.reshape(pool_out, (FILTER_SIZE,))
    scores = (W_sm_express * pool_out + b_sm_express).npvalue()
    return np.argmax(scores), activations

def display_activations(words, activations):
    pad_begin = (WIN_SIZE - 1) / 2
    pad_end = WIN_SIZE - 1 - pad_begin
    words_padded = ['pad' for i in range(pad_begin)] + words + ['pad' for i in range(pad_end)]

    ngrams = []
    for act in activations:
        ngrams.append('[' + ', '.join(words_padded[act:act+WIN_SIZE]) + ']')

    return ngrams

for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    train_correct = 0.0
    start = time.time()
    for _, wids, tag in train:
        scores = calc_scores(wids)
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
    for _, wids, tag in dev:
        scores = calc_scores(wids).npvalue()
        predict = np.argmax(scores)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct/len(dev)))


for words, wids, tag in dev:
    predict, activations = calc_predict_and_activations(wids)
    print 'sent: %s' % ' '.join(words)
    print display_activations(words, activations)
    raw_input()