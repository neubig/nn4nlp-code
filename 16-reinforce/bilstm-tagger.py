from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import dynet as dy
import numpy as np

#the parameters from mixer
NXENT = 40
NXER = 20
delta = 2

# format of files: each line is "word1|tag1 word2|tag2 ..."
train_file = "../data/tags/train.txt"
dev_file = "../data/tags/dev.txt"

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))


def read(fname):
    """
    Read tagged file
    """
    with open(fname, "r") as f:
        for line in f:
            words, tags = [], []
            for wt in line.strip().split():
                w, t = wt.split('|')
                words.append(w2i[w])
                tags.append(t2i[t])
            yield (words, tags)


# Read the data
train = list(read(train_file))
unk_word = w2i["<unk>"]
w2i = defaultdict(lambda: unk_word, w2i)
unk_tag = t2i["<unk>"]
t2i = defaultdict(lambda: unk_tag, t2i)
nwords = len(w2i)
ntags = len(t2i)
dev = list(read(dev_file))

# DyNet Starts
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Model parameters
EMBED_SIZE = 64
HIDDEN_SIZE = 128

# Lookup parameters for word embeddings
LOOKUP = model.add_lookup_parameters((nwords, EMBED_SIZE))

# Word-level BiLSTM
LSTM = dy.BiRNNBuilder(1, EMBED_SIZE, HIDDEN_SIZE, model, dy.LSTMBuilder)

# Word-level softmax
W_sm = model.add_parameters((ntags, HIDDEN_SIZE))
b_sm = model.add_parameters(ntags)


#Baseline reward parameters
W_bl_p = model.add_parameters((1, HIDDEN_SIZE))
b_bl_p = model.add_parameters(1)

# Calculate the scores for one example
def calc_scores(words):
    dy.renew_cg()

    # Transduce all batch elements with an LSTM
    word_reps = LSTM.transduce([LOOKUP[x] for x in words])

    # Softmax scores
    W = dy.parameter(W_sm)
    b = dy.parameter(b_sm)
    scores = [dy.affine_transform([b, W, x]) for x in word_reps]

    return scores


# Calculate MLE loss for one example
def calc_loss(scores, tags):
    losses = [dy.pickneglogsoftmax(score, tag) for score, tag in zip(scores, tags)]
    return dy.esum(losses)

def calc_reinforce_loss(words, tags, delta):
    dy.renew_cg()

    # Transduce all batch elements with an LSTM
    word_reps = LSTM.transduce([LOOKUP[x] for x in words])

    # Softmax scores
    W = dy.parameter(W_sm)
    b = dy.parameter(b_sm)

    #calculate the probability distribution 
    scores = [dy.affine_transform([b, W, x]) for x in word_reps]
    losses = [dy.pickneglogsoftmax(score, tag) for score, tag in zip(scores, tags)]
    probs = [-dy.exp(loss).as_array() for loss in losses]

    #then take samples from the probability distribution
    samples = [np.random.choice(range(len(x)), p=x) for x in probs]

    #calculate accuracy=reward
    correct = [sample == tag for sample, tag in zip(samples, tags)]
    r_i = float(sum(correct))/len(correct)
    r = dy.constant((1), r_i)
    # Reward baseline for each word
    W_bl = dy.parameter(W_bl_p)
    b_bl = dy.parameter(b_bl_p)
    r_b = [dy.affine_transform([b_bl, W_bl, dy.nobackprop(x)]) for x in word_reps]

    #we need to take the value in order to break the computation graph
    #as the reward portion is trained seperatley and not backpropogated through during the overall score
    rewards_over_baseline = [(r - dy.nobackprop(x)) for x in r_b]
    #the scores for training the baseline
    baseline_scores = [dy.square(r - x) for x in r_b]

    #then calculate the reinforce scores using reinforce
    reinforce_scores = [r_s*score for r_s, score in zip(rewards_over_baseline, scores)]

    #we want the first len(sent)-delta scores from xent then delta scores from reinforce
    #for mixer
    if len(scores) > delta:
        mixer_scores = scores[:len(scores)-delta] + reinforce_scores[delta-1:]
    else:
        mixer_scores = reinforce_scores
    return dy.esum(mixer_scores), dy.esum(baseline_scores)


# Calculate number of tags correct for one example
def calc_correct(scores, tags):
    correct = [np.argmax(score.npvalue()) == tag for score, tag in zip(scores, tags)]
    return sum(correct)


# Perform training
for ITER in range(NXENT+NXER):
    random.shuffle(train)
    start = time.time()
    this_sents = this_words = this_loss = this_correct = 0
    for sid in range(0, len(train)):
        this_sents += 1
        if this_sents % int(1000) == 0:
            print("train loss/word=%.4f, acc=%.2f%%, word/sec=%.4f" % (
                this_loss / this_words, 100 * this_correct / this_words, this_words / (time.time() - start)),
                  file=sys.stderr)
        # train on the example
        if ITER < NXER:
            words, tags = train[sid]
            scores = calc_scores(words)
            loss_exp = calc_loss(scores, tags)
            this_correct += calc_correct(scores, tags)
            this_loss += loss_exp.scalar_value()
            this_words += len(words)
            loss_exp.backward()
            trainer.update()
        else:
            delta = 2*(ITER - XENT)
            mixer_loss, baseline_loss = calc_reinforce_loss(words, tags, delta)
            this_loss += mixer_loss.scalar_value() + baseline_loss.scalar_value()
            this_words += len(words)
            mixer_loss.backward()
            baseline_loss.backward()
            trainer.update()
    # Perform evaluation 
    start = time.time()
    this_sents = this_words = this_loss = this_correct = 0
    for words, tags in dev:
        this_sents += 1
        scores = calc_scores(words)
        loss_exp = calc_loss(scores, tags)
        this_correct += calc_correct(scores, tags)
        this_loss += loss_exp.scalar_value()
        this_words += len(words)
    print("dev loss/word=%.4f, acc=%.2f%%, word/sec=%.4f" % (
        this_loss / this_words, 100 * this_correct / this_words, this_words / (time.time() - start)), file=sys.stderr)
