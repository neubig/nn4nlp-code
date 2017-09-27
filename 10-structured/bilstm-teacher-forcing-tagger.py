from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import dynet as dy
import numpy as np

parser = argparse.ArgumentParser(description='BiLSTM variants.')
parser.add_argument('--teacher', action='store_true')
parser.add_argument('--perceptron', action='store_true')
parser.add_argument('--hinge', action='store_true')

args = parser.parse_args()
use_teacher_forcing = args.teacher
use_structure_perceptron = args.perceptron
use_hinge = args.hinge

print("Training BiLSTM %s teacher forcing, %s structured perceptron loss, %s margin."
      % ("with" if use_teacher_forcing else "without",
         "with" if use_structure_perceptron else "without",
         "with" if use_hinge else "without")
      )

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
start_tag = t2i["<start>"]
t2i = defaultdict(lambda: unk_tag, t2i)
nwords = len(w2i)
ntags = len(t2i)
dev = list(read(dev_file))

# DyNet Starts
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Model parameters
EMBED_SIZE = 64
TAG_EMBED_SIZE = 16
HIDDEN_SIZE = 128

assert HIDDEN_SIZE % 2 == 0

# Lookup parameters for word embeddings
LOOKUP = model.add_lookup_parameters((nwords, EMBED_SIZE))

if use_teacher_forcing:
    TAG_LOOKUP = model.add_lookup_parameters((ntags, TAG_EMBED_SIZE))

# Word-level BiLSTM is just a composition of two LSTMs.
if use_teacher_forcing:
    fwdLSTM = dy.SimpleRNNBuilder(1, EMBED_SIZE + TAG_EMBED_SIZE, HIDDEN_SIZE / 2, model)  # Forward LSTM
else:
    fwdLSTM = dy.SimpleRNNBuilder(1, EMBED_SIZE, HIDDEN_SIZE / 2, model)  # Forward LSTM

# We cannot insert previous predicted tag to the backward LSTM anyway.
bwdLSTM = dy.SimpleRNNBuilder(1, EMBED_SIZE, HIDDEN_SIZE / 2, model)  # Backward LSTM

# Word-level softmax
W_sm = model.add_parameters((ntags, HIDDEN_SIZE))
b_sm = model.add_parameters(ntags)


# Calculate the scores for one example
def calc_scores(words):
    dy.renew_cg()

    word_embs = [LOOKUP[x] for x in words]

    # Transduce all batch elements with an LSTM
    fwd_init = fwdLSTM.initial_state()
    fwd_word_reps = fwd_init.transduce(word_embs)
    bwd_init = bwdLSTM.initial_state()
    bwd_word_reps = bwd_init.transduce(reversed(word_embs))

    combined_word_reps = [dy.concatenate([f, b]) for f, b in zip(fwd_word_reps, reversed(bwd_word_reps))]

    # Softmax scores
    W = dy.parameter(W_sm)
    b = dy.parameter(b_sm)
    scores = [dy.affine_transform([b, W, x]) for x in combined_word_reps]

    return scores


def calc_scores_with_tags(words, tags):
    dy.renew_cg()

    last_tags = [start_tag] + tags[:-1]

    word_embs = [LOOKUP[x] for x in words]
    tag_embs = [TAG_LOOKUP[x] for x in last_tags]

    input_embs = [dy.concatenate([w, t]) for w, t in zip(word_embs, tag_embs)]

    # Transduce all batch elements with an LSTM
    fwd_init = fwdLSTM.initial_state()
    fwd_word_reps = fwd_init.transduce(input_embs)  # NOTE: We use the concatenated embeddings for the forward LSTM.

    bwd_init = bwdLSTM.initial_state()
    bwd_word_reps = bwd_init.transduce(reversed(word_embs))  # We use the original embeddings for the backward LSTM.

    combined_word_reps = [dy.concatenate([f, b]) for f, b in zip(fwd_word_reps, reversed(bwd_word_reps))]

    # Softmax scores
    W = dy.parameter(W_sm)
    b = dy.parameter(b_sm)
    scores = [dy.affine_transform([b, W, x]) for x in combined_word_reps]

    return scores


def calc_scores_with_previous_tag(words):
    dy.renew_cg()

    word_embs = [LOOKUP[x] for x in words]

    # Transduce all batch elements for the backward LSTM, using the original word embeddings.
    bwd_init = bwdLSTM.initial_state()
    bwd_word_reps = bwd_init.transduce(reversed(word_embs))

    # Softmax scores
    W = dy.parameter(W_sm)
    b = dy.parameter(b_sm)

    scores = []
    # Transduce one by one for the forward LSTM
    fwd_init = fwdLSTM.initial_state()
    s_fwd = fwd_init
    prev_tag = start_tag
    for word, bwd_word_rep in zip(word_embs, reversed(bwd_word_reps)):
        # Concatenate word and tag representation just as training.
        fwd_input = dy.concatenate([word, TAG_LOOKUP[prev_tag]])
        s_fwd = s_fwd.add_input(fwd_input)
        combined_rep = dy.concatenate([s_fwd.output(), bwd_word_rep])
        score = dy.affine_transform([b, W, combined_rep])
        prediction = np.argmax(score)
        prev_tag = prediction
        scores.append(score)

    return scores


# Calculate MLE loss for one example
def calc_loss(scores, tags):
    losses = [dy.pickneglogsoftmax(score, tag) for score, tag in zip(scores, tags)]
    return dy.esum(losses)


# Calculate number of tags correct for one example
def calc_correct(scores, tags):
    correct = [np.argmax(score.npvalue()) == tag for score, tag in zip(scores, tags)]
    return sum(correct)


# Perform training
for ITER in range(100):
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
        words, tags = train[sid]

        if use_teacher_forcing:
            scores = calc_scores_with_tags(words, tags)
        else:
            scores = calc_scores(words)

        loss_exp = calc_loss(scores, tags)
        this_correct += calc_correct(scores, tags)
        this_loss += loss_exp.scalar_value()
        this_words += len(words)
        loss_exp.backward()
        trainer.update()
    # Perform evaluation 
    start = time.time()
    this_sents = this_words = this_loss = this_correct = 0
    for words, tags in dev:
        this_sents += 1

        if use_teacher_forcing:
            scores = calc_scores_with_previous_tag(words)
        else:
            scores = calc_scores(words)

        loss_exp = calc_loss(scores, tags)
        this_correct += calc_correct(scores, tags)
        this_loss += loss_exp.scalar_value()
        this_words += len(words)
    print("dev loss/word=%.4f, acc=%.2f%%, word/sec=%.4f" % (
        this_loss / this_words, 100 * this_correct / this_words, this_words / (time.time() - start)), file=sys.stderr)
