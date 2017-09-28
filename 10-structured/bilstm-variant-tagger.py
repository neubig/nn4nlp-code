from __future__ import print_function
import time

from collections import defaultdict
import random
import sys
import argparse

import dynet as dy
import numpy as np

parser = argparse.ArgumentParser(description='BiLSTM variants.')
parser.add_argument('--teacher', action='store_true')
parser.add_argument('--perceptron', action='store_true')
parser.add_argument('--cost', action='store_true')
parser.add_argument('--hinge', action='store_true')
parser.add_argument('--schedule', action='store_true')

args = parser.parse_args()
use_teacher_forcing = args.teacher
use_structure_perceptron = args.perceptron
use_cost_augmented = args.cost
use_hinge = args.hinge
use_schedule = args.schedule

print("Training BiLSTM %s teacher forcing (%s schedule), %s structured perceptron loss, %s augmented cost, %s margin."
      % ("with" if use_teacher_forcing else "without",
         "with" if use_schedule else "without",
         "with" if use_structure_perceptron else "without",
         "with" if use_cost_augmented else "without",
         "with" if use_hinge else "without"
         )
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


class AlwaysTrueSampler:
    """
    An always true sampler, only sample fromtrue distribution.
    """

    def sample_true(self):
        return True

    def decay(self):
        pass


class ScheduleSampler:
    """
    A linear schedule sampler.
    """

    def __init__(self, start_rate=1, min_rate=0.2, decay_rate=0.1):
        self.min_rate = min_rate
        self.iter = 0
        self.decay_rate = decay_rate
        self.start_rate = start_rate
        self.reach_min = False
        self.sample_rate = start_rate

    def decay_func(self):
        if not self.reach_min:
            self.sample_rate = self.start_rate - self.iter * self.decay_rate
            if self.sample_rate < self.min_rate:
                self.reach_min = True
                self.sample_rate = self.min_rate

    def decay(self):
        self.iter += 1
        self.decay_func()
        print("Sample rate is now %.2f" % self.sample_rate)

    def sample_true(self):
        return random.random() < self.sample_rate


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

if use_schedule:
    sampler = ScheduleSampler()
else:
    sampler = AlwaysTrueSampler()

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
    """
    Calculate scores using BiLSTM.
    :param words:
    :return:
    """
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


def calc_scores_with_previous_tag(words, referent_tags=None):
    """
    Calculate scores using previous tag as input. If the referent tags are provided, then we will sample from previous
    referent tag or previous system prediction.
    :param words:
    :param referent_tags:
    :return:
    """
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

    index = 0
    for word, bwd_word_rep in zip(word_embs, reversed(bwd_word_reps)):
        # Concatenate word and tag representation just as training.
        fwd_input = dy.concatenate([word, TAG_LOOKUP[prev_tag]])
        s_fwd = s_fwd.add_input(fwd_input)
        combined_rep = dy.concatenate([s_fwd.output(), bwd_word_rep])
        score = dy.affine_transform([b, W, combined_rep])
        prediction = np.argmax(score.npvalue())

        if referent_tags:
            if sampler.sample_true():
                prev_tag = referent_tags[index]
            else:
                prev_tag = prediction
            index += 1
        else:
            prev_tag = prediction

        scores.append(score)

    return scores


def mle(scores, tags):
    losses = [dy.pickneglogsoftmax(score, tag) for score, tag in zip(scores, tags)]
    return dy.esum(losses)


def hamming_cost(predictions, reference):
    return sum(p != r for p, r in zip(predictions, reference))


def calc_sequence_score(scores, tags):
    return dy.esum([score[tag] for score, tag in zip(scores, tags)])


def hamming_augmented_decode(scores, reference):
    """
    Local decoding with hamming cost.
    :param scores: Local decoding scores.
    :param reference: Referent tag result.
    :return:
    """
    augmented_result = []
    for score, referent_tag in zip(scores, reference):
        origin_scores = score.npvalue()
        cost = np.ones(origin_scores.shape)
        cost[referent_tag] = 0
        augmented_result.append(np.argmax(np.add(origin_scores, cost)))
    return augmented_result


def perceptron_loss(scores, reference):
    if use_cost_augmented:
        predictions = hamming_augmented_decode(scores, reference)
    else:
        predictions = [np.argmax(score.npvalue()) for score in scores]

    margin = dy.scalarInput(-2)

    if predictions != reference:
        reference_score = calc_sequence_score(scores, reference)
        prediction_score = calc_sequence_score(scores, predictions)
        if use_cost_augmented:
            # One could actually get the hamming augmented value during decoding, but we didn't do it here for
            # demonstration purpose.
            hamming = dy.scalarInput(hamming_cost(predictions, reference))
            loss = prediction_score + hamming - reference_score
        else:
            loss = prediction_score - reference_score

        if use_hinge:
            loss = dy.emax([dy.scalarInput(0), loss - margin])

        return loss
    else:
        return dy.scalarInput(0)


# Calculate MLE loss for one example
def calc_loss(scores, tags):
    if use_structure_perceptron:
        return perceptron_loss(scores, tags)
    else:
        return mle(scores, tags)


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
        # choose whether to use teacher forcing
        if use_teacher_forcing:
            scores = calc_scores_with_previous_tag(words, tags)
        else:
            scores = calc_scores(words)
        loss_exp = calc_loss(scores, tags)
        this_correct += calc_correct(scores, tags)
        this_loss += loss_exp.scalar_value()
        this_words += len(words)
        loss_exp.backward()
        trainer.update()
    # Decay the schedule sampler if using schedule sampling.
    sampler.decay()
    # Perform evaluation
    start = time.time()
    this_sents = this_words = this_loss = this_correct = 0
    for words, tags in dev:
        this_sents += 1
        # choose whether to use teacher forcing
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
