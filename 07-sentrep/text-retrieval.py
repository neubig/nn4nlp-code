from __future__ import print_function
import time

from collections import defaultdict
import random
import math
import sys
import argparse

import dynet as dy
import numpy as np

# format of files: each line is "word1 word2 ..." aligned line-by-line
train_src_file = "../data/parallel/train.ja"
train_trg_file = "../data/parallel/train.en"
dev_src_file = "../data/parallel/dev.ja"
dev_trg_file = "../data/parallel/dev.en"

w2i_src = defaultdict(lambda: len(w2i_src))
w2i_trg = defaultdict(lambda: len(w2i_trg))

def read(fname_src, fname_trg):
    """
    Read parallel files where each line lines up
    """
    with open(fname_src, "r") as f_src, open(fname_trg, "r") as f_trg:
        for line_src, line_trg in zip(f_src, f_trg):
            sent_src = [w2i_src[x] for x in line_src.strip().split()]
            sent_trg = [w2i_trg[x] for x in line_trg.strip().split()]
            yield (sent_src, sent_trg)

# Read the data
train = list(read(train_src_file, train_trg_file))
unk_src = w2i_src["<unk>"]
w2i_src = defaultdict(lambda: unk_src, w2i_src)
unk_trg = w2i_trg["<unk>"]
w2i_trg = defaultdict(lambda: unk_trg, w2i_trg)
nwords_src = len(w2i_src)
nwords_trg = len(w2i_trg)
dev = list(read(dev_src_file, dev_trg_file))

# DyNet Starts
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Model parameters
EMBED_SIZE = 64
HIDDEN_SIZE = 128
BATCH_SIZE = 16

# Lookup parameters for word embeddings
LOOKUP_SRC = model.add_lookup_parameters((nwords_src, EMBED_SIZE))
LOOKUP_TRG = model.add_lookup_parameters((nwords_trg, EMBED_SIZE))

# Word-level BiLSTMs
LSTM_SRC_FWD = dy.LSTMBuilder(1, EMBED_SIZE, HIDDEN_SIZE/2, model)
LSTM_SRC_BWD = dy.LSTMBuilder(1, EMBED_SIZE, HIDDEN_SIZE/2, model)
LSTM_TRG_FWD = dy.LSTMBuilder(1, EMBED_SIZE, HIDDEN_SIZE/2, model)
LSTM_TRG_BWD = dy.LSTMBuilder(1, EMBED_SIZE, HIDDEN_SIZE/2, model)

def encode_sents(look, fwd, bwd, sents):
    embs = [[look[x] for x in sent] for sent in sents]
    return [dy.concatenate([fwd.transduce(x)[-1], bwd.transduce(x)[-1]]) for x in embs]

# Calculate loss for one mini-batch
def calc_loss(sents):
    dy.renew_cg()

    src_fwd = LSTM_SRC_FWD.initial_state()
    src_bwd = LSTM_SRC_BWD.initial_state()
    trg_fwd = LSTM_TRG_FWD.initial_state()
    trg_bwd = LSTM_TRG_BWD.initial_state()

    # Encoding
    src_reps = encode_sents(LOOKUP_SRC, src_fwd, src_bwd, [src for src, trg in sents])
    trg_reps = encode_sents(LOOKUP_TRG, trg_fwd, trg_bwd, [trg for src, trg in sents])

    # Concatenate the sentence representations to a single matrix
    mtx_src = dy.concatenate_cols(src_reps)
    mtx_trg = dy.concatenate_cols(trg_reps)

    # Do matrix multiplication to get a matrix of dot product similarity scores
    sim_mtx = dy.transpose(mtx_src) * mtx_trg

    # Calculate the hinge loss over all dimensions 
    loss = dy.hinge_dim(sim_mtx, list(range(len(sents))), d=1)

    return dy.sum_elems(loss)

# Calculate representations for one corpus
def index_corpus(sents):
    
    # To take advantage of auto-batching, do several at a time
    for sid in range(0, len(sents), BATCH_SIZE):
        dy.renew_cg()

        src_fwd = LSTM_SRC_FWD.initial_state()
        src_bwd = LSTM_SRC_BWD.initial_state()
        trg_fwd = LSTM_TRG_FWD.initial_state()
        trg_bwd = LSTM_TRG_BWD.initial_state()
        
        # Set up the computation graph
        src_exprs = encode_sents(LOOKUP_SRC, src_fwd, src_bwd, [src for src, trg in sents[sid:min(sid+BATCH_SIZE,len(sents))]])
        trg_exprs = encode_sents(LOOKUP_TRG, trg_fwd, trg_bwd, [trg for src, trg in sents[sid:min(sid+BATCH_SIZE,len(sents))]])

        # Perform the forward pass to calculate everything at once
        trg_exprs[-1][1].forward()

        for src_expr, trg_expr in zip(src_exprs, trg_exprs):
            yield (src_expr.npvalue(), trg_expr.npvalue())

# Perform retrieval, and return both scores and ranked order of candidates
def retrieve(src, db_mtx):
    scores = np.dot(db_mtx,src)
    ranks = np.argsort(-scores)
    return ranks, scores

# Perform training
start = time.time()
train_mbs = all_time = dev_time = all_tagged = this_sents = this_loss = 0
for ITER in range(100):
    random.shuffle(train)
    for sid in range(0, len(train), BATCH_SIZE):
        my_size = min(BATCH_SIZE, len(train)-sid)
        train_mbs += 1
        if train_mbs % int(1000/BATCH_SIZE) == 0:
            trainer.status()
            print("loss/sent=%.4f, sent/sec=%.4f" % (this_loss / this_sents, (train_mbs * BATCH_SIZE) / (time.time() - start - dev_time)), file=sys.stderr)
            this_loss = this_sents = 0
        # train on the minibatch
        loss_exp = calc_loss(train[sid:sid+BATCH_SIZE])
        this_loss += loss_exp.scalar_value()
        this_sents += BATCH_SIZE
        loss_exp.backward()
        trainer.update()
    # Perform evaluation 
    dev_start = time.time()
    rec_at_1, rec_at_5, rec_at_10 = 0, 0, 0
    reps = list(index_corpus(dev))
    trg_mtx = np.stack([trg for src, trg in reps])
    for i, (src, trg) in enumerate(reps):
        ranks, scores = retrieve(src, trg_mtx)
        if ranks[0] == i: rec_at_1 += 1
        if i in ranks[:5]: rec_at_5 += 1
        if i in ranks[:10]: rec_at_10 += 1
    dev_time += time.time()-dev_start
    print("epoch %r: dev recall@1=%.2f%% recall@5=%.2f%% recall@10=%.2f%%" % (ITER, rec_at_1/len(dev)*100, rec_at_5/len(dev)*100, rec_at_10/len(dev)*100))
