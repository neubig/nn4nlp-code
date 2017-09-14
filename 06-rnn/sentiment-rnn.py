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

# Start DyNet and defin trainer
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Define the model
EMB_SIZE = 64
HID_SIZE = 64
W_emb = model.add_lookup_parameters((nwords, EMB_SIZE))  # Word embeddings
fwdLSTM = dy.SimpleRNNBuilder(1, EMB_SIZE, HID_SIZE, model)  # Forward LSTM
bwdLSTM = dy.SimpleRNNBuilder(1, EMB_SIZE, HID_SIZE, model)  # Backward LSTM
W_sm = model.add_parameters((ntags, 2 * HID_SIZE))  # Softmax weights
b_sm = model.add_parameters((ntags))  # Softmax bias


# A function to calculate scores for one value
def calc_scores(words):
    dy.renew_cg()
    word_embs = [dy.lookup(W_emb, x) for x in words]
    fwd_init = fwdLSTM.initial_state()
    fwd_embs = fwd_init.transduce(word_embs)
    bwd_init = bwdLSTM.initial_state()
    bwd_embs = bwd_init.transduce(reversed(word_embs))
    W_sm_exp = dy.parameter(W_sm)
    b_sm_exp = dy.parameter(b_sm)
    return W_sm_exp * dy.concatenate([fwd_embs[-1], bwd_embs[-1]]) + b_sm_exp


for ITER in range(100):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for words, tag in train:
        my_loss = dy.pickneglogsoftmax(calc_scores(words), tag)
        train_loss += my_loss.value()
        my_loss.backward()
        trainer.update()
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))
    # Perform training
    test_correct = 0.0
    for words, tag in dev:
        scores = calc_scores(words).npvalue()
        predict = np.argmax(scores)
        if predict == tag:
            test_correct += 1
    print("iter %r: test acc=%.4f" % (ITER, test_correct / len(dev)))
