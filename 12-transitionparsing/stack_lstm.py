from operator import itemgetter
from itertools import count
from collections import Counter, defaultdict
import random
import dynet as dy
import numpy as np
import re

#taken from: https://github.com/clab/dynet_tutorial_examples/blob/master/tutorial_transition_parser.py
#adopted for python 3
# actions the parser can take
SHIFT = 0
REDUCE_L = 1
REDUCE_R = 2
NUM_ACTIONS = 3

class Vocab:
  def __init__(self, w2i):
    self.w2i = dict(w2i)
    self.i2w = {i:w for w,i in w2i.items()}
  @classmethod
  def from_list(cls, words):
    w2i = {}
    idx = 0
    for word in words:
      w2i[word] = idx
      idx += 1
    return Vocab(w2i)
  @classmethod
  def from_file(cls, vocab_fname):
    words = []
    with open(vocab_fname) as fh:
      for line in fh:
        line.strip()
        word, count = line.split()
        words.append(word)
    return Vocab.from_list(words)

  def size(self): return len(self.w2i.keys())

def read_oracle(fname, vw, va):
  with open(fname) as fh:
    for line in fh:
      line = line.strip()
      ssent, sacts = re.split(r' \|\|\| ', line)
      sent = [vw.w2i[x] for x in ssent.split()]
      acts = [va.w2i[x] for x in sacts.split()]
      yield (sent, acts)

WORD_DIM = 64
LSTM_DIM = 64
ACTION_DIM = 32

class TransitionParser:
  def __init__(self, model, vocab):
    self.vocab = vocab
    self.pW_comp = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
    self.pb_comp = model.add_parameters((LSTM_DIM, ))
    self.pW_s2h = model.add_parameters((LSTM_DIM, LSTM_DIM * 2))
    self.pb_s2h = model.add_parameters((LSTM_DIM, ))
    self.pW_act = model.add_parameters((NUM_ACTIONS, LSTM_DIM))
    self.pb_act = model.add_parameters((NUM_ACTIONS, ))

    # layers, in-dim, out-dim, model
    self.buffRNN = dy.LSTMBuilder(1, WORD_DIM, LSTM_DIM, model)
    self.stackRNN = dy.LSTMBuilder(1, WORD_DIM, LSTM_DIM, model)
    self.pempty_buffer_emb = model.add_parameters((LSTM_DIM,))
    nwords=vocab.size()
    self.WORDS_LOOKUP = model.add_lookup_parameters((nwords, WORD_DIM))

  # returns an expression of the loss for the sequence of actions
  # (that is, the oracle_actions if present or the predicted sequence otherwise)
  def parse(self, t, oracle_actions=None):
    dy.renew_cg()
    if oracle_actions:
      oracle_actions = list(oracle_actions)
      oracle_actions.reverse()
    stack_top = self.stackRNN.initial_state()
    toks = list(t)
    toks.reverse()
    stack = []
    cur = self.buffRNN.initial_state()
    buffer = []
    empty_buffer_emb = dy.parameter(self.pempty_buffer_emb)
    W_comp = dy.parameter(self.pW_comp)
    b_comp = dy.parameter(self.pb_comp)
    W_s2h = dy.parameter(self.pW_s2h)
    b_s2h = dy.parameter(self.pb_s2h)
    W_act = dy.parameter(self.pW_act)
    b_act = dy.parameter(self.pb_act)
    losses = []
    for tok in toks:
      tok_embedding = self.WORDS_LOOKUP[tok]
      cur = cur.add_input(tok_embedding)
      buffer.append((cur.output(), tok_embedding, self.vocab.i2w[tok]))

    while not (len(stack) == 1 and len(buffer) == 0):
      # based on parser state, get valid actions
      valid_actions = []
      if len(buffer) > 0:  # can only reduce if elements in buffer
        valid_actions += [SHIFT]
      if len(stack) >= 2:  # can only shift if 2 elements on stack
        valid_actions += [REDUCE_L, REDUCE_R]

      # compute probability of each of the actions and choose an action
      # either from the oracle or if there is no oracle, based on the model
      action = valid_actions[0]
      log_probs = None
      if len(valid_actions) > 1:
        buffer_embedding = buffer[-1][0] if buffer else empty_buffer_emb
        stack_embedding = stack[-1][0].output() # the stack has something here
        parser_state = dy.concatenate([buffer_embedding, stack_embedding])
        h = dy.tanh(W_s2h * parser_state + b_s2h)
        logits = W_act * h + b_act
        log_probs = dy.log_softmax(logits, valid_actions)
        if oracle_actions is None:
          action = max(enumerate(log_probs.vec_value()), key=itemgetter(1))[0]
      if oracle_actions is not None:
        action = oracle_actions.pop()
        if log_probs is not None:
          # append the action-specific loss
          losses.append(dy.pick(log_probs, action))

      # execute the action to update the parser state
      if action == SHIFT:
        _, tok_embedding, token = buffer.pop()
        stack_state, _ = stack[-1] if stack else (stack_top, '<TOP>')
        stack_state = stack_state.add_input(tok_embedding)
        stack.append((stack_state, token))
      else: # one of the reduce actions
        right = stack.pop()
        left = stack.pop()
        head, modifier = (left, right) if action == REDUCE_R else (right, left)
        top_stack_state, _ = stack[-1] if stack else (stack_top, '<TOP>')
        head_rep, head_tok = head[0].output(), head[1]
        mod_rep, mod_tok = modifier[0].output(), modifier[1]
        composed_rep = dy.rectify(W_comp * dy.concatenate([head_rep, mod_rep]) + b_comp)
        top_stack_state = top_stack_state.add_input(composed_rep)
        stack.append((top_stack_state, head_tok))
        if oracle_actions is None:
          print('{0} --> {1}'.format(head_tok, mod_tok))

    # the head of the tree that remains at the top of the stack is now the root
    if oracle_actions is None:
      head = stack.pop()[1]
      print('ROOT --> {0}'.format(head))
    return -dy.esum(losses) if losses else None

acts = ['SHIFT', 'REDUCE_L', 'REDUCE_R']
vocab_acts = Vocab.from_list(acts)

vocab_words = Vocab.from_file('../data/parsing/shift_reduce/vocab.txt')
train = list(read_oracle('../data/parsing/shift_reduce/small-train.unk.txt', vocab_words, vocab_acts))
dev = list(read_oracle('../data/parsing/shift_reduce/small-dev.unk.txt', vocab_words, vocab_acts))

model = dy.Model()
trainer = dy.AdamTrainer(model)

tp = TransitionParser(model, vocab_words)

i = 0
for epoch in range(5):
  words = 0
  total_loss = 0.0
  for (s,a) in train:
    loss = tp.parse(s, a)
    words += len(s)
    if loss is not None:
      total_loss += loss.scalar_value()
      loss.backward()
      trainer.update()
    e = float(i) / len(train)
    if i % 50 == 0:
      print('epoch {}: per-word loss: {}'.format(e, total_loss / words))
      words = 0
      total_loss = 0.0
    if i % 500 == 0:
      tp.parse(dev[209][0])
      dev_words = 0
      dev_loss = 0.0
      for (ds, da) in dev:
        loss = tp.parse(ds, da)
        dev_words += len(ds)
        if loss is not None:
          dev_loss += loss.scalar_value()
      print('[validation] epoch {}: per-word loss: {}'.format(e, dev_loss / dev_words))
    i += 1
