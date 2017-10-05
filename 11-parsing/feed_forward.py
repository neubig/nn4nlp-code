from operator import itemgetter
from itertools import count
from collections import Counter, defaultdict
import random
import dynet as dy
import numpy as np
import re
import pdb

#taken from: https://github.com/clab/dynet_tutorial_examples/blob/master/tutorial_transition_parser.py
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
      #we need a null to represent null entries in the stack/buffer
      words.append('NULL')
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

#this represents a head and it's children
#we need to track the children to extract features
class Head:
  def __init__(self, word, rep):
    self.word = word
    self.rep = rep
    self.right_children = []
    self.left_children = []
  def add_child(self, child, side = 'left'):
    if side == 'left':
      self.left_children.append(child)
    elif side == 'right':
      self.right_children.append(child)

WORD_DIM = 64
HIDDEN_DIM = 256
ACTION_DIM = 32

class TransitionParser:
  def __init__(self, model, vocab):
    self.vocab = vocab
    self.pW1 = model.add_parameters((HIDDEN_DIM, WORD_DIM*18))
    self.pb1 = model.add_parameters((HIDDEN_DIM, ))
    self.pW_act = model.add_parameters((NUM_ACTIONS, HIDDEN_DIM))
    self.pb_act = model.add_parameters((NUM_ACTIONS, ))

    self.nwords=vocab.size()
    self.WORDS_LOOKUP = model.add_lookup_parameters((self.nwords, WORD_DIM))

  def extract_features(self, stack, buffer):
    #the top of the stack and the buffer
    #the order of these doesn't matter, so i wasn't too careful
    top_of_stack = stack[-3:]
    top_of_buffer = buffer[-3:]
    while len(top_of_stack) < 3:
      top_of_stack = [Head('NULL', self.NULL_REP)] + top_of_stack
    while len(top_of_buffer) < 3:
      top_of_buffer = [Head('NULL', self.NULL_REP)] + top_of_buffer
    children = []
    grandchildren = []
    for i in range(2):
      head = stack[-i]
      if head.word == 'NULL':
        for i in range(4):
          children.append(Head('NULL', self.NULL_REP))
        for i in range(2):
          grandchildren.append(Head('NULL', self.NULL_REP))
      else:
        #children
        left_two_children = head.left_children[:2]
        while len(left_two_children) < 2:
          left_two_children.append(Head('NULL', self.NULL_REP))
        right_two_children = head.right_children[-2:]
        while len(right_two_children) < 2:
          right_two_children.append(Head('NULL', self.NULL_REP))
        children = children + left_two_children + right_two_children
        #grandchildren
        if len(head.left_children) > 0 and len(head.left_children[0].left_children) > 0:
          grandchildren.append(head.left_children[0].left_children[0])
        else:
          grandchildren.append(Head('NULL', self.NULL_REP))
        if len(head.right_children) > 0 and len(head.right_children[-1].right_children) > 0:
          grandchildren.append(head.right_children[-1].right_children[-1])
        else:
          grandchildren.append(Head('NULL', self.NULL_REP))
    representations = [x.rep for x in top_of_stack + top_of_buffer + children + grandchildren]
    return representations


  # returns an expression of the loss for the sequence of actions
  # (that is, the oracle_actions if present or the predicted sequence otherwise)
  def parse(self, t, oracle_actions=None):
    dy.renew_cg()
    self.NULL_REP = self.WORDS_LOOKUP[self.nwords-1]
    if oracle_actions:
      oracle_actions = list(oracle_actions)
      oracle_actions.reverse()
    toks = list(t)
    toks.reverse()
    stack = []
    buffer = []
    W1 = dy.parameter(self.pW1)
    b1 = dy.parameter(self.pb1)
    W_act = dy.parameter(self.pW_act)
    b_act = dy.parameter(self.pb_act)
    losses = []
    for tok in toks:
      tok_embedding = self.WORDS_LOOKUP[tok]
      buffer.append(Head(self.vocab.i2w[tok], tok_embedding))

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
        representations = self.extract_features(stack, buffer)
        h = dy.cube(W1*dy.concatenate(representations) + b1)
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
        token = buffer.pop()
        stack.append(token)
      else: # one of the reduce actions
        right = stack.pop()
        left = stack.pop()
        head, modifier = (left, right) if action == REDUCE_R else (right, left)
        #add the tokens and their embeddings into the children list
        if action == REDUCE_R:
          head.add_child(modifier, 'right')
        else:
          head.add_child(modifier, 'left')
        stack.append(head)
        if oracle_actions is None:
          print('{0} --> {1}'.format(head.word, modifier.word))

    # the head of the tree that remains at the top of the stack is now the root
    if oracle_actions is None:
      head = stack.pop().word
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
