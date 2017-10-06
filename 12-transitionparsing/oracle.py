import pdb
class Word:
	def __init__(self, word, location, head = -1):
		self.word = word
		self.location = location
		self.head = head
		self.rightmost_child = -1

#abreviated conll file like gold.dev.txt
def read_abrv(file_name):
	f = [l.strip() for l in open(file_name)]
	sents = []
	sent = []
	loc = 0
	for line in f:
		if line != '':
			line = line.split()
			word = line[0]
			head = int(line[1])
			sent.append(Word(word, loc, head))
			loc += 1
		else:
			for word in sent:
				if word.location > sent[word.head].rightmost_child:
					sent[word.head].rightmost_child = word.location
			sents.append(sent)
			sent = []
			loc = 0
	return sents

sentences = read_abrv('../data/parsing/gold.txt')
actions_for_sents = []
f = open('../data/parsing/output.txt', 'w')
for sent in sentences:
	stack, buffer = [], []
	acts = []
	for word in sent:
		buffer.append(word)
	buffer = list(reversed(buffer))
	while len(buffer) > 0 or len(stack) > 1:
		if len(stack) < 2:
			stack.append(buffer.pop())
			acts.append('SHIFT')
		elif stack[-1].head == stack[-2].location  and (len(buffer) == 0 or stack[-1].rightmost_child < buffer[-1].location or stack[-2].rightmost_child == -1):
			acts.append('REDUCE_R')
			stack.pop()
		elif stack[-2].head == stack[-1].location and (len(buffer) == 0 or stack[-2].rightmost_child < buffer[-1].location or stack[-2].rightmost_child == -1):
			acts.append('REDUCE_L')
			temp = stack.pop()
			stack.pop()
			stack.append(temp)
		elif len(buffer) > 0:
			stack.append(buffer.pop())
			acts.append('SHIFT')
		else:
			break
	actions_for_sents.append(acts)
	f.write(' '.join([s.word for s in sent]) + ' ||| ' + ' '.join(acts) + '\n')

f.close()

