# This visualizer is based off of 
# http://nlp.yvespeirsman.be/blog/visualizing-word-embeddings-with-tsne/

import pylab as Plot
import numpy as np
import argparse
from tsne import tsne # from http://lvdmaaten.github.io/tsne/
import pdb

parser = argparse.ArgumentParser(description='Visualize word embeddings using TSNE')
parser.add_argument('vector_file', type=str, help='location of the word vector file')
parser.add_argument('label_file', type=str, help='location of the word vector file')
parser.add_argument('--target_words', dest='target_words', type=str, default=None, help='a list of words to display (if none, shows 1000 random words')

args = parser.parse_args()

#read the datafile, with the option for a seperate labels file
def read_data(vector_file_path, labels_file_path=None):
  vocab = []
  word_vectors = []

  with open(labels_file_path) as sample_file:
    for line in sample_file:
      vocab.append(line.strip())
  with open(vector_file_path) as vector_file:
    for line in vector_file:
      line = line.strip()
      word_vector = line.split()
      word_vectors.append([float(i) for i in word_vector])
  return np.array(word_vectors), vocab

def display_data(word_vectors, words, target_words=None):
  target_matrix = word_vectors.copy()
  if target_words:
    target_words = [line.strip().lower() for line in open(target_words)][:2000]
    rows = [words.index(word) for word in target_words if word in words]
    target_matrix = target_matrix[rows,:]
  else:
    rows = np.random.choice(len(word_vectors), size=1000, replace=False)
    target_matrix = target_matrix[rows,:]
  reduced_matrix = tsne(target_matrix, 2);

  Plot.figure(figsize=(200, 200), dpi=100)
  max_x = np.amax(reduced_matrix, axis=0)[0]
  max_y = np.amax(reduced_matrix, axis=0)[1]
  Plot.xlim((-max_x,max_x))
  Plot.ylim((-max_y,max_y))

  Plot.scatter(reduced_matrix[:, 0], reduced_matrix[:, 1], 20);

  for row_id in range(0, len(rows)):
      target_word = words[rows[row_id]]
      x = reduced_matrix[row_id, 0]
      y = reduced_matrix[row_id, 1]
      Plot.annotate(target_word, (x,y))
  Plot.savefig("word_vectors.png");

if __name__ == "__main__":
  X, labels = read_data(args.vector_file, args.label_file)
  display_data(X, labels, args.target_words)

