import os
import oracle
from ucca import diffutil, ioutil, textutil, layer1, evaluation
from pdb import set_trace


files = ['../ucca_corpus_pickle/' + f for f in os.listdir('../ucca_corpus_pickle')]
passages = list(ioutil.read_files_and_dirs(files))

passage = passages[0]
ora = oracle.Oracle(passage)
set_trace()