import sys

# Usage:
#  kwic.py word < corpus.txt > output.tsv

N = 4

for line in sys.stdin:
  arr = ["<s>"] * N + line.strip().split() + ["<s>"] * N
  for i, w in enumerate(arr):
    if w == sys.argv[1]:
      print("\t".join(arr[i-N:i+N+1]))
