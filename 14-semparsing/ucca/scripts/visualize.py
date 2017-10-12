from argparse import ArgumentParser

import matplotlib.pyplot as plt

from ucca import visualization
from ucca.ioutil import read_files_and_dirs

if __name__ == "__main__":
    argparser = ArgumentParser(description="Visualize the given passages as graphs.")
    argparser.add_argument("passages", nargs="+", help="UCCA passages, given as xml/pickle file names")
    args = argparser.parse_args()
    for passage in read_files_and_dirs(args.passages):
        visualization.draw(passage)
        plt.show()
