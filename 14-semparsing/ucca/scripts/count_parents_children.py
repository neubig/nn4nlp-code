#!/usr/bin/env python3

import argparse
import glob
import sys
from collections import Counter, defaultdict

from ucca.ioutil import file2passage
from ucca import layer1

desc = """Parses XML files in UCCA standard format, and creates a histogram for the number of parents per unit.
"""


def plot_histogram(counter, label, plot=None):
    import matplotlib.pyplot as plt
    plt.figure()
    nums = list(counter.keys())
    counts = list(counter.values())
    indices = range(len(counts))
    bars = plt.bar(indices, counts, align="center")
    plt.xticks(indices, nums)
    top = 1.06 * max(counts)
    plt.ylim(min(counts), top)
    plt.xlabel("number of %s" % label)
    plt.ylabel("count")
    for bar in bars:
        count = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., count, "%.1f%%" % (100.0 * count / sum(counts)),
                 ha="center", va="bottom")
    if plot:
        plt.savefig(plot + "histogram_" + label + ".png")
    else:
        plt.show()


def plot_pie(counter, label, plot=None):
    import matplotlib.pyplot as plt
    plt.figure()
    nums = list(counter.keys())
    counts = list(counter.values())
    plt.pie(counts, labels=nums, autopct="%1.1f%%",
            counterclock=True, wedgeprops={"edgecolor": "white"})
    plt.axis("equal")
    if plot:
        plt.savefig(plot + "pie_" + label + ".png")
    else:
        plt.show()


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="file names to analyze")
    argparser.add_argument("-o", "--outfile", default="data/counts_",
                           help="output file prefix for histogram")
    argparser.add_argument("-p", "--plot", default="data/plot_",
                           help="output file prefix for plot image file")
    args = argparser.parse_args()

    histograms = defaultdict(Counter)
    for pattern in args.filenames:
        for filename in glob.glob(pattern):
            sys.stderr.write("Reading passage '%s'...\n" % filename)
            passage = file2passage(filename)
            for node in passage.layer(layer1.LAYER_ID).all:
                if node.ID != "1.1":  # Exclude the root node
                    histograms["parents"][clip(node.incoming, 3)] += 1
                    histograms["children"][clip(node.outgoing, 7)] += 1

    for label, counter in histograms.items():
        handle = open(args.outfile + label + ".txt", "w", encoding="utf-8") if args.outfile else sys.stdout
        handle.writelines(["%s\t%d\n" % (num, count) for num, count in counter.items()])
        if handle is not sys.stdout:
            handle.close()
        # noinspection PyBroadException
        try:
            plot_histogram(counter, label, plot=args.plot)
            plot_pie(counter, label, plot=args.plot)
        except:
            pass

    sys.exit(0)


def clip(l, m):
    return len(l) if len(l) <= m else ">%d" % m


if __name__ == "__main__":
    main()
