#!/usr/bin/env python3

import argparse
import os
import sys
from collections import Counter

from ucca import layer1
from ucca.ioutil import file2passage

desc = """Finds edge tags that are empirically always unique: occur at most once in edges per node
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument('-d', '--directory', required=True, help="directory with passage files to process")
    argparser.add_argument('-o', '--outfile', default="data/unique_roles.txt", help="output file for data")
    argparser.add_argument('-D', '--direction', default="out", help="direction of edges to check (out|in)")
    args = argparser.parse_args()

    out = args.direction == "out"
    if not os.path.isdir(args.directory):
        raise Exception("Not a directory: " + args.directory)
    roles = set(tag for name, tag in layer1.EdgeTags.__dict__.items()
                if isinstance(tag, str) and not name.startswith('__'))
    for filename in os.listdir(args.directory):
        sys.stderr.write("Reading passage '%s'...\n" % filename)
        passage = file2passage(args.directory + os.path.sep + filename)
        for node in passage.layer(layer1.LAYER_ID).all:
            counts = Counter(edge.tag for edge in (node if out else node.incoming))
            roles.difference_update(tag for tag, count in counts.items() if count > 1)

    lines = "\n".join(sorted(roles))
    print(lines)
    if args.outfile:
        with open(args.outfile, "w", encoding="utf-8") as f:
            print(lines, file=f)

    sys.exit(0)


if __name__ == '__main__':
    main()
