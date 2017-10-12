#!/usr/bin/env python3

import argparse
import glob
import sys

import numpy as np

from ucca import layer0, layer1
from ucca.ioutil import file2passage
from ucca.layer1 import NodeTags
from ucca.textutil import break2sentences

desc = """Prints statistics on UCCA passages
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="files to process")
    argparser.add_argument("-o", "--outfile", help="output file for data")
    args = argparser.parse_args()

    print("id,passages,paragraphs,sentences,nodes,terminals,non-terminals,implicit,linkage,discont,"
          "edges,primary,remote,linkage,parents,children,mult-parents")
    data = []
    for pattern in args.filenames:
        for filename in glob.glob(pattern):
            passage = file2passage(filename)
            terminals = passage.layer(layer0.LAYER_ID).all
            non_terminals = [n for n in passage.layer(layer1.LAYER_ID).all if n.ID != "1.1"]
            non_linkage = [n for n in non_terminals if n.tag != NodeTags.Linkage]
            linkage_nodes = passage.layer(layer1.LAYER_ID).top_linkages
            edges = {e for n in non_terminals for e in n}
            remote = [e for e in edges if e.attrib.get("remote")]
            linkage_edges = [e for n in linkage_nodes for e in n]
            fields = (int(passage.ID),
                      1,
                      len({t.paragraph for t in terminals}),
                      len(break2sentences(passage)),
                      len(terminals) + len(non_terminals),
                      len(terminals),
                      len(non_terminals),
                      len([n for n in non_linkage if n.attrib.get("implicit")]),
                      len(linkage_nodes),
                      len([n for n in non_linkage if n.tag == NodeTags.Foundational and n.discontiguous]),
                      len(edges),
                      len(edges) - len(remote) - len(linkage_edges),
                      len(remote),
                      len(linkage_edges),
                      sum(len([p for p in n.parents if p.ID != "1.1"]) for n in non_linkage),
                      sum(len(n.children) for n in non_linkage),
                      len([n for n in non_linkage if len([p for p in n.parents if p.ID != "1.1"]) > 1]),
                      )
            print(",".join("%d" % f for f in fields))
            data.append(fields)
    data = np.array(data, dtype=int)
    if args.outfile:
        np.savetxt(args.outfile, data[data[:, 0].argsort()], fmt="%i", delimiter="\t")

    sys.exit(0)


if __name__ == '__main__':
    main()
