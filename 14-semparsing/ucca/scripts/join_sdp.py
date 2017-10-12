#!/usr/bin/env python3

import argparse
import glob
import os
import sys

desc = """Combines several SDP parsed files to one.
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+",
                           help="SDP file names to join")
    argparser.add_argument("-o", "--outfile",
                           help="output filename (standard output if unspecified)")
    argparser.add_argument("-H", "--header", default="SDP 2015",
                           help="first line in the file, not including prefix")
    argparser.add_argument("-p", "--prefix", default="#",
                           help="prefix for comment lines")
    args = argparser.parse_args()

    lines = [args.prefix + args.header + "\n"]
    for pattern in args.filenames:
        filenames = sorted(glob.glob(pattern))
        if not filenames:
            raise IOError("Not found: " + pattern)
        for filename in filenames:
            base = os.path.basename(os.path.splitext(filename)[0])
            lines.append(args.prefix + base + "\n")
            with open(filename, encoding="utf-8") as f:
                lines += f.readlines()
        f = sys.stdout if args.outfile is None else open(args.outfile, "w", encoding="utf-8")
        f.writelines(lines)
        if args.outfile is not None:
            f.close()

    sys.exit(0)


if __name__ == '__main__':
    main()
