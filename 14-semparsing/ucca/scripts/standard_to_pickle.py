#!/usr/bin/env python3
import argparse
import os
import sys

from ucca.ioutil import file2passage, passage2file

desc = """Parses an XML in UCCA standard format, and writes them in binary Pickle format.
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument('filenames', nargs='+', help="XML file names to convert")
    argparser.add_argument('-o', '--outdir', default='.', help="output directory")
    args = argparser.parse_args()

    for filename in args.filenames:
        sys.stderr.write("Reading passage '%s'...\n" % filename)
        passage = file2passage(filename)
        basename = os.path.splitext(os.path.basename(filename))[0]
        outfile = args.outdir + os.path.sep + basename + ".pickle"
        sys.stderr.write("Writing file '%s'...\n" % outfile)
        passage2file(passage, outfile, binary=True)

    sys.exit(0)


if __name__ == '__main__':
    main()
