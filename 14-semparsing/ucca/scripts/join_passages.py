#!/usr/bin/env python3

import argparse
import glob
import sys

from collections import defaultdict

import ucca.convert
from ucca.ioutil import passage2file, file2passage

desc = """Parses XML/pickle files in UCCA standard format, and writes a single passage.
"""


def main():
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("filenames", nargs="+", help="passage file names to join")
    argparser.add_argument("-o", "--outdir", default=".", help="output directory")
    argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
    argparser.add_argument("-r", "--remarks", action="store_true", help="annotate original IDs")
    argparser.add_argument("-b", "--binary", action="store_true", help="write in pickle binary format (.pickle)")
    argparser.add_argument("-j", "--join-by-prefix", action="store_true",
                           help="join each set of passages whose IDs share all but the last 3 characters")
    args = argparser.parse_args()

    passages = [file2passage(filename) for pattern in args.filenames for filename in sorted(glob.glob(pattern))]
    if args.join_by_prefix:
        subsets = defaultdict(list)
        for passage in passages:
            subsets[passage.ID[:-3]].append(passage)
    else:
        subsets = {passages[0].ID: passages}
    for passage_id, subset in sorted(subsets.items()):
        sys.stderr.write("Joining passages " + ", ".join(passage.ID for passage in subset) + "\n")
        joined = ucca.convert.join_passages(passages, passage_id=passage_id, remarks=args.remarks)
        outfile = "%s/%s.%s" % (args.outdir, args.prefix + joined.ID, "pickle" if args.binary else "xml")
        sys.stderr.write("Writing joined passage file '%s'...\n" % outfile)
        passage2file(joined, outfile, args.binary)

    sys.exit(0)


if __name__ == '__main__':
    main()
