from argparse import ArgumentParser

from ucca import constructions
from ucca.ioutil import read_files_and_dirs

if __name__ == "__main__":
    argparser = ArgumentParser(description="Extract linguistic constructions from UCCA corpus.")
    argparser.add_argument("passages", nargs="+", help="the corpus, given as xml/pickle file names")
    constructions.add_argument(argparser, False)
    argparser.add_argument("-v", "--verbose", action="store_true", help="print tagged text for each passage")
    args = argparser.parse_args()
    for passage in read_files_and_dirs(args.passages):
        if args.verbose:
            print("%s:" % passage.ID)
        extracted = constructions.extract_edges(passage, constructions=args.constructions, verbose=args.verbose)
        if any(extracted.values()):
            if not args.verbose:
                print("%s:" % passage.ID)
            for construction, edges in extracted.items():
                if edges:
                    print("  %s:" % construction.description)
                    for edge in edges:
                        print("    %s [%s %s]" % (edge, edge.tag, edge.child))
            print()
