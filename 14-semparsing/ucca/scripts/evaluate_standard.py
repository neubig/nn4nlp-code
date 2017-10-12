#!/usr/bin/env python3
"""
The evaluation script for UCCA layer 1.
"""
import sys
from argparse import ArgumentParser

from ucca import evaluation, constructions, ioutil


if __name__ == "__main__":
    argparser = ArgumentParser(description="Compare two UCCA passages or two directories containing passage files.")
    argparser.add_argument("guessed", help="xml/pickle file name for the guessed annotation, or directory of files")
    argparser.add_argument("ref", help="xml/pickle file name for the reference annotation, or directory of files")
    argparser.add_argument("-u", "--units", action="store_true",
                           help="the units the annotations have in common, and those each has separately")
    argparser.add_argument("-f", "--fscore", action="store_true",
                           help="outputs the traditional P,R,F instead of the scene structure evaluation")
    argparser.add_argument("-e", "--errors", action="store_true",
                           help="prints the error distribution according to its frequency")
    argparser.add_argument("--no-normalize", dest="normalize", action="store_false",
                           help="do not normalize passages before evaluation")
    argparser.add_argument("--out-file", help="file to write results for each evaluated passage to, in CSV format")
    argparser.add_argument("--summary-file", help="file to write aggregated results to, in CSV format")
    group = argparser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", action="store_true",
                       help="prints the results for every single pair (always true if there is only one pair)")
    group.add_argument("-q", "--quiet", action="store_true", help="do not print anything")
    constructions.add_argument(argparser)
    args = argparser.parse_args()

    if not (args.units or args.fscore or args.errors):
        argparser.error("At least one of -u, -f or -e is required.")

    guessed, ref = [ioutil.read_files_and_dirs((x,)) for x in (args.guessed, args.ref)]
    if len(guessed) != len(ref):
        raise ValueError("Number of passages to compare does not match: %d != %d" % (len(guessed), len(ref)))
    if len(guessed) > 1:
        guessed_by_id = {}
        for g in guessed:
            sys.stdout.write("\rReading %s..." % g.ID)
            sys.stdout.flush()
            guessed_by_id[g.ID] = g
        ids = [p.ID for p in ref]
        try:
            guessed = [guessed_by_id[i] for i in ids]
        except KeyError as e:
            raise ValueError("Passage IDs do not match") from e
    results = []
    for g, r in zip(guessed, ref):
        if len(guessed) > 1:
            sys.stdout.write("\rEvaluating %s%s" % (g.ID, ":" if args.verbose else "..."))
            sys.stdout.flush()
        if args.verbose:
            print()
        result = evaluation.evaluate(g, r, constructions=args.constructions, units=args.units, fscore=args.fscore,
                                     errors=args.errors, verbose=args.verbose or len(guessed) == 1,
                                     normalize=args.normalize)
        if args.verbose:
            print("Average labeled F1 score: %.3f\n" % result.average_f1())
        results.append(result)
    summary = evaluation.Scores.aggregate(results)
    if len(results) > 1:
        if args.verbose:
            print("Aggregated scores:")
        else:
            print(end="\r")
            if not args.quiet:
                summary.print()
        if not args.quiet:
            print("Average labeled F1 score: %.3f" % summary.average_f1())
    args_constructions = summary.evaluators
    if args.out_file:
        with open(args.out_file, "w", encoding="utf-8") as f:
            print(",".join(summary.titles()), file=f)
            for result in results:
                print(",".join(result.fields()), file=f)
    if args.summary_file:
        with open(args.summary_file, "w", encoding="utf-8") as f:
            print(",".join(summary.titles()), file=f)
            print(",".join(summary.fields()), file=f)
