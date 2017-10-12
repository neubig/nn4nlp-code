#!/usr/bin/env python3
"""
The evaluation software for UCCA layer 1.
"""

from optparse import OptionParser

from scripts import ucca_db
from ucca import convert
from ucca.evaluation import evaluate


##############################################################################
# Returns the command line parser.
##############################################################################
def cmd_line_parser():
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("--db", "-d", dest="db_filename",
                          action="store", type="string",
                          help="the db file name")
    opt_parser.add_option("--host", "--hst", dest="host",
                          action="store", type="string",
                          help="the host name")
    opt_parser.add_option("--pid", "-p", dest="pid", action="store",
                          type="int", help="the passage ID")
    opt_parser.add_option("--from_xids", "-x", dest="from_xids",
                          action="store_true", help="interpret the ref \
                          and the guessed parameters as Xids in the db")
    opt_parser.add_option("--guessed", "-g", dest="guessed", action="store",
                          type="string", help="if a db is defined - \
                          the username for the guessed annotation; \
                          else - the xml file name for the guessed annotation")
    opt_parser.add_option("--ref", "-r", dest="ref", action="store",
                          type="string", help="if a db is defined - \
                          the username for the reference annotation; else - \
                          the xml file name for the reference annotation")
    opt_parser.add_option("--units", "-u", dest="units", action="store_true",
                          help="the units the annotations have in common, \
                          and those each has separately")
    opt_parser.add_option("--fscore", "-f", dest="fscore", action="store_true",
                          help="outputs the traditional P,R,F \
                          instead of the scene structure evaluation")
    opt_parser.add_option("--debug", dest="debug", action="store_true",
                          help="run in debug mode")
    opt_parser.add_option("--errors", "-e", dest="errors", action="store_true",
                          help="prints the error distribution\
                          according to its frequency")
    return opt_parser


def main():
    opt_parser = cmd_line_parser()
    (options, args) = opt_parser.parse_args()
    if len(args) > 0:
        opt_parser.error("all arguments must be flagged")

    if (options.guessed is None) or (options.ref is None) or (options.db_filename is None):
        opt_parser.error("missing arguments. type --help for help.")
    if options.pid is not None and options.from_xids is not None:
        opt_parser.error("inconsistent parameters. \
        you can't have both a pid and from_xids paramters.")

    keys = [options.guessed, options.ref]
    if options.from_xids:
        xmls = ucca_db.get_by_xids(options.db_filename, options.host, keys)
    else:
        xmls = ucca_db.get_xml_trees(options.db_filename, options.host,
                                     options.pid, keys)

    guessed, ref = [convert.from_site(x) for x in xmls]
    if options.units or options.fscore or options.errors:
        evaluate(guessed, ref,
                 units=options.units, fscore=options.fscore, errors=options.errors, verbose=True)


if __name__ == '__main__':
    main()
