import argparse
from posix import mkdir

import os
import re
from shutil import copyfile

desc = """Split a directory of files into "train", "dev" and "test" directories.
All files not in either "train" or "dev" will go into "test".
"""
TRAIN_DEFAULT = 300
DEV_DEFAULT = 34
# TEST on all the rest


def copy(src, dest, link=False):
    if link:
        try:
            os.symlink(src, dest)
        except (NotImplementedError, OSError):
            copyfile(src, dest)
    else:
        copyfile(src, dest)


def numeric(s):
    try:
        return int(re.findall("([0-9]+)", s)[-1])
    except (ValueError, IndexError):
        return s


def not_split_dir(filename):
    return filename not in ("train", "dev", "test")


def split_passages(directory, train, dev, link, quiet=False):
    filenames = sorted(filter(not_split_dir, os.listdir(directory)), key=numeric)
    assert filenames, "No files to split"
    assert train + dev <= len(filenames), "Not enough files to split: %d+%d>%d" % (train, dev, len(filenames))
    directory = os.path.abspath(directory)
    if not directory.endswith(os.sep):
        directory += os.sep
    for subdirectory in "train", "dev", "test":
        if not os.path.exists(directory + subdirectory):
            mkdir(directory + subdirectory)
    print("%d files to split: %d/%d/%d" % (len(filenames), train, dev, len(filenames) - train - dev))
    print_format = "Creating link in %s to: " if link else "Copying to %s: "
    if not quiet:
        print(print_format % "train", end="", flush=True)
    for f in filenames[:train]:
        copy(directory + f, directory + "train" + os.sep + f, link)
        if not quiet:
            print(f, end=" ", flush=True)
    if not quiet:
        print()
        print(print_format % "dev", end="", flush=True)
    for f in filenames[train:train + dev]:
        copy(directory + f, directory + "dev" + os.sep + f, link)
        if not quiet:
            print(f, end=" ", flush=True)
    if not quiet:
        print()
        print(print_format % "test", end="", flush=True)
    for f in filenames[train + dev:]:
        copy(directory + f, directory + "test" + os.sep + f, link)
        if not quiet:
            print(f, end=" ", flush=True)
    if not quiet:
        print()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=desc)
    argparser.add_argument("directory", default=".", nargs="?", help="directory to split (default: current directory)")
    argparser.add_argument("-t", "--train", type=int, default=TRAIN_DEFAULT,
                           help="size of train split (default: %d)" % TRAIN_DEFAULT)
    argparser.add_argument("-d", "--dev", type=int, default=DEV_DEFAULT,
                           help="size of dev split (default: %d)" % DEV_DEFAULT)
    argparser.add_argument("-l", "--link", action="store_true", help="create symbolic link instead of copying")
    argparser.add_argument("-q", "--quiet", action="store_true", help="less output")
    args = argparser.parse_args()

    split_passages(args.directory, args.train, args.dev, link=args.link, quiet=args.quiet)
