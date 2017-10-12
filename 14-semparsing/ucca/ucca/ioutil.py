"""Input/output utility functions for UCCA scripts."""
import os
import sys
import time
from collections import defaultdict
from xml.etree.ElementTree import ParseError

from ucca.convert import file2passage, passage2file, from_text, to_text, split2segments
from ucca.core import Passage


class LazyLoadedPassages(object):
    """
    Iterable interface to Passage objects that loads files on-the-go and can be iterated more than once
    """
    def __init__(self, files, sentences=False, paragraphs=False, converters=None):
        self.files = files
        self.sentences = sentences
        self.paragraphs = paragraphs
        self.split = self.sentences or self.paragraphs
        self.converters = defaultdict(lambda: from_text) if converters is None else converters
        self._files_iter = None
        self._split_iter = None
        self._file_handle = None
        self._next_index = None

    def __iter__(self):
        self._next_index = 0
        self._files_iter = iter(self.files)
        self._split_iter = None
        self._file_handle = None
        return self

    def __next__(self):
        passage = self._next_passage()
        self._next_index += 1
        return passage

    def _next_passage(self):
        passage = None
        if self._split_iter is None:
            try:
                file = next(self._files_iter)
            except StopIteration:  # Finished iteration
                raise
            if isinstance(file, Passage):  # Not really a file, but a Passage
                passage = file
            else:  # A file
                attempts = 3
                while not os.path.exists(file):
                    if attempts == 0:
                        print("File not found: %s" % file, file=sys.stderr)
                        return next(self)
                    print("Failed reading %s, trying %d more times..." % (file, attempts), file=sys.stderr)
                    time.sleep(5)
                    attempts -= 1
                try:
                    passage = file2passage(file)  # XML or binary format
                except (IOError, ParseError):  # Failed to read as passage file
                    base, ext = os.path.splitext(os.path.basename(file))
                    converter = self.converters[ext.lstrip(".")]
                    self._file_handle = open(file, encoding="utf-8")
                    self._split_iter = iter(converter(self._file_handle, passage_id=base))
            if self.split:
                if self._split_iter is None:
                    self._split_iter = (passage,)
                self._split_iter = iter(s for p in self._split_iter for s in
                                        split2segments(p, is_sentences=self.sentences))
        if self._split_iter is not None:  # Either set before or initialized now
            try:
                # noinspection PyTypeChecker
                passage = next(self._split_iter)
            except StopIteration:  # Finished this converter
                self._split_iter = None
                if self._file_handle is not None:
                    self._file_handle.close()
                    self._file_handle = None
                return next(self)
        return passage

    # The following three methods are implemented to support shuffle;
    # note files are shuffled but there is no shuffling within files, as it would not be efficient.
    # Note also the inconsistency because these access the files while __iter__ accesses individual passages.
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.files[i]

    def __setitem__(self, i, value):
        self.files[i] = value

    def __bool__(self):
        return bool(self.files)


def read_files_and_dirs(files_and_dirs, sentences=False, paragraphs=False, converters=None):
    """
    :param files_and_dirs: iterable of files and/or directories to look in
    :param sentences: whether to split to sentences
    :param paragraphs: whether to split to paragraphs
    :param converters: dict of input format converters to use based on the file extension
    :return: list of (lazy-loaded) passages from all files given,
             plus any files directly under any directory given
    """
    files = list(files_and_dirs)
    files += [os.path.join(d, f) for d in files if os.path.isdir(d) for f in os.listdir(d)]
    files = [f for f in files if not os.path.isdir(f)]
    return LazyLoadedPassages(files, sentences, paragraphs, converters)


def write_passage(passage, output_format, binary, outdir, prefix, converter=None):
    suffix = output_format if output_format and output_format != "ucca" else ("pickle" if binary else "xml")
    outfile = outdir + os.path.sep + prefix + passage.ID + "." + suffix
    print("Writing passage '%s'..." % outfile)
    if output_format is None or output_format in ("ucca", "pickle", "xml"):
        passage2file(passage, outfile, binary=binary)
    else:
        output = "\n".join(line for line in (converter or to_text)(passage))
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(output + "\n")
    return outfile
