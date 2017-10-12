#!/usr/bin/env python3
import argparse
import sys

from ucca.convert import from_json, CONVERTERS, TO_FORMAT
from ucca.ioutil import write_passage
from uccaapp.api import ServerAccessor

desc = """Download task from UCCA-App and convert to a passage in standard format"""


class TaskDownloader(ServerAccessor):
    def download_tasks(self, task_ids, **kwargs):
        for task_id in task_ids:
            yield self.download_task(task_id, **kwargs)

    def download_task(self, task_id, write=True, out_format=None, binary=None, out_dir=None, prefix=None, **kwargs):
        del kwargs
        passage = from_json(self.get_user_task(task_id), all_categories=self.layer["categories"])
        if write:
            write_passage(passage, out_format, binary, out_dir, prefix, TO_FORMAT.get(out_format))
        return passage

    @staticmethod
    def add_arguments(argparser):
        argparser.add_argument("task_ids", nargs="+", type=int, help="IDs of tasks to download and convert")
        TaskDownloader.add_write_arguments(argparser)
        ServerAccessor.add_arguments(argparser)

    @staticmethod
    def add_write_arguments(argparser):
        argparser.add_argument("-f", "--out-format", choices=CONVERTERS, help="output file format (default: UCCA)")
        argparser.add_argument("-o", "--out-dir", default=".", help="output directory")
        argparser.add_argument("-p", "--prefix", default="", help="output filename prefix")
        argparser.add_argument("-b", "--binary", action="store_true", help="write in binary format (.pickle)")


def main(**kwargs):
    list(TaskDownloader(**kwargs).download_tasks(**kwargs))


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=desc)
    TaskDownloader.add_arguments(argument_parser)
    main(**vars(argument_parser.parse_args()))
    sys.exit(0)
