import argparse
import sys
from glob import glob

from requests.exceptions import HTTPError

from ucca.evaluation import evaluate, Scores
from ucca.ioutil import read_files_and_dirs
from uccaapp.download_task import TaskDownloader
from uccaapp.upload_task import TaskUploader

try:
    from simplejson.scanner import JSONDecodeError
except ImportError:
    from json.decoder import JSONDecodeError

desc = """Convert a passage file to JSON format and upload to UCCA-App as a completed task,
then download task from UCCA-App and convert to a passage in standard format again,
then evaluate the result against the original"""


def main(filenames, write, **kwargs):
    uploader = TaskUploader(**kwargs)
    downloader = TaskDownloader(**kwargs)
    scores = []
    try:
        for pattern in filenames:
            filenames = glob(pattern)
            if not filenames:
                raise IOError("Not found: " + pattern)
            for ref in read_files_and_dirs(filenames):
                print("Converting passage " + ref.ID + "... ", end="")
                task = uploader.upload_task(ref)
                guessed = downloader.download_task(task["id"], write=write, **kwargs)
                score = evaluate(guessed, ref, **kwargs)
                print("F1=%.3f" % score.average_f1())
                scores.append(score)
    except HTTPError as e:
        try:
            raise ValueError(e.response.json()) from e
        except JSONDecodeError:
            raise ValueError(e.response.text) from e
    print()
    if len(scores) > 1:
        print("Aggregated scores:")
    Scores.aggregate(scores).print()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=desc)
    TaskUploader.add_arguments(argument_parser)
    argument_parser.add_argument("--write", action="store_true", help="Write converted passage to file")
    TaskDownloader.add_write_arguments(argument_parser)
    main(**vars(argument_parser.parse_args()))
    sys.exit(0)
