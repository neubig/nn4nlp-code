#!/bin/bash
set -xe

pip install pypandoc twine
python setup.py sdist
python setup.py bdist_wheel
twine upload --skip-existing dist/*

