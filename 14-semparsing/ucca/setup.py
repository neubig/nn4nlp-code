#!/usr/bin/env python

from setuptools import setup, find_packages

try:
    import pypandoc
    try:
        pypandoc.convert_file("README.md", "rst", outputfile="README.rst")
    except (IOError, ImportError, RuntimeError):
        pass
    long_description = pypandoc.convert_file("README.md", "rst")
except (IOError, ImportError, RuntimeError):
    long_description = ""


setup(name="UCCA",
      version="1.0.11",
      install_requires=["spacy", "requests"],
      extras_require={"visualize": ["matplotlib", "networkx"]},
      description="Universal Conceptual Cognitive Annotation",
      long_description=long_description,
      author="Daniel Hershcovich",
      author_email="danielh@cs.huji.ac.il",
      url="https://github.com/huji-nlp/ucca",
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3.6",
          "Topic :: Text Processing :: Linguistic",
          "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
      ],
      packages=find_packages(),
      )
