Universal Conceptual Cognitive Annotation
============================
UCCA is a linguistic framework for semantic annotation, whose details
are available at [the following paper](http://www.cs.huji.ac.il/~oabend/papers/ucca_acl.pdf):

    @inproceedings{abend2013universal,
      author={Abend, Omri  and  Rappoport, Ari},
      title={{U}niversal {C}onceptual {C}ognitive {A}nnotation ({UCCA})},
      booktitle={Proc. of ACL},
      month={August},
      year={2013},
      pages={228--238},
      url={http://aclweb.org/anthology/P13-1023}
    }

This Python 3 package provides an API to the UCCA annotation and tools to
manipulate and process it. Its main features are conversion between different
representations of UCCA annotations, and rich objects for all of the linguistic
relations which appear in the theoretical framework (see `core`, `layer0`, `layer1`
and `convert` modules under the `ucca` package).

The `scripts` package contains various utilities for processing passage files.


Authors
------
* Amit Beka: amit.beka@gmail.com
* Daniel Hershcovich: danielh@cs.huji.ac.il


License
-------
This package is licensed under the GPLv3 or later license.

[![Build Status](https://travis-ci.org/danielhers/ucca.svg?branch=master)](https://travis-ci.org/danielhers/ucca)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/danielhers/ucca?svg=true)](https://ci.appveyor.com/project/danielh/ucca)
[![PyPI version](https://badge.fury.io/py/UCCA.svg)](https://badge.fury.io/py/UCCA)
