`ucca` package
====================

List of Modules
---------------
1. `constructions` -- provides methods for extracting linguistic constructions from text
1. `convert` -- provides functions to convert between the UCCA objects (Pythonic)
to site annotation XML, standard XML representation and text
1. `core` -- provides the basic objects of UCCA relations: `Node`, `Edge`, `Layer`
and `Passage`, which are the basic items to work with
1. `evaluation` -- provides methods for comparing passages and inspecting the differences
1. `layer0` -- provides the text layer (layer 0) objects: `Layer0` and `Terminal`
1. `layer1` -- provides the foundational layer objects: `Layer1`, `FoundationalNode`,
`PunctNode` and `Linkage`
1. `textutil` -- provides text processing utilities

In addition, a `tests` package is present, enabling unit-testing.

Authors
------
* Amit Beka: amit.beka@gmail.com
* Daniel Hershcovich: danielh@cs.huji.ac.il