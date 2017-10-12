#!/usr/bin/env bash

case "$TEST_SUITE" in
unit)
    # unit tests
    python -m unittest discover -v || exit 1
    PASSAGES=../doc/toy.xml
    ;;
convert)
    mkdir pickle
    curl -L http://www.cs.huji.ac.il/~danielh/ucca/ucca_corpus_pickle.tgz | tar xz -C pickle || curl -L https://www.dropbox.com/s/q4ycn45zlmhuf9k/ucca_corpus_pickle.tgz | tar xz -C pickle
    PASSAGES=../pickle/*.pickle
    ;;
esac
cd $(dirname $0)
mkdir -p converted
for FORMAT in conll sdp export "export --tree"; do
    echo === Evaluating $FORMAT ===
    if [ $# -lt 1 -o "$FORMAT" = "$1" ]; then
        python ../scripts/convert_and_evaluate.py "$PASSAGES" -f $FORMAT | tee "$FORMAT.log"
    fi
done