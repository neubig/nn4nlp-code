This folder contains a simple neural NER tagger based on bidirectional Long-Short Term Memory (LSTM) networks. Its purpose is to serve as an introductionary tutorial for CMU 11-747 to  demonstrate the usage of Pytorch on common NLP tasks.

## Data

We will use the CoNLL 2003 Named Entity Recognition dataset. The dataset is included in `/data/ner/conll2003`.

## Conda Environment
The code is written in Python 3.6 and Pytorch 1.3 with some supporting third-party libraries. We provided a conda environment to install Python 3.6 with required libraries. Simply run

```
conda env create -f env.yml
```

A conda environment named pytorch1.3 will be created. 

## Usage

To start training, simply run

```bash
python ner.py --cuda \ 
    train \
    --batch-size 32 \
    --embedding-size 256 \
    --hidden-size 256 \
    --train-set ../data/ner/conll/train.txt \
    --dev-set ../data/ner/conll/valid.txt \
    --model-save-path model.bin
```

To evaluate a trained model (e.g., `model.bin`) on the testing test, use the following command
```bash
python ner.py --cuda \
    test \
    --batch-size 32 \
    --model-path model.bin \
    --test-set ../data/ner/conll/test.txt \
    --output test.predict.txt
```
The prediction results will be saved in `test.predict.txt`. It will call the official evaluation script `conlleval` to compute evaluation metrics.

## License
This work is licensed under a Creative Commons Attribution 4.0 International License.
