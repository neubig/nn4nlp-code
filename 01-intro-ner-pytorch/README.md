This folder contains a simple neural NER tagger based on bidirectional Long-Short Term Memory (LSTM) networks. Its purpose is to serve as an introductionary tutorial for CMU 11-747 to  demonstrate the usage of Pytorch on common NLP tasks.

## The Named Entity Recognition Task

According to [paperswithcode.com](https://paperswithcode.com/task/named-entity-recognition-ner):

> Named entity recognition (NER) is the task of tagging entities in text with their corresponding type. Approaches typically use BIO notation, which differentiates the beginning (B) and the inside (I) of entities. O is used for non-entity tokens.

Here is an example of a sentence and the NER tags of its words:

| Harry | Potter | attended | Hogwarts | in | Scotland | . |
|-------|--------|----------|----------|----|----------|---|
| B-PER | I-PER  | O        | B-ORG    | O  | B-LOC    | . |

Where the prefixes `B-` and `I-` are used to differentiate the beginning (B) and the inside (I) of entities. 
There are four types of NER labels in total: `PER` for person, `ORG` for orgnization, and `LOC` for location.

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

## (My Own) PyTorch Best Practices

* Use an IDE (PyCharm or VS Code) for debugging whenever possible!
    - If you need to use GPUs on a server, try using the remote debugging feature offered by IDEs.
* Log the shapes of PyTorch tensors in comments.
* Allow `targets=None` in `forward(...)` arguments.  

## License
This work is licensed under a Creative Commons Attribution 4.0 International License.
