"""A bi-lstm neural NER tagger"""

import itertools
import os
import re
from argparse import ArgumentParser
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
from typing import Any, Union, Dict, List, Tuple, Optional, Iterator

Vocab = namedtuple('Vocab', ['word_to_id', 'ner_tag_to_id', 'id_to_ner_tag'])
Example = namedtuple('Example', ['sentence', 'pos_tags', 'syn_tags', 'ner_tags'])


def load_data(file_path: Union[Path, str]) -> List[Example]:
    """load the CONLL2003 NER dataset"""
    if isinstance(file_path, str):
        file_path = Path(file_path)

    assert file_path.exists()
    examples = []
    sentence, pos_tags, syn_tags, ner_tags = [], [], [], []
    for line in file_path.open():
        line = line.strip()
        if line:
            token, pos_tag, syn_tag, ner_label = line.split(' ')

            sentence.append(token)
            pos_tags.append(pos_tag)
            syn_tags.append(syn_tag)
            ner_tags.append(ner_label)
        else:
            if sentence:
                examples.append(
                    Example(sentence=sentence, pos_tags=pos_tags, syn_tags=syn_tags, ner_tags=ner_tags)
                )

                sentence, pos_tags, syn_tags, ner_tags = [], [], [], []

    return examples


def build_vocab(train_set: List[Example]) -> Vocab:
    """
    Create index (dictionary) for word types and ner tags in the training set.
    Singletons (words with that only appears once) will be ignored
    """

    all_words = itertools.chain.from_iterable(
        e.sentence
        for e
        in train_set
    )

    word_freq = Counter(all_words)
    valid_words = sorted(
        filter(lambda x: word_freq[x] > 1, word_freq),
        key=lambda x: word_freq[x],
        reverse=True
    )

    word_to_id = {
        '<pad>': 0,
        '<unk>': 1,
    }

    word_to_id.update({
        word: idx
        for idx, word
        in enumerate(valid_words, start=2)
    })

    ner_tag_to_id = dict()
    ner_tag_to_id.update({
        tag: idx
        for idx, tag
        in enumerate(
            sorted(
                set(
                    itertools.chain.from_iterable(
                        e.ner_tags for e in train_set
                    )
                )
            ),
            start=0
        )
    })

    id_to_ner_tag = {
        idx: tag
        for tag, idx
        in ner_tag_to_id.items()
    }

    vocab = Vocab(
        word_to_id=word_to_id,
        ner_tag_to_id=ner_tag_to_id,
        id_to_ner_tag=id_to_ner_tag
    )

    return vocab


def to_input_tensor(
    token_sequence: List[List[Any]],
    token_to_id_map: Dict,
    pad_index: int = 0,
    unk_index: int = 1
) -> torch.Tensor:
    """
    Given a batched list of sequences, convert them to a pytorch tensor

    Args:
        token_sequence: a batched list of sequences, each sequence is a list of tokens
        token_to_id_map: a dictionary that maps tokens into indices
        pad_index: padding index
        unk_index: index for unknown tokens not included in the `token_to_id_map`

    Output:
        sequence_array: a Pytorch tensor of size (batch_size, max_sequence_len),
            representing input to the neural network
    """

    max_sequence_len = max(len(seq) for seq in token_sequence)
    batch_size = len(token_sequence)
    sequence_array = np.zeros((batch_size, max_sequence_len), dtype=np.int64)
    sequence_array.fill(pad_index)
    for e_id in range(batch_size):
        sequence_i = token_sequence[e_id]

        id_sequence = [
            token_to_id_map.get(token, unk_index)
            for token
            in sequence_i
        ]

        sequence_array[e_id, :len(id_sequence)] = id_sequence

    sequence_array = torch.from_numpy(sequence_array)

    return sequence_array


class NerModel(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, vocab: Vocab):
        super(NerModel, self).__init__()

        self.embedding = nn.Embedding(len(vocab.word_to_id), embedding_size)
        self.bi_lstm = nn.LSTM(
            input_size=embedding_size, hidden_size=hidden_size,
            batch_first=True, bidirectional=True
        )
        self.predictor = nn.Linear(hidden_size * 2, len(vocab.ner_tag_to_id))

        self.vocab = vocab
        self.config = {
            'embedding_size': embedding_size,
            'hidden_size': hidden_size,
            'vocab': vocab
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, sentences: List[List[str]], targets: List[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform the forward pass, given a batched list of sentences, compute the neural network output,
        which is the distribution of NER tags for each word in input sentences.

        If the gold-standard NER labels (`targets`) is given, the forward function will also return the
        loss value, which is the negative log-likelihood of predicting the gold NER labels.

        All outputs are packaged in a Python dictionary.

        Args:
            sentences: a mini-batch, consisting of a list of tokenized sentences
            targets: Optional, the list of gold NER tags for each input sentence in the mini-batch

        Outputs:
            output_dict: a python dictionary of keyed tensors, which are outputs of the neural network.
                If `targets` is provided, the dictionary will also contain the loss variable for back propagation.

                By default, the dictionary includes a tensor `tag_logits`, which is the unnormalized output from the
                final prediction layer with shape (batch_size, max_sequence_len, tag_num).
        """

        # First, we convert the input batch of tokenized sentences to a matrix (tensor) of word indices.
        # The shape and data type of the matrix is:
        # torch.LongTensor: (batch_size, max_sequence_len),
        # where `max_sequence_len` denotes the length of the longest sentence in the batch.
        # For sentences shorter than `max_sequence_len`, we right-pad them using the `pad_index` (e.g., zero)
        # We also replace singleton word types with a special `<unk>` token to prevent over fitting.
        # Finally, we move the newly created tensor to the device the model resides on (CPU or indexed GPU card)

        # Side Note:
        #   (1) You may also represent the tensor by swapping the dimensions: (max_sequence_len, batch_size),
        #   making `batch_size` the second dimension. However, it is commonly the case to use the first dimension
        #   to represent batch size.
        #   (2) Somehow, in PyTorch, many functions use the second dimension to represent batch size by default
        #   (e.g., the built-in LSTM). This is due to certain technical reasons (some computations might be faster if
        #   the first dimension is not batch size). However, for clarity reasons I would not recommend this.
        word_ids = to_input_tensor(
            sentences, self.vocab.word_to_id,
            pad_index=self.vocab.word_to_id['<pad>'], unk_index=self.vocab.word_to_id['<unk>']
        ).to(self.device)

        # Next, use the tensor of word indices to query the word embedding layer, and
        # get the resulting word embeddings for each token in `word_ids`.
        # torch.FloatTensor: (batch_size, max_sequence_len, embedding_size)
        word_embeddings = self.embedding(word_ids)

        # When running LSTMs over batched inputs, we need to take care of the padding entries,
        # as they are not supposed to be involved in the computation. Fortunately, Pytorch provides
        # a special function to mark the proper length for each sequence (i.e., the sequence of embeddings
        # of a sentence) in the batch. The BiLSTM module will then ignore the padded indices using the
        # length information.
        # type: PackedSequence
        packed_word_embeddings = pack_padded_sequence(
            word_embeddings,
            lengths=torch.tensor([len(seq) for seq in sentences], device=self.device),
            batch_first=True, enforce_sorted=False
        )

        # source_encodings: (batch_size, max_sequence_len, hidden_size * 2)
        # last_state, last_cell: List[(batch_size, hidden_size * 2)]
        source_encodings, (last_state, last_cell) = self.bi_lstm(packed_word_embeddings)
        source_encodings, _ = pad_packed_sequence(source_encodings, batch_first=True)

        # We use a simple linear layer to transform the LSTM hidden states to a categorical distribution
        # over target NER labels. Note that the following quantity is the ``logits'' before the Softmax
        # layer. Softmax will be implicitly applied by `nn.CrossEntropyLoss` when computing the loss value
        # in below
        # (batch_size, max_sequence_len, tag_num)
        tag_logits = self.predictor(source_encodings)

        output_dict = {
            'tag_logits': tag_logits
        }

        if targets is not None:
            # We convert the target tag sequences in the batch to a Pytorch tensor of their indices
            # Note that we use `-1` to as the padding index
            # torch.LongTensor: (batch_size, max_sequence_len)
            target_tag_ids = to_input_tensor(targets, self.vocab.ner_tag_to_id, pad_index=-1).to(self.device)

            # the number of all possible tags
            tag_num = len(self.vocab.ner_tag_to_id)

            # scalar of the average of the log-likelihood of predicting the gold-standard tag at each position,
            # for each sentence in the batch.
            # This corresponds to:
            #   (1) apply Softmax to the `tag_logits`, generating the categorical distribution
            # over NER labels for each word;
            #   (2) Grab the probabilities of the gold-standard NER label for each word, and compute their average
            #       as the loss value.
            # Note: `nn.CrossEntropyLoss` will ignore the padded indices specified by `ignore_index`
            loss = nn.CrossEntropyLoss(ignore_index=-1)(
                tag_logits.view(-1, tag_num),  # (batch_size * max_sequence_len, tag_num)
                target_tag_ids.view(-1)  # (batch_size * max_sequence_len)
            )

            output_dict['loss'] = loss

        return output_dict

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Performs inference, predicting the most likely tag for each input token

        Args:
            sentences: a batched list of sentences

        Output:
            predicted_tags: a list of predicted tag sequences, one for each input sentence in the batch
        """

        # We do not need to perform back-propagation in inference, so we use the context manager `torch.no_grad()`.
        # This will avoid book-keeping all necessary information required by back-propagation during forward
        # computation, and saves memory and significantly improves speed.
        with torch.no_grad():
            # The best practice to write a `predict()` function for testing is to reuse
            # as much code for training as possible. The `forward()` function should compute
            # the network's outputs used in both training and testing. In our case, it is the
            # ``logits'' for computing the distribution of NER labels for each word.
            encoding_dict = self.forward(sentences)

            # (batch_size, max_sequence_len, tag_num)
            tag_logits = encoding_dict['tag_logits']

            # Perform greedy decoding to compute the most likely NER label for each word.
            # The code snnippt in the following for loop is cpu intensive and uses lots of
            # random tensors indexing. It is usually a good idea of move the tensor to CPU.
            # (and convert the tensor to a `numpy.ndarray` if you like)
            # (batch_size, max_sequence_len)
            predicted_tags_ids = torch.argmax(tag_logits, dim=-1).cpu().numpy()

            predicted_tags = []
            for e_id, sentence in enumerate(sentences):
                tag_sequence = []
                for token_pos, token in enumerate(sentence):
                    # Grab the index of the NER label with the highest probability
                    pred_tag_id = predicted_tags_ids[e_id, token_pos]
                    # Get the original NER tag
                    pred_tag = self.vocab.id_to_ner_tag[pred_tag_id]

                    tag_sequence.append(pred_tag)

                predicted_tags.append(tag_sequence)

            return predicted_tags

    def save(self, model_path: Union[Path, str]) -> None:
        """save the model to `model_path`"""
        if isinstance(model_path, Path):
            model_path = str(model_path)

        # `self.state_dict()` returns a dictionary consisting of all model parameters
        # `config` records the hyper-parameters (embedding_size, hidden_size, etc.)
        model_state = {
            'state_dict': self.state_dict(),
            'config':  self.config
        }

        torch.save(model_state, model_path)

    @classmethod
    def load(cls, model_path: Union[Path, str]) -> 'NerModel':
        """load a trained model from `model_path`"""

        model_state = torch.load(str(model_path), map_location=lambda storage, loc: storage)
        args = model_state['config']

        model = cls(**args)
        model.load_state_dict(model_state['state_dict'])

        return model


def batch_iter(data: List[Any], batch_size: int = 32, shuffle: bool = False) -> Iterator[Any]:
    """
    Return an iterator of batches for a dataset

    Args:
        data: a dataset represented by a list of examples
        batch_size: batch size
        shuffle: whether to randomly shuffle the examples in the data set

    Output:
        a generator of batched examples.
    """
    batch_num = int(np.ceil(len(data) / batch_size))
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        batch_examples = [data[idx] for idx in indices]

        yield batch_examples


def evaluate(
    model: NerModel,
    data_set: List[Example],
    batch_size: int = 32,
    output_file: Optional[Path] = None
) -> Dict:
    """Evaluate a model over a dataset"""

    was_training = model.training
    # Set the model to evaluation mode, this will impact behaviour of
    # some stochastic operations like `Dropout`
    model = model.eval()

    predict_results = []
    reference = []

    for batch in batch_iter(data_set, batch_size=batch_size):
        sentences = [e.sentence for e in batch]
        ref_tag_sequences = [e.ner_tags for e in batch]

        batch_pred_result = model.predict(sentences)

        predict_results.extend(batch_pred_result)
        reference.extend(ref_tag_sequences)

    if was_training:
        model = model.train()

    # The following code snippet dumps the prediction results to CoNLL evaluation
    # format and call the official evaluation script for evaluation.
    output_file = output_file or Path('_tmp_prediction.txt')

    with output_file.open('w') as f:
        for example, hyp in zip(data_set, predict_results):
            sent_len = len(example.sentence)
            for idx in range(sent_len):
                f.write(' '.join([
                    example.sentence[idx],
                    example.pos_tags[idx],
                    example.syn_tags[idx],
                    example.ner_tags[idx],
                    hyp[idx]]
                ) + '\n')

            f.write('\n')

    eval_output = os.popen(f'perl conlleval < {output_file}').read()
    print(eval_output)
    accuracy = re.search(r'accuracy: +(\d+\.\d+)\%', eval_output).group(1)
    accuracy = float(accuracy)

    eval_result = {
        'accuracy': accuracy
    }

    return eval_result


def train(args):
    """training procedure"""

    # load the training and development set
    train_set = load_data(args.train_set)
    dev_set = load_data(args.dev_set)
    # build the vocabulary, create dictionaries that map words and NER tags to indices (and vice versa)
    vocab = build_vocab(train_set)
    # Build the model
    model = NerModel(embedding_size=args.embedding_size, hidden_size=args.hidden_size, vocab=vocab)

    if args.cuda:
        # Move model to GPU if `--cuda` flag is specified
        model = model.cuda()

    # Set the model to train mode, this is important since some modules behave differently
    # in training and testing (e.g., Dropout)
    model = model.train()

    # Create an Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    running_dev_accuracy = 0.

    for epoch_id in range(args.max_epoch):
        train_batch_iter = batch_iter(train_set, batch_size=args.batch_size, shuffle=True)
        # Iterate over mini-batches for the current epoch
        for batch_id, batch in enumerate(train_batch_iter):
            # Clear the gradients of parameters
            optimizer.zero_grad()

            sentences = [e.sentence for e in batch]
            tag_sequences = [e.ner_tags for e in batch]

            # Perform forward pass to get neural network outputs
            return_dict = model(sentences, tag_sequences)
            # Grab the loss tensor
            loss = return_dict['loss']
            # Call `backward()` on `loss` for back-propagation to compute
            # gradients w.r.t. model parameters
            loss.backward()
            # Perform one step of parameter update using the newly-computed gradients
            optimizer.step()

            print(f'Epoch {epoch_id}, batch {batch_id}, loss={loss.item()}')

        eval_result = evaluate(model, dev_set, args.batch_size)
        dev_accuracy = eval_result["accuracy"]

        print(f'Epoch {epoch_id} dev. accuracy={dev_accuracy}')
        # We save the model if the dev. accuracy is better than previous epochs
        if dev_accuracy > running_dev_accuracy:
            model.save(args.model_save_path)


def test(args):
    """testing procedure"""

    test_set = load_data(args.test_set)

    model = NerModel.load(args.model_path)
    if args.cuda:
        model = model.cuda()

    evaluate(model, test_set, batch_size=args.batch_size, output_file=args.output)


def main():
    arg_parser = ArgumentParser('A bi-lstm neural NER tagger')
    arg_parser.add_argument('--cuda', action='store_true', help='Whether to use GPU')
    arg_parser.set_defaults(cuda=False)
    subparsers = arg_parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help='Training procedure')
    train_parser.set_defaults(action='train')
    train_parser.add_argument('--train-set', type=Path, required=True, help='Path to the training set')
    train_parser.add_argument('--dev-set', type=Path, required=True, help='Path to the development set for validation')
    train_parser.add_argument('--embedding-size', type=int, default=256, help='Size of the embedding vectors.')
    train_parser.add_argument('--hidden-size', type=int, default=256, help='Size of the LSTM hidden layer.')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    train_parser.add_argument('--max-epoch', type=int, default=32, help='Maximum number of training epoches')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam')
    train_parser.add_argument('--model-save-path', type=Path, default='model.bin', help='Model save path')

    test_parser = subparsers.add_parser('test', help='Testing procedure')
    test_parser.set_defaults(action='test')
    test_parser.add_argument('--model-path', type=Path, required=True, help='Path to the model to evaluate')
    test_parser.add_argument('--test-set', type=Path, required=True, help='Path to the testing set')
    test_parser.add_argument('--output', type=Path, required=True, help='Path to the prediction output')
    test_parser.add_argument('--batch-size', type=int, default=32, help='Testing batch size')

    args = arg_parser.parse_args()
    if args.action == 'train':
        train(args)
    elif args.action == 'test':
        test(args)


if __name__ == '__main__':
    main()
