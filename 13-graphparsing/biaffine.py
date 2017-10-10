import copy
import dynet as dy
import numpy as np
from mst import mst


class DeepBiaffineAttentionDecoder(object):
    """ For MST dependency parsing (ICLR 2017)"""

    def __init__(self,
                 model,
                 n_labels,
                 src_ctx_dim=400,
                 n_arc_mlp_units=500,
                 n_label_mlp_units=100,
                 arc_mlp_dropout=0.33,
                 label_mlp_dropout=0.33):
        """
        To reproduce the results of the original paper,
        requires (1) the encoder to be a 3-layer bilstm;
        (2) dropout rate of bilstm to be 0.33
        (3) pretrained embeddings and embedding dropout rate to be 0.33
        """

        self.src_ctx_dim = src_ctx_dim
        self.label_mlp_dropout = label_mlp_dropout
        self.arc_mlp_dropout = arc_mlp_dropout
        self.n_labels = n_labels
        self.W_arc_hidden_to_head = model.add_parameters((n_arc_mlp_units, src_ctx_dim))
        self.b_arc_hidden_to_head = model.add_parameters((n_arc_mlp_units,))
        self.W_arc_hidden_to_dep = model.add_parameters((n_arc_mlp_units, src_ctx_dim))
        self.b_arc_hidden_to_dep = model.add_parameters((n_arc_mlp_units,))

        self.W_label_hidden_to_head = model.add_parameters((n_label_mlp_units, src_ctx_dim))
        self.b_label_hidden_to_head = model.add_parameters((n_label_mlp_units,))
        self.W_label_hidden_to_dep = model.add_parameters((n_label_mlp_units, src_ctx_dim))
        self.b_label_hidden_to_dep = model.add_parameters((n_label_mlp_units,))

        self.U_arc_1 = model.add_parameters((n_arc_mlp_units, n_arc_mlp_units))
        self.u_arc_2 = model.add_parameters((n_arc_mlp_units))

        self.U_label_1 = [model.add_parameters((n_label_mlp_units, n_label_mlp_units)) for _ in range(n_labels)]
        self.u_label_2_2 = [model.add_parameters((1, n_label_mlp_units)) for _ in range(n_labels)]
        self.u_label_2_1 = [model.add_parameters((n_label_mlp_units, 1)) for _ in range(n_labels)]
        self.b_label = [model.add_parameters((1,)) for _ in range(n_labels)]

    def cal_scores(self, src_encodings):
        src_len = len(src_encodings)

        src_encodings = dy.concatenate_cols(src_encodings)  # src_ctx_dim, src_len, batch_size

        W_arc_hidden_to_head = dy.parameter(self.W_arc_hidden_to_head)
        b_arc_hidden_to_head = dy.parameter(self.b_arc_hidden_to_head)
        W_arc_hidden_to_dep = dy.parameter(self.W_arc_hidden_to_dep)
        b_arc_hidden_to_dep = dy.parameter(self.b_arc_hidden_to_dep)

        W_label_hidden_to_head = dy.parameter(self.W_label_hidden_to_head)
        b_label_hidden_to_head = dy.parameter(self.b_label_hidden_to_head)
        W_label_hidden_to_dep = dy.parameter(self.W_label_hidden_to_dep)
        b_label_hidden_to_dep = dy.parameter(self.b_label_hidden_to_dep)

        U_arc_1 = dy.parameter(self.U_arc_1)
        u_arc_2 = dy.parameter(self.u_arc_2)

        U_label_1 = [dy.parameter(x) for x in self.U_label_1]
        u_label_2_1 = [dy.parameter(x) for x in self.u_label_2_1]
        u_label_2_2 = [dy.parameter(x) for x in self.u_label_2_2]
        b_label = [dy.parameter(x) for x in self.b_label]

        h_arc_head = dy.rectify(dy.affine_transform([b_arc_hidden_to_head, W_arc_hidden_to_head, src_encodings]))  # n_arc_ml_units, src_len, bs
        h_arc_dep = dy.rectify(dy.affine_transform([b_arc_hidden_to_dep, W_arc_hidden_to_dep, src_encodings]))
        h_label_head = dy.rectify(dy.affine_transform([b_label_hidden_to_head, W_label_hidden_to_head, src_encodings]))
        h_label_dep = dy.rectify(dy.affine_transform([b_label_hidden_to_dep, W_label_hidden_to_dep, src_encodings]))

        h_arc_head_transpose = dy.transpose(h_arc_head)
        h_label_head_transpose = dy.transpose(h_label_head)

        s_arc = h_arc_head_transpose * dy.colwise_add(U_arc_1 * h_arc_dep, u_arc_2)

        s_label = []
        for U_1, u_2_1, u_2_2, b in zip(U_label_1, u_label_2_1, u_label_2_2, b_label):
            e1 = h_label_head_transpose * U_1 * h_label_dep
            e2 = h_label_head_transpose * u_2_1 * dy.ones((1, src_len))
            e3 = dy.ones((src_len, 1)) * u_2_2 * h_label_dep
            s_label.append(e1 + e2 + e3 + b)
        return s_arc, s_label

    def decode_loss(self, src_encodings, tgt_seqs):
        """
        :param tgt_seqs: (tgt_heads, tgt_labels): list (length=batch_size) of (src_len)
        """

        # todo(NOTE): Sentences should start with empty token (as root of dependency tree)!

        tgt_heads, tgt_labels = tgt_seqs

        src_len = len(tgt_heads[0])
        batch_size = len(tgt_heads)
        np_tgt_heads = np.array(tgt_heads).flatten()  # (src_len * batch_size)
        np_tgt_labels = np.array(tgt_labels).flatten()
        s_arc, s_label = self.cal_scores(src_encodings)  # (src_len, src_len, bs), ([(src_len, src_len, bs)])

        s_arc_value = s_arc.npvalue()
        s_arc_choice = np.argmax(s_arc_value, axis=0).transpose().flatten()  # (src_len * batch_size)

        s_pick_labels = [dy.pick_batch(dy.reshape(score, (src_len,), batch_size=src_len * batch_size), s_arc_choice)
                     for score in s_label]
        s_argmax_labels = dy.concatenate(s_pick_labels, d=0)  # n_labels, src_len * batch_size

        reshape_s_arc = dy.reshape(s_arc, (src_len,), batch_size=src_len * batch_size)
        arc_loss = dy.pickneglogsoftmax_batch(reshape_s_arc, np_tgt_heads)
        label_loss = dy.pickneglogsoftmax_batch(s_argmax_labels, np_tgt_labels)

        loss = dy.sum_batches(arc_loss + label_loss) / batch_size
        return loss

    def decoding(self, src_encodings):
        src_len = len(src_encodings)

        # NOTE: should transpose before calling `mst` method!
        s_arc, s_label = self.cal_scores(src_encodings)
        s_arc_values = s_arc.npvalue().transpose()  # src_len, src_len
        s_label_values = np.asarray([x.npvalue() for x in s_label]).transpose((2, 1, 0))  # src_len, src_len, n_labels

        # weights = np.zeros((src_len + 1, src_len + 1))
        # weights[0, 1:(src_len + 1)] = np.inf
        # weights[1:(src_len + 1), 0] = np.inf
        # weights[1:(src_len + 1), 1:(src_len + 1)] = s_arc_values[batch]
        weights = s_arc_values
        pred_heads = mst(weights)
        pred_labels = [np.argmax(labels[head]) for head, labels in zip(pred_heads, s_label_values)]

        return pred_heads, pred_labels

    def cal_accuracy(self, pred_head, pred_labels, true_head, true_labels):
        head_acc = np.sum(np.equal(pred_head, true_head)).astype(np.float32) / len(pred_labels)

        label_acc = np.sum(np.equal(pred_head, true_head) * np.equal(pred_labels, true_labels)).astype(np.float32) / len(pred_head)

        return head_acc, label_acc
