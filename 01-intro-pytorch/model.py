import torch
from torch import nn
from torch.autograd import Variable

class BoW(torch.nn.Module):
    def __init__(self, nwords, ntags):
        super(BoW, self).__init__()

        """ variables """
        type = torch.FloatTensor
        use_cuda = torch.cuda.is_available()

        if use_cuda:
            type = torch.cuda.FloatTensor

        self.bias = Variable(torch.zeros(ntags),
                             requires_grad=True).type(type)

        """ layers """
        self.embedding = nn.Embedding(nwords, ntags)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        nn.init.xavier_uniform_(self.embedding.weight)


    def forward(self, words):
        emb = self.embedding(words)
        out = torch.sum(emb, dim=0) + self.bias # size(out) = N
        out = out.view(1, -1) # size(out) = 1 x N
        return out


class CBoW(torch.nn.Module):
    def __init__(self, nwords, ntags, emb_size):
        super(CBoW, self).__init__()

        """ layers """
        self.embedding = nn.Embedding(nwords, emb_size)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        nn.init.xavier_uniform_(self.embedding.weight)
        self.linear = nn.Linear(emb_size, ntags) # bias is True by default
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, words):
        emb = self.embedding(words)
        emb_sum = torch.sum(emb, dim=0) # size(emb_sum) = emb_size
        emb_sum = emb_sum.view(1, -1) # size(emb_sum) = 1 x emb_size
        out = self.linear(emb_sum) # size(out) = 1 x ntags 
        return out


class DeepCBoW(torch.nn.Module):
    def __init__(self, nwords, ntags, nlayers, emb_size, hid_size):
        super(DeepCBoW, self).__init__()

        """ variables """
        self.nlayers = nlayers

        """ layers """
        self.embedding = nn.Embedding(nwords, emb_size)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        nn.init.xavier_uniform_(self.embedding.weight)

        # add nlayers number of layers
        self.linears = nn.ModuleList([
                nn.Linear(emb_size if i == 0 else hid_size, hid_size) \
                for i in range(nlayers)])
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        for i in range(nlayers):
            nn.init.xavier_uniform_(self.linears[i].weight)

        self.output_layer = nn.Linear(hid_size, ntags)
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, words):
        emb = self.embedding(words)
        emb_sum = torch.sum(emb, dim=0) # size(emb_sum) = emb_size
        h = emb_sum.view(1, -1) # size(h) = 1 x emb_size
        for i in range(self.nlayers):
            h = torch.tanh(self.linears[i](h))
        out = self.output_layer(h)
        return out
