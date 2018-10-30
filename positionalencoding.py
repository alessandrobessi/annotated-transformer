import math

import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # noinspection PyUnresolvedReferences
        pe = torch.zeros(max_len, d_model)
        # noinspection PyUnresolvedReferences
        position = torch.arange(0, max_len).unsqueeze(1)
        # noinspection PyUnresolvedReferences
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # noinspection PyUnresolvedReferences
        pe[:, 0::2] = torch.sin(position * div_term)
        # noinspection PyUnresolvedReferences
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
