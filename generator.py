from torch import nn as nn
from torch import functional as F


class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.projection = nn.Linear(d_model, vocab)

    def forward(self, x):
        # noinspection PyUnresolvedReferences
        return F.log_softmax(self.projection(x), dim=-1)
