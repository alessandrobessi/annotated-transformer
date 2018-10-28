from torch import nn as nn


class EncoderDecoder(nn.Module):

    def __init__(self,
                 encoder,
                 decoder,
                 source_embed,
                 target_embed,
                 generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator

    def forward(self,
                source,
                target,
                source_mask,
                target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.source_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)
