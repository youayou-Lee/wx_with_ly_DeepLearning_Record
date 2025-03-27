import torch
from torch import nn
from src.transformer.EncoderLayer_hand import EncoderLayer
from src.transformer.DecoderLayer_hand import DecoderLayer
from src.transformer.PositionalEncoding_hand import positional_encoding
class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, N):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, N, vocab_size):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, n_heads) for _ in range(N)])
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = positional_encoding(d_model)

class Decoder(nn.Module):
    pass
