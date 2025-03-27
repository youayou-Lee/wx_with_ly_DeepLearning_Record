import torch
from torch import nn
from src.transformer import MultiHeadAttention, PoswiseFeedForwardNet

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, mask):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.mask = mask

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.self_attn_mask = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PoswiseFeedForwardNet(d_model, d_ff)
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.LayerNorm2 = nn.LayerNorm(d_model)
        self.LayerNorm3 = nn.LayerNorm(d_model)

    def forward(self, enc_inputs, X):
        output1 = self.self_attn_mask(X, X, X, self.mask)
        output2 = self.LayerNorm1(output1 + X)
        output3 = self.self_attn(output2, enc_inputs, enc_inputs)
        output4 = self.LayerNorm2(output3 + output2)
        output5 = self.feed_forward(output4)
        return self.LayerNorm3(output5 + output4)

if __name__ == '__main__':
    mask = torch.rand(10, 10)
    example = torch.rand(10, 10, 512)
    net = DecoderLayer(d_model=512, d_ff=2048, n_heads=8, mask=mask)
    print(net(example, example).shape)


