import torch
from torch import nn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        """针对每一个encoder块，大概分为四个部分
        1. 多头自注意力部分
        2. 残差连接
        3. Layer Normalization
        4. 全连接
        """
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PoswiseFeedForwardNet(d_model, d_ff)
        self.LayerNorm1 = nn.LayerNorm(d_model)
        self.LayerNorm2 = nn.LayerNorm(d_model)

    def forward(self, X):
        # encoder的 MultiHeadAttention 和 decoder的 不一样，前者QKV一致，后者KV来自Encoder，Q来自Decoder的input，所以此处传入三个一样的参数
        output1 = self.self_attn(X, X, X)
        output2 = self.LayerNorm1(output1 + X)
        output3 = self.feed_forward(output2)
        return self.LayerNorm2(output3 + output2)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # 先将QKV做投影，即线性层
        # 对于每一个Linear应该大小事 d_model x d_model // n_heads ,但是假设有8个头，定义 3 * 8个Linear有点过于麻烦了
        # 所以此处 选择 先定义三个Linear， 然后得到不同的 output再分割 ，然后传入 ScaledDotProductAttention
        # self.W_Q = nn.Linear(d_model, d_model // n_heads * n_heads)
        # self.W_K = nn.Linear(d_model, d_model // n_heads * n_heads)
        # self.W_V = nn.Linear(d_model, d_model // n_heads * n_heads)

        self.W_Q = [nn.Linear(d_model, d_model // n_heads)] * n_heads
        self.W_K = [nn.Linear(d_model, d_model // n_heads)] * n_heads
        self.W_V = [nn.Linear(d_model, d_model // n_heads)] * n_heads

        # 做完注意力后会对所有结果拼接，然后投影回原始维度
        self.W_O = nn.Linear(d_model // n_heads * n_heads, d_model)
    def forward(self, Q, K, V, mask=None):
        q = torch.concat([w(Q) for w in self.W_Q])
        k = torch.concat([w(K) for w in self.W_K])
        v = torch.concat([w(V) for w in self.W_V])

        output = torch.concat([ScaledDotProductAttention(q[i], k[i], v[i]) for i in range(self.n_heads)], dim=1)
        return self.W_O(output)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)
        self.bias1 = nn.Parameter(torch.zeros(d_ff))
        self.bias2 = nn.Parameter(torch.zeros(d_model))
        self.relu = nn.ReLU()

    def forward(self, X):
        return self.W_2(self.relu(self.W_1(X) + self.bias1)) + self.bias2

def ScaledDotProductAttention(q, k, v, mask=None):
    scores = q @ k.T / k.shape[1] ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    return torch.softmax(scores, dim=-1) @ v



if __name__ == '__main__':

    net = EncoderLayer(d_model=512, d_ff=2048, n_heads=8)
    X = torch.randn(10, 10, 512, dtype=torch.float)
    print(net(X).shape)