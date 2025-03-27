import torch
from torch import nn
import math
class Positional_Encoding(nn.Module):
    """
    位置编码，构造一个PE矩阵，包含位置信息
    input：每一个词有多少个特征维度表示 d_model, 输入语句的长度 seq_len
    output：size = seq_len * d_model
    """
    def __init__(self, d_model, seq_len):
        super(positional_encoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len



        # self.pe = self.get_pe1()
        self.pe = self.get_pe2()
        # print(self.pe == self.get_pe1())
        # print(torch.allclose(self.pe, self.get_pe1(), rtol=1e-4, atol=1e-6))
        # 为什么这个地方显示 false
    def forward(self):
        return self.pe

    def get_pe1(self):
        """这是一种写法，比较基础"""
        pe = torch.zeros(self.seq_len, self.d_model, dtype=torch.float)
        pos = torch.arange(self.seq_len, dtype=torch.float).unsqueeze(dim=1)

        div_part = torch.arange(0, self.d_model, step=2, dtype=torch.float)
        pe[:, 1::2] = torch.cos(pos / 10000 ** (div_part / self.d_model))
        pe[:, 0::2] = torch.sin(pos / 10000 ** (div_part / self.d_model))
        pe = pe.unsqueeze(dim=0)
        return pe

    def get_pe2(self):
        """这是官方写法，对于数值部分进行了等效转换"""
        pe = torch.zeros(self.seq_len, self.d_model, dtype=torch.float)
        position = torch.arange(0, self.seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) *
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        return pe

## 此部分代码其实不需要用到类，只需要定义一个函数即可

def positional_encoding(d_model, seq_len):
    """
    位置编码，构造一个PE矩阵，包含位置信息
    input：每一个词有多少个特征维度表示 d_model, 输入语句的长度 seq_len
    output：size = seq_len * d_model
    """
    pe = torch.zeros(seq_len, d_model, dtype=torch.float)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(dim=1)
    div_part = torch.exp(torch.arange(0, d_model, step=2, dtype=torch.float) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(pos * div_part)
    pe[:, 1::2] = torch.cos(pos * div_part)

    return pe


if __name__ == '__main__':
    # example_sentence = torch.arange(200, dtype=torch.float)
    pe = positional_encoding(d_model=100, seq_len=200)
    # print(pe.forward())
