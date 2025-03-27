# Attention is all you need

文章中提到的几个点：

1. 起初用于机器翻译等任务
2. 训练速度相较于 RNN 快很多

与卷积相比，假设现在有一个很大的图片，两个距离较远的像素点，需要通过好几层卷积，才能将二者融合到一个值里，但是对于Attention，可以一次就看到所有点之间的关系

但是卷积有一个好处是可以做多输出通道映射，所以Transformer设计了多头自注意力机制（MultiHead self-Attention）

因为残差连接的原因，所以每一个子层的输出维度 $d_{model}=512$,又因为output是V的加权和，所以$d_{output} = d_{v} = 512$

在Encoder中，一次可以看到所有输入的词，但是在Decoder中，output是一个一个的生成，然后将t-1时刻的词作为输入放到input中（auto-regresive 自回归）

一般情况下，每一个词用512个维度的向量表示，即$d_k=512$

对于一般的点积注意力机制，$Q = [n * d_k],K = [m * d_k], V = [m * d_v]$
但是**在self-attention中，Q，K，V size 一致，都来自于输出做投影**

## Batch normalization 对三维数据切片做归一化时，为什么是对每个特征对应的所有batch 和 词 进行归一化，而不是 对每一个 词对应的所有batch 和 特征 进行归一化

## 为什么在Attention计算中需要除$\sqrt{d_k}$?

当$d_k过大时，QK^T的值会较大，导致整个向量（经过softmax之后的size是n*d_k）$也就是说会让d_k个数值组成的区域变大，而softmax会导致output更靠近两端，容易出现梯度消失

## 为什么在embedding中需要乘上$\sqrt{d_k}$?

## tokenizer 和 embedding有什么区别？

区别：

1. tokenizer 是将原始文本分割成更小的单元，并将其映射为模型可以理解的离散符号（通常是整数索引）
2. Embedding 将离散的符号转换为连续的向量表示，这些向量能够捕获词汇之间的语义信息



