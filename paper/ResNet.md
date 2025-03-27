# Deep Residual Learning for Image Recognition
## Introduction

提出问题，随着网络越来越深，梯度会出现爆炸或消失   

**解决办法**：   
1. 在初始化的时候做好一点，采用Xavier初始化。
2. 在每一层之间加一些normalization, 比如batch normalization,layer normalization. 可以使得每个层之间输出和梯度在比较深的网络中仍然能不出现梯度消失和爆炸

文章提出 对于更深的神经网络，精度变低了，不是因为模型更复杂导致的过拟合，而是因为本身过于复杂，导致模型底层参数train不动

过拟合一般指 训练误差低但验证误差高

## 提出的疑问 ： 学习更优的网络是否只是简单地堆叠更多的层？

阻碍1：过多的层数会引起梯度消失和梯度爆炸 (vanishing / exploding gradient)

解决方法：
1. 初始化权重参数的时候进行归一化  
2. 对中间的一些层进行归一化


