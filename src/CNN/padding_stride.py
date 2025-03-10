import torch
from torch import nn

def comp_conv2d(conv2d, X):
    """这个函数用来告诉我们输入X 经过卷积层之后的size变化"""
    X = X.reshape((1,1)+X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])
# 定义一个kernel为3*3，padding为1 的 卷积层
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
# 输出size = [X.shape[0] - kernel_size[0] + 2*padding[0] + 1, X.shape[1] - kernel_size[1] + 2*padding[1] + 1]
X = torch.rand(size=(8,8))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,3), padding=(2,1))
print(comp_conv2d(conv2d, X).shape)

# 带stride的卷积
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

conf2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5,3), padding=(2,3), stride=(1,2))
print(comp_conv2d(conv2d, X).shape)