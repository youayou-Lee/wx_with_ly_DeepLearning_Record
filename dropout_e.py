from torch import nn
import torch
from d2l import torch as d2l

dropout1, dropout2 = 0.2, 0.5
num_epochs = 10
lr = 0.5
batch_size = 256


def Net():
    net = nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Dropout(dropout1),nn.Linear(256,256),nn.ReLU(),nn.Dropout(dropout2),nn.Linear(256,10))
    return net
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)   #使用正态分布初始化全连接层权重，std:标准差0.01

net = Net()
net.apply(init_weights)

if __name__ == '__main__':
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer =torch.optim.SGD(net.parameters(),lr = lr)
    loss = nn.CrossEntropyLoss()
    d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)

