import math
import torch
import numpy as np
from torch import nn
from d2l import torch as d2l

max_degree = 20
n_train,n_test = 100,100
true_w = np.zeros(max_degree) #真实的权重向量，初始化为零向量，并在特定位置设置非零值。
true_w[0:4] = np.array([5,1.2,-3.4,5.6])


features = np.random.normal(size = (n_train+n_test,1))
np.random.shuffle(features)
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1)#(i + 1) 的阶乘
labels = np.dot(poly_features,true_w)
labels += np.random.normal(scale=0.1,size=labels.shape)
true_w,features,poly_features,labels = [torch.tensor(x,dtype=torch.float32)for x in [true_w,features,poly_features,labels]]
def evaluate_loss(net,data_iter,loss):
    metric = d2l.Accumulator(2)
    for x,y in data_iter:     #X 是模型输入的特征数据。它是从数据迭代器 data_iter 中提取的一个批次（batch）的数据
        out = net(x)
        y = y.reshape(out.shape)
        l = loss(out,y)
        metric.add(l.sum(),l.numel())
    return metric[0]/metric[1]
def train(train_features,test_features,train_labels,test_labels,num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape,1,bias = False))
    batch_size = min(10,train_labels.shape[0])
    train_iter = d2l.load_array((train_features,train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features,test_labels.reshape(-1,1)),batch_size ,is_train=False)
    trainer = torch.optim.SGD(net.parameters(),lr = 0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    all_loss = 0
    train_ls=[]
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
            all_loss += l.sum().item()
        train_ls.append(all_loss / batch_size)
        all_loss = 0
        if epoch == 0 or (epoch+1)%20 ==0:
            animator.add(epoch+1,(evaluate_loss(net,train_iter,loss),evaluate_loss(net,test_iter,loss)))
        print('weight:',net[0].weight.data.numpy())
        print('epoch %d, train loss %f' % (epoch + 1, train_ls[epoch]))
train(poly_features[:n_train,:4],poly_features[n_train:,:4],labels[:n_train],labels[n_train:])




