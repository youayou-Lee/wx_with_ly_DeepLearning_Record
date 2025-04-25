import torch
from torch import nn
from d2l import torch as d2l


def dropout_layer(X,dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) >dropout).float()   #rand生成0到1之间均匀分布的随机数
    return mask*X/(1.0-dropout)
X = torch.arange(16,dtype = torch.float32).reshape((2,8))

#print(X)
   # print(dropout_layer(X,0.))
   # print(dropout_layer(X,0.5))
   # print(dropout_layer(X,1.))

#定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元
num_inputs,num_outputs,num_hiddens1,num_hiddens2= 784,10,256,256
dropout1 ,dropout2 = 0.2,0.5

class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,num_hiddens2,is_training = True):
        super(Net,self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs,num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1,num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2,num_outputs)
        self.relu = nn.ReLU()

    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
        if self.training == True:
            H1 = dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2,dropout2)
        out = self.lin3(H2)
        return out

net = Net(num_inputs,num_outputs, num_hiddens1, num_hiddens2)


#def init_weights(m):
#    if type(m) == nn.Linear or type(m) == nn.Conv2d:
#        nn.init.xavier_uniform_(m.weight)


#net.apply(init_weights)

num_epochs,lr,batch_size = 10, 0.001, 256
loss = nn.CrossEntropyLoss(reduction='none')

def train(net,train_iter,loss,num_epochs,trainer):
    train_ls = []
    all_loss = 0
    for epoch in range(num_epochs):

        for x,y in train_iter:
            trainer.zero_grad()
            l = loss( net(x) , y)
            l.sum().backward()
            trainer.step()
            all_loss += l.sum().item()
        train_ls.append(all_loss / batch_size)
        all_loss = 0
        print('epoch %d, train loss %f' % (epoch + 1, train_ls[epoch]))

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# if __name__ == '__main__':
#     train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
#
#     trainer = torch.optim.SGD(net.parameters(), lr=lr)
#     #train(net,train_iter,loss,num_epochs,trainer)
#     d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)