import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 256
train_data = datasets.FashionMNIST(root='../data', train=True, download=False, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root='../data', train=False, download=False, transform=transforms.ToTensor())
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

input_size = 28*28
num_classes = 10
dim_hiddens = 256

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, dim_hiddens),
    nn.ReLU(),
    nn.Linear(dim_hiddens, num_classes)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
lr = 0.1
num_epochs = 10

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print(f'epoch {epoch + 1}, loss {l:f}')