import torch
import matplotlib as plt

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def generate_data(n_train=50,n_test=100,n_inputs=200,batch_size=5):
    true_w = torch.ones((n_inputs, 1)) * 0.01
    true_b = 0.05
    train_features = torch.normal(0, 1, (n_train, n_inputs))
    train_labels = torch.matmul(train_features, true_w) + true_b
    train_iter = DataLoader(TensorDataset(train_features, train_labels),batch_size, shuffle=True)

    test_features = torch.normal(0, 1, (n_test, n_inputs))
    test_labels = torch.matmul(test_features, true_w) + true_b
    test_iter = DataLoader(TensorDataset(test_features, test_labels),batch_size, shuffle=True)
    return train_iter,test_iter,true_w,true_b

def init_params(n_inputs=200):
    w = torch.normal(0, 1, (n_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w, b = init_params(n_inputs=100)
    net = nn.Sequential(nn.Linear(100, 1))
    loss = nn.MSELoss()
    for param in net.parameters():
        param.data.normal_()
    num_epochs, lr, batch_size = 100, 0.003, 5
    train_iter, test_iter, true_w, true_b = generate_data(n_inputs=100)
    train_loss = []
    all_loss = 0
    optimizer = torch.optim.SGD(
        [
            {
            'params': net[0].weight,
            'weight_decay': lambd
            },
            {"params": net[0].bias}
        ],lr=lr
    )
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            optimizer.step()
            all_loss += l.sum().item()
        train_loss.append(all_loss / len(train_iter))
        all_loss = 0
        print('epoch %d, train loss %f' % (epoch + 1, train_loss[epoch]))
    print('w的L2范数是：', torch.norm(w).item())
    print("w: ", w)

if __name__ == '__main__':
    train(30)