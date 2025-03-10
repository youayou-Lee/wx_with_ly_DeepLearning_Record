import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)
net_relu = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16*5*5, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10)
)

# load dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 256
train_data = datasets.FashionMNIST(root=r'F:\code\AI\wx_with_ly_DeepLearning_Record\data', train=True, download=False, transform=transforms.ToTensor())
test_data = datasets.FashionMNIST(root=r'F:\code\AI\wx_with_ly_DeepLearning_Record\data', train=False, download=False, transform=transforms.ToTensor())
train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    train_correct, train_total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
            train_correct += (net(X).argmax(axis=1) == y).sum().item()
            train_total += y.numel()
    # return metric[0] / metric[1]
    return train_correct / train_total

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)

    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    train_e_loss, train_e_acc, test_e_acc = [], [], []
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        train_l = 0
        train_acc = 0

        net.train()

        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            train_l += l.item()
            train_acc += (y_hat.argmax(axis=1) == y).sum().item()

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        trian_acc = evaluate_accuracy_gpu(net, train_iter)
        print(f'epoch {epoch + 1}, ', f'loss {train_l / len(train_iter):.3f}')
        print(f'train acc {trian_acc:.3f}, test_acc {test_acc:.3f}')
        train_e_loss.append(train_l / len(train_iter))
        train_e_acc.append(trian_acc)
        test_e_acc.append(test_acc)
    plt.plot(range(1, num_epochs + 1), train_e_loss, label='Train Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()

    plt.plot(range(1, num_epochs + 1), train_e_acc, label='Train Acc')
    plt.plot(range(1, num_epochs + 1), test_e_acc, label='Test Acc')
    plt.show()


lr, num_epochs = 0.9, 10

# train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
train_ch6(net_relu, train_iter, test_iter, num_epochs, lr=0.1, device=d2l.try_gpu())

