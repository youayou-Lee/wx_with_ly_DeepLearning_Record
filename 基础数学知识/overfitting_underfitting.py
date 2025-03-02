import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 设置设备为 CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_data(dim=20, num_data=1000):

    true_w = torch.zeros(dim).to(device)  # 将 true_w 移动到 GPU
    true_w[0:4] = torch.tensor([5, 1.2, -3.4, 5.6]).to(device)

    x = torch.normal(0, 1, size=(num_data, 1)).to(device)  # 将 features 移动到 GPU

    # 第 n 项 为 X^(n-1)
    poly_features = torch.pow(x, torch.arange(dim).to(device))  # 将 poly_features 移动到 GPU
    # poly_features[:, 4:] = 0

    for i in range(dim):
        poly_features[:, i] /= math.gamma(i + 1)

    temp = poly_features @ true_w
    labels = temp + torch.normal(0, 0.1, size=temp.shape).to(device)  # 将 labels 移动到 GPU
    return poly_features, labels


def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    all_loss = []
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        out = net(X)
        l = loss(out, y.reshape(out.shape))
        all_loss.append(l.mean().item())
    return sum(all_loss) / len(all_loss)

def train(train_features, test_features, train_labels, test_labels, num_epochs=20,lr=0.5, you=20):
    loss = torch.nn.MSELoss()
    batch_size = min(100, train_labels.shape[0])

    train_features = train_features[:,:you]
    test_features = test_features[:,:you]

    net = torch.nn.Sequential(torch.nn.Linear(train_features.shape[-1], 1, bias=False)).to(device)  # 将模型移动到 GPU

    train_data = TensorDataset(train_features, train_labels)
    test_data = TensorDataset(test_features, test_labels)
    train_iter = DataLoader(train_data, batch_size,shuffle=True)
    test_iter = DataLoader(test_data, batch_size,shuffle=True)

    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)  # 将 X 和 y 移动到 GPU
            trainer.zero_grad()
            output = net(X)
            # print("predic: ", output)
            l = loss(output, y)
            l.sum().backward()
            trainer.step()
            train_ls.append(l.sum().item())
        # test_ls.append(l.sum())
        # print('epoch %d, train loss %f, test loss %f' % (epoch, train_ls[-1], test_ls[-1]))
        print('epoch %d, train loss %f' % (epoch, train_ls[-1]))
    print("weight: ", net[0].weight.data.cpu().numpy())  # 将权重数据移回 CPU 并转换为 numpy 数组

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), train_ls, label='Train Loss')
    # plt.plot(range(num_epochs), test_ls, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over Epochs')
    plt.legend()
    plt.show()

    # 绘制数据点的坐标位置（x轴为 feature[:, 1]，y轴为 labels 和 loss）
    plt.figure(figsize=(10, 5))
    plt.scatter(train_features[:, 1].cpu().numpy(), train_labels.cpu().numpy(), c='blue', label='Labels')
    plt.scatter(train_features[:, 1].cpu().numpy(), net(train_features).detach().cpu().numpy(), c='red', label='Predictions')
    plt.xlabel('Feature[:, 1]')
    plt.ylabel('Labels / Predictions')
    plt.title('Data Points: Feature[:, 1] vs Labels and Predictions')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train_data, train_labels = generate_data()
    test_data, test_labels = generate_data()

    train(train_data, test_data, train_labels, test_labels,num_epochs=100, lr=0.05, you=4)