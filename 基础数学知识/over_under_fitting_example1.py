import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_data(poly_dim=20, num_data=1000):
    # 生成单一特征x，并创建多项式特征
    x = torch.normal(0, 1, size=(num_data, 1)).to(device)  # (num_data, 1)
    poly_features = torch.zeros(num_data, poly_dim).to(device)
    for i in range(poly_dim):
        poly_features[:, i] = x.squeeze() ** i  # 第i列是x^i

    # 归一化：除以i!
    for i in range(poly_dim):
        poly_features[:, i] /= math.gamma(i + 1)

    # 真实权重：前4维非零
    true_w = torch.zeros(poly_dim).to(device)
    true_w[0:4] = torch.tensor([5, 1.2, -3.4, 5.6])

    # 生成标签
    labels = poly_features @ true_w + torch.normal(0, 0.1, size=(num_data,)).to(device)
    return poly_features, labels


def evaluate_loss(net, data_iter, loss):
    all_loss = []
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        out = net(X)
        l = loss(out, y.reshape(out.shape))
        all_loss.append(l.mean().item())
    return sum(all_loss) / len(all_loss)


def train(train_features, test_features, train_labels, test_labels, num_epochs=100, lr=0.1, you=4):
    loss = torch.nn.MSELoss()
    batch_size = 100

    # 使用前you个特征
    train_f = train_features[:, :you].to(device)
    test_f = test_features[:, :you].to(device)

    # 模型：包含偏置项（由第0维特征x^0=1充当）
    net = torch.nn.Linear(you, 1, bias=False).to(device)
    torch.nn.init.normal_(net.weight, mean=0, std=0.01)

    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ls, test_ls = [], []

    for epoch in range(num_epochs):
        # 随机打乱数据
        indices = torch.randperm(train_f.size(0))
        train_f_shuffled = train_f[indices]
        train_labels_shuffled = train_labels[indices]

        for i in range(0, train_f.size(0), batch_size):
            X = train_f_shuffled[i:i + batch_size]
            y = train_labels_shuffled[i:i + batch_size]
            trainer.zero_grad()
            l = loss(net(X), y.reshape(-1, 1))
            l.backward()
            trainer.step()

        # 记录损失
        train_ls.append(loss(net(train_f), train_labels.reshape(-1, 1)).item())
        test_ls.append(loss(net(test_f), test_labels.reshape(-1, 1)).item())
        print(f'Epoch {epoch}, Train Loss: {train_ls[-1]:.4f}, Test Loss: {test_ls[-1]:.4f}')

    # 绘制损失曲线和预测结果
    plt.plot(train_ls, label='Train Loss')
    plt.plot(test_ls, label='Test Loss')
    plt.legend()
    plt.show()

    # 绘制预测与标签对比（取第一个多项式特征，即x）
    plt.scatter(train_f[:, 1].cpu(), train_labels.cpu(), label='True')
    plt.scatter(train_f[:, 1].cpu(), net(train_f).detach().cpu(), label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_data, train_labels = generate_data(poly_dim=20)
    test_data, test_labels = generate_data(poly_dim=20)
    train(train_data, test_data, train_labels, test_labels, num_epochs=100, lr=0.1, you=4)