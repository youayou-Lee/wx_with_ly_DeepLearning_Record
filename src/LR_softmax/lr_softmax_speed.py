import torch
from torch import nn
import sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 超参数设置
batch_size = 512  # 增大批处理规模
num_epochs = 10
lr = 0.1


# 数据加载优化
def load_data_fashion_mnist(batch_size, resize=None):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=0 if sys.platform == 'win32' else 4,  # Windows下禁用多进程
    pin_memory=torch.cuda.is_available()),
        DataLoader(test, batch_size, shuffle=False, num_workers=0 if sys.platform == 'win32' else 4,  # Windows下禁用多进程
    pin_memory=torch.cuda.is_available())
    )


train_iter, test_iter = load_data_fashion_mnist(batch_size)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # 启用cuDNN基准优化

# 模型定义（使用内置模块）
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
).to(device)

# 损失函数和优化器（使用内置函数）
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)


# 训练函数（优化后）
def train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer):
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        net.train()
        total_loss = total_correct = total_samples = 0

        for X, y in train_iter:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # 前向传播
            y_hat = net(X)
            l = loss(y_hat, y)

            # 反向传播
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            # 统计指标
            total_loss += l.detach().cpu().item() * y.size(0)
            total_correct += (y_hat.argmax(axis=1) == y).sum().item()
            total_samples += y.size(0)

        # 验证集评估
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        test_acc = evaluate_accuracy(net, test_iter)

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1}: Train loss {train_loss:.4f}, Train acc {train_acc:.3f}, Test acc {test_acc:.3f}")

    return train_losses, train_accs, test_accs


# 评估函数
def evaluate_accuracy(net, data_iter):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            correct += (net(X).argmax(axis=1) == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == '__main__':
    print("开始训练...")
    # 运行训练
    train_losses, train_accs, test_accs = train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)

    # 可视化结果
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc', linestyle='--')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()