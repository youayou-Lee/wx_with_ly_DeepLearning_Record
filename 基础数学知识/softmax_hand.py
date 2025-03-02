import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 图片的尺寸是 28*28 = 784， 通道数为 3，但是 softmax的输入必须是一个向量，所以此处需要对图片进行展平
num_inputs = 784
# 数据集有 10 个类别，故输出维度为 10
num_outputs = 10

W = torch.normal(0, 0.01, (num_inputs, num_outputs), device=device, requires_grad=True)
b = torch.zeros(num_outputs, device=device, requires_grad=True)
def softmax(X):
    """input size: [m * n], outpu size: [m * 1]"""
    X_exp = torch.exp(X)
    # sum 中axis = 1，表示对每一行进行求和，keepdim=True 表示返回的 Tensor 与 X 的形状相同
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制 因为两个矩阵size不一样，通过broadcast机制使其size一致

def linear_softmax(X):
    """确保 X 与 W 能进行矩阵乘法，需要把X的 shape[1] 调整为 W的 shape[0]"""
    X.to(device)
    linear = torch.matmul(X.reshape((-1, W.shape[0])), W) + b
    return softmax(linear)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # 如果y_hat 维数大于1，列数大于1，则 取每一行的最大值 作为预测值
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# 修正后的评估准确率函数
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 评估模式
    total_correct, total_samples = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            total_correct += accuracy(y_hat, y)
            total_samples += y.size(0)  # 当前batch的样本数
    return total_correct / total_samples  # 正确数目 / 总样本数

# 修正后的训练单个epoch函数
def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 训练模式
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()  # 计算平均损失梯度
            updater.step()
            total_loss += l.sum().item()
        else:
            l.sum().backward()
            updater(X.shape[0])  # 假设updater内部会处理梯度缩放
            total_loss += l.sum().item()
        total_correct += accuracy(y_hat, y)
        total_samples += y.size(0)
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    # 新增：记录训练过程的列表
    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1}:")
        # 修改：获取每个epoch的平均损失和准确率
        avg_loss, avg_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        # 记录数据
        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        test_accs.append(test_acc)

        print(f"Train loss: {avg_loss:.4f}, Train acc: {avg_acc:.3f}, Test acc: {test_acc:.3f}")

    return train_losses, train_accs, test_accs

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10

if __name__ == '__main__':

    # 运行训练并获取数据
    train_losses, train_accs, test_accs = train(linear_softmax, train_iter, test_iter, cross_entropy, num_epochs, updater)

    # 绘制结果
    plt.figure(figsize=(12, 4))

    # 子图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    # 子图2：准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
    plt.plot(range(1, num_epochs+1), test_accs, label='Test Acc', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()