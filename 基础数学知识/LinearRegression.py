import torch
import random

# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    # torch.normal(mean, std, size)
    X = torch.normal(0, 1, (num_examples, len(w)))
    # torch.matmul(X, w) 矩阵乘法 而非点乘 (m*n) * (n*p) = (m*p)
    y = torch.matmul(X, w) + b
    # 加上噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 读取数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机读取
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]



# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方误差"""
    # y_hat 是预测值， y是 实际值
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降 MBGD"""
    # params 是 [w, b]
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_MBGD(batch_size, lr, num_epochs, w, b):
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X和y的小批量损失
            # 因为l形状是(batch_size,1)，而不是一个标量。l.sum()计算l元素的和。
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

def final_show():
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')

    print("w", w)
    print("b", b)


def train_BGD(lr, num_epochs, w, b):
    for epoch in range(num_epochs):
        l = loss(net(features, w, b), labels)
        l.sum().backward()

        # 梯度下降
        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()
            b -= lr * b.grad
            b.grad.zero_()
            print(f'epoch {epoch + 1}, loss {float(loss(net(features, w, b), labels).mean()):f}')

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print("features: ", features[0], "\nlabel: ", labels[0])

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
batch_size = 5

lr = 1     # BGD 6e-5
num_epochs = 10   # BGD 100
net = linreg
loss = squared_loss


if __name__ == '__main__':
    # train_BGD(lr, num_epochs, w, b)
    train_MBGD(batch_size, lr, num_epochs, w, b)
    final_show()



