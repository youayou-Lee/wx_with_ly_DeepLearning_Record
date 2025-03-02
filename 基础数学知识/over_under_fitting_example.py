import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 生成数据
def generate_data(n_samples=100, noise=0.5):
    x = torch.linspace(-3, 3, n_samples)
    y = 0.5 * x ** 3 - 2 * x ** 2 + 0.5 * x + torch.randn(n_samples) * noise
    return x, y


# 定义多项式模型
class PolynomialModel(nn.Module):
    def __init__(self, degree):
        super(PolynomialModel, self).__init__()
        self.degree = degree
        self.weights = nn.Parameter(torch.randn(degree + 1))

    def forward(self, x):
        y = torch.zeros_like(x)
        for i in range(self.degree + 1):
            y += self.weights[i] * (x ** i)
        return y


# 训练模型
def train_model(model, x, y, epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


# 可视化结果
def plot_results(x, y, model, title):
    plt.scatter(x.numpy(), y.numpy(), label='Data')
    x_plot = torch.linspace(-3, 3, 100)
    y_plot = model(x_plot).detach().numpy()
    plt.plot(x_plot.numpy(), y_plot, label='Model', color='red')
    plt.title(title)
    plt.legend()
    plt.show()


# 生成数据
x, y = generate_data()

# # 欠拟合：使用低阶多项式（degree=1）
# model_underfit = PolynomialModel(degree=1)
# train_model(model_underfit, x, y)
# plot_results(x, y, model_underfit, 'Underfitting (Degree=1)')
#
# # 正常拟合：使用适当阶数的多项式（degree=3）
model_goodfit = PolynomialModel(degree=3)
train_model(model_goodfit, x, y, lr=6e-4)
plot_results(x, y, model_goodfit, 'Good Fit (Degree=3)')

# 过拟合：使用高阶多项式（degree=7）
# model_overfit = PolynomialModel(degree=5)
# train_model(model_overfit, x, y, lr=6e-7)
# plot_results(x, y, model_overfit, 'Overfitting (Degree=10)')