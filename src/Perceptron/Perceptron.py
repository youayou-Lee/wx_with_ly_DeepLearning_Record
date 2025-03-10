import torch
import torch.nn as nn
import torch.optim as optim


# 定义感知机模型
class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# 生成一些简单的二分类数据
# 假设我们有两个特征，数据点为 2D
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

# 初始化模型、损失函数和优化器
input_dim = 2
model = Perceptron(input_dim)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    outputs = model(X)
    loss = criterion(outputs, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    predicted = model(X).round()
    print(f'Predicted: {predicted.squeeze().numpy()}')
    print(f'Actual: {y.squeeze().numpy()}')