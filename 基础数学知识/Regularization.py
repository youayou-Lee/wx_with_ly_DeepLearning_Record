import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成示例数据
np.random.seed(0)
n_samples, n_features = 100, 10
X = np.random.randn(n_samples, n_features)  # 输入特征 (100x10)
true_coef = np.random.randn(n_features)     # 真实权重 (10x1)
y = X.dot(true_coef) + np.random.randn(n_samples) * 0.5  # 目标输出 (100x1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Ridge回归（L2正则化）
alpha = 1.0  # 正则化系数
ridge = Ridge(alpha=alpha)
ridge.fit(X_train, y_train)

# 预测
y_pred = ridge.predict(X_test)

# 可视化
plt.figure(figsize=(10, 6))

# 绘制真实值和预测值
plt.scatter(range(len(y_test)), y_test, color='blue', label='True Values', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Values', alpha=0.6)

# 添加标签和标题
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('True Values vs Predicted Values (Ridge Regression)')
plt.legend()

# 显示图形
plt.show()

# 输出均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")