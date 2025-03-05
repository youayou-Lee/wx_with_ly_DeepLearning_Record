import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成一些正态分布的样本数据
np.random.seed(0)
mu_true = 5.0  # 真实均值
sigma_true = 2.0  # 真实标准差
sample_size = 1000
data = np.random.normal(mu_true, sigma_true, sample_size)

# 定义负对数似然函数
def neg_log_likelihood(params, data):
    mu, sigma = params
    n = len(data)
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)
    return -log_likelihood  # 返回负对数似然

# 初始猜测值
initial_guess = [0.0, 1.0]

# 使用 minimize 函数来最小化负对数似然函数
result = minimize(neg_log_likelihood, initial_guess, args=(data,), bounds=[(None, None), (1e-10, None)])

# 输出结果
mu_mle, sigma_mle = result.x
print(f"估计的均值 (MLE): {mu_mle:.4f}")
print(f"估计的标准差 (MLE): {sigma_mle:.4f}")

# 可视化
plt.figure(figsize=(10, 6))

# 绘制原始数据的直方图
plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='原始数据直方图')

# 绘制真实正态分布的 PDF
x = np.linspace(min(data), max(data), 1000)
true_pdf = norm.pdf(x, mu_true, sigma_true)
plt.plot(x, true_pdf, 'r-', linewidth=2, label=f'真实分布 ($\mu$={mu_true}, $\sigma$={sigma_true})')

# 绘制 MLE 估计的正态分布的 PDF
mle_pdf = norm.pdf(x, mu_mle, sigma_mle)
plt.plot(x, mle_pdf, 'b--', linewidth=2, label=f'MLE 估计分布 ($\mu$={mu_mle:.2f}, $\sigma$={sigma_mle:.2f})')

# 添加图例和标签
plt.legend()
plt.title('极大似然估计 (MLE) 可视化')
plt.xlabel('数据值')
plt.ylabel('概率密度')
plt.grid(True)
plt.show()