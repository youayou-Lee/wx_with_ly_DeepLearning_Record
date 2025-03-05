## 2025-2.26

### 随机梯度下降，梯度下降，小批量梯度下降

今天根据李沐的《动手写深度学习》中 MBGD，修改代码，实现BGD发现：

- **学习率的调整** ：对于BGD，损失函数计算的是所有样本的损失，所以loss对w的偏导值会很大，如果沿用MBGD的lr，会发现loss无法收敛，必须降低lr。
- **MBGD的w更新**：对于每一轮epoch，都需要计算一次grad，然后只更新一次，计算grad开销实在太大，而MBGD对小批量的样本计算grad则开销小得多，且每一轮循环可以多次更新w。lr 为 1 算正常，不需要较多轮次便可完成训练
- **矩阵运算 torch.matmul()**：pytorch中的一维数组是以列向量为数学计算约定，而以行向量为表示形式的向量 ,因为此时的size 为 [m] 的tensor 实际上是 列向量

## 2025-2.28

### debug 多线程问题

报错：这有一段很长的报错，请问是为什么：Traceback (most recent call last):
File "<string>", line 1, in <module>
File "G:\Anaconda\envs\you\Lib\multiprocessing\spawn.py", line 122, in spawn_main
exitcode = _main(fd, parent_sentinel)

... 截取最顶端和最低端的报错

File "G:\Anaconda\envs\you\Lib\site-packages\torch\utils\data\dataloader.py", line 1144, in _try_get_data
raise RuntimeError(f'DataLoader worker (pid(s) {pids_str}) exited unexpectedly') from e
RuntimeError: DataLoader worker (pid(s) 30128, 18392, 21084, 11432) exited unexpectedly

这个错误是典型的Windows系统下多进程编程问题，主要由于没有正确使用if __name__ == '__main__'保护主程序入口导致。以下是具体原因和解决方案：

问题原因分析： Windows多进程机制限制

Windows使用spawn方式创建子进程（而非Linux的fork方式），
子进程会重新导入主模块，导致循环执行代码，
需要if __name__ == '__main__'保护主程序入口
DataLoader多进程冲突：
设置了num_workers > 0时会启用多进程加载数据， 主程序中没有正确使用保护机制导致进程冲突

### softmax函数实现时，为什么分母按行求和不按列求和？

样本维度：

1. 在 softmax 中，我们通常是对每个样本的特征进行归一化，而不是对所有样本的同一特征进行归一化。因此，我们需要对每个样本的特征值进行求和，即对每一行进行求和。
2. 如果按列求和，我们会将所有样本的同一特征值相加，这会导致不同样本的特征值相互影响，这不符合 softmax 的设计初衷。

归一化的目标：

1. softmax 的目标是对每个样本的特征进行归一化，使得每个样本的特征值在 0 到 1 之间，并且所有特征值的和为 1。因此，我们需要对每个样本的特征值进行求和，即对每一行进行求和。
2. 如果按列求和，我们会得到所有样本的同一特征值的总和，这并不能帮助我们实现每个样本的特征归一化。

## 2025-3.1~3.3

### 过拟合和欠拟合

- **1.数据集是一堆Tensor该如何处理？**：对于数据集是一堆Tensor，比如通过torch.normal()随机生成，可以选择先 构造 TensorDatasetm对象，再构造 DataLoader对象

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

train_f = torch.normal(0, 1, (1000, 2))
train_labels = torch.where(train_f[:, 0] + train_f[:, 1] > 1, 1, 0)
train_data = TensorDataset(train_f, train_labels)
train_iter = DataLoader(train_data, 10, shuffle=True)
```

- **2.损失函数的传参size问题**：尽量确保 loss 的input size一致，否则可能得出意料之外的结果

```python
import torch
loss = torch.nn.MSELoss()
mat1 = torch.normal(0,1,(100,1))
mat2 = torch.normal(0,1,(1,100))
loss(mat1, mat2)
#UserWarning: Using a target size (torch.Size([1, 100])) that is different to the input size (torch.Size([100, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
#return F.mse_loss(input, target, reduction=self.reduction)
# Out[6]: tensor(2.1837)
loss(mat1, mat2.T)
#tensor(2.5413)
```

就是这个问题导致我的预测结果 基本保持在一个高度，所以需要明确对于不同size的tensor 调用loss的区别。------------这个任务待定

- **3.通过神经网络拟合数据的流程**：
  1. 构造数据集：可以选择直接构造tensor，也可以选网络上的数据集。

```python
import torch, sys
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 256


def load_data_fashion_mnist(batch_size, resize=None):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=0 if sys.platform == 'win32' else 4,
                   pin_memory=torch.cuda.is_available()),
        DataLoader(test, batch_size, shuffle=False, num_workers=0 if sys.platform == 'win32' else 4,
                   pin_memory=torch.cuda.is_available())
    )
```

2. 定义模型，可以选择自己定义也可以用torch的nn，后者需要继承nn.Module，并且需要重写__init__方法和forward方法

```python
from torch import nn
import torch
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
```

- **4.矩阵求高次项式**：torch.pow函数
- **5.dataset严重不足时该怎么办**：采用k折交叉验证

### weight_dacay L2正则化

通过weight_decay参数限制参数更新时的权重衰减，从而达到防止过拟合的效果。 （此处不是很理解，代码仍需上手重复实践）

### 为什么说dropout等同于正则化？

## 2025-3.4

### 为什么合理的模型权重初始化 会让数值更稳定？

### **梯度消失的示例**

假设我们有一个简单的 5 层全连接神经网络，每层只有一个神经元，激活函数为 Sigmoid。

- 权重初始化过小，比如 $( W = 0.01 $)。
- 输入值 $( x = 1 $)。

---

#### **前向传播**

每一层的输出为 $( z = W \cdot a $)，激活值为 $( a = \sigma(z) $)，其中 $( \sigma $) 是 Sigmoid 函数。
Sigmoid 函数的导数为 $( \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z)) $)，最大值仅为 0.25。

- **第 1 层**：
  $( z_1 = 0.01 \cdot 1 = 0.01 )$
  $( a_1 = \sigma(0.01) \approx 0.5025 $)
- **第 2 层**：
  $( z_2 = 0.01 \cdot 0.5025 = 0.005025 $)
  $( a_2 = \sigma(0.005025) \approx 0.50125 $)
- **第 3 层**：
  $( z_3 = 0.01 \cdot 0.50125 = 0.0050125 $)
  $( a_3 = \sigma(0.0050125) \approx 0.50125 $)
- **依此类推**，激活值逐渐接近 0.5。

---

#### **反向传播**

假设损失函数为 $( L $)，反向传播时梯度为：
$[\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial a_i} \cdot \sigma'(z_i) \cdot a_{i-1}]$

由于 $( \sigma'(z_i) \leq 0.25 $)，且每一层的梯度会乘以 $( \sigma'(z_i) $)，梯度会逐渐衰减：

- **第 1 层梯度**：
  $( \frac{\partial L}{\partial W_1} \approx \frac{\partial L}{\partial a_1} \cdot 0.25 \cdot 1)$
- **第 2 层梯度**：
  $( \frac{\partial L}{\partial W_2} \approx \frac{\partial L}{\partial a_2} \cdot 0.25 \cdot 0.5025 ) $
- **第 3 层梯度**：
  $( \frac{\partial L}{\partial W_3} \approx \frac{\partial L}{\partial a_3} \cdot 0.25 \cdot 0.50125 )$
- **依此类推**，梯度会指数级衰减，最终接近 0，导致权重无法有效更新。

---

#### 模型权重初始化关于 每一层的input,output都控制在相同的方差，均值为0的随机分布的公式推导

$\sum_j Var(w_j^t) Var({h_j}^{t-1}) = n_{t-1}\gamma_t Var({h_j}^{t-1}$)

3.5解答：首先弄清楚期望和方差的概念：现在一个随机变量，可能取到1,2,3,4,5中的任意一个数，概率一样，那么期望是指定为：$E[X] = \sum_i^5{X_i}P(X_i)$，而方差是指定：$Var[X] = E[(X-E[X])^2]$
根据上述期望公式可得，$E[X] = 5,D[X] = 2$
假设现在有n个随机变量，$X_i$之间独立同分布，那么$E(X_1) = E(X_2) = ... = E(X_n)，$且$D(X_1) = D(X_2) = D(X_n)$

---

### BF16 是什么？ TF16 又是什么？

* **BF16** 和 **TF16** 都是 16 位浮点数格式，主要用于深度学习和高性能计算。

---

### sigmoid容易引起梯度消失，尝试使用relu

Sigmoid 函数的公式为：

$\sigma(x)=\frac1{1+e^{-x}}$其输出范围是(0, 1)，形状为 S 形曲线。
求导：$\sigma\prime(x) = \sigma(x)(1-\sigma(x))$

**Sigmoid 引起梯度消失的原因​**

梯度消失问题是指在反向传播过程中，梯度逐渐变小，最终接近于 0，导致模型参数无法有效更新。Sigmoid 函数容易引起梯度消失的原因如下：

1. ​**梯度值较小**​：
   * Sigmoid 的导数 **σ**′**(**x**)** 的最大值是 0.25，且当输入 **x** 的绝对值较大时，**σ**′**(**x**)** 会趋近于 0。
   * 在反向传播中，梯度是通过链式法则逐层相乘的。如果每一层的梯度都小于 1，多层相乘后梯度会变得非常小。
2. ​**饱和现象**​：
   * 当输入 **x** 的绝对值较大时，Sigmoid 的输出会接近 0 或 1，此时梯度几乎为 0。
   * 这种饱和现象会导致神经元的参数更新非常缓慢，甚至停止更新。

---

### 数据集中有字符串的处理方法：

1. all_features = pd.get_dummies(all_features, dummy_na=True)

`pd.get_dummies(all_features, dummy_na=True)` 这段代码会将 `all_features` 中的分类变量（通常是字符串或类别型数据）转换为虚拟变量（dummy variables），并且会处理缺失值（`dummy_na=True`）。这个过程会导致 `all_features` 的 size 发生变化，原因如下：

### 1. **虚拟变量的生成**

- 对于 `all_features` 中的每一个分类变量（如性别、颜色等），`pd.get_dummies` 会将其转换为多个二进制列（0 或 1），每一列代表该分类变量的一个可能取值。
- 例如，如果有一个分类变量 `color`，其取值为 `['red', 'green', 'blue']`，`pd.get_dummies` 会生成三列：`color_red`、`color_green` 和 `color_blue`。每一列的值表示该行是否属于该类别（1 表示是，0 表示否）。

### 2. **处理缺失值**

- 当 `dummy_na=True` 时，`pd.get_dummies` 会为缺失值（NaN）生成额外的列。例如，如果 `color` 列中有缺失值，`pd.get_dummies` 会生成一列 `color_nan`，表示该行是否缺失。

### 3. **size 的变化**

- 假设原始数据有 `n` 列，其中 `k` 列是分类变量。每个分类变量有 `m_i` 个不同的取值（包括缺失值），那么 `pd.get_dummies` 会将这些分类变量转换为 `sum(m_i)` 列。
- 因此，`all_features` 的列数会增加，导致 size 发生变化。

### 示例

假设 `all_features` 有以下数据：

| color  | size |
|--------|------|
| red    | S    |
| green  | M    |
| blue   | L    |
| NaN    | M    |

执行 `pd.get_dummies(all_features, dummy_na=True)` 后，数据会变为：

| color_red | color_green | color_blue | color_nan | size_L | size_M | size_S |
|-----------|-------------|------------|-----------|--------|--------|--------|
| 1         | 0           | 0          | 0         | 0      | 0      | 1      |
| 0         | 1           | 0          | 0         | 0      | 1      | 0      |
| 0         | 0           | 1          | 0         | 1      | 0      | 0      |
| 0         | 0           | 0          | 1         | 0      | 1      | 0      |

可以看到，原始数据有 2 列，转换后有 7 列，size 明显增加了。

### 总结

`pd.get_dummies` 通过将分类变量转换为虚拟变量，并处理缺失值，导致 `all_features` 的列数增加，从而改变了其 size。

2. 假设字符串只包含["red","blue","green"], 那么可以采用index, 将对应列的数据转为index

---



