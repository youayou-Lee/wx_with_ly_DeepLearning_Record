# NumPy + Pytorch

## transpose 转置  
```python
import numpy as np
arr = np.arange(16).reshape((2, 2, 4))
arr.transpose((1, 0, 2))
```
## pytorch中的一维数组，是列向量还是行向量？
pytorch中的一维数组是以列向量为数学计算约定，而以行向量为表示形式的向量

所以torch.matmul()函数允许 size 为 [n*m] 和 [m]的tensor 进行乘法运算
因为此时的size 为 [m] 的tensor 实际上是 列向量