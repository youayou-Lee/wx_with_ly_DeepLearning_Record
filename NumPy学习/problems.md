# NumPy 函数 

## transpose 转置  
```python
import numpy as np
arr = np.arange(16).reshape((2, 2, 4))
arr.transpose((1, 0, 2))
```