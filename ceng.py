import torch
from torch import nn
from torch.nn import functional as F

#net = nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

X = torch.rand(2,20)
#net(X)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))#nn.ReLU()是构造了一个ReLU对象，不是函数，是class，而F.ReLU()是函数调用

#net = MLP()
#net(X)

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx,module in enumerate(args):
            #enumerate是Python内置函数，用于在遍历可迭代对象（如列表、元组）时，同时获取元素的索引和值。例如，对于一个列表['a', 'b', 'c']，使用enumerate会生成(0, 'a'), (1, 'b'), (2, 'c')这样的元组序列。
            #PyTorch的nn.Module类有一个特殊属性_modules，它是一个有序字典（OrderedDict），
            # 用于跟踪子模块。
            #将模块添加到_modules中后，PyTorch会自动管理这些模块的参数（如梯度更新、设备移动等）。
            #使用str(idx)将索引转换为字符串，确保键的类型一致性
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print(net(X))



