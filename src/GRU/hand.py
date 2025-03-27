import torch
from torch import nn


batch_size, num_steps = 32, 35
sentences = torch.randn(size=(320,10,10), dtype=torch.float32, requires_grad=True)
