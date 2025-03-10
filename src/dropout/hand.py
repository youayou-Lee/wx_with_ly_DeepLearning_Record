import torch
from torch import nn

def drop_out_layer(X, drop_prob):
    assert 0 <= drop_prob <= 1

    if drop_prob == 1:
        return torch.zeros_like(X)
    if drop_prob == 0:
        return X
    mask = (torch.rand(X.shape)> drop_prob).float()
    return mask * X / (1.0 - drop_prob)