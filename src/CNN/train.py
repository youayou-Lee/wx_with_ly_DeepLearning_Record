import torch
from torch import nn
def corr2d(X, K):
    """
    :param X: input
    :param K: kernel
    :return: output
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h,j:j+w] * K).sum()
    return Y
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,2),bias=False)
X = torch.ones((6, 8))
X[:, 2:6] = 0
K = torch.tensor([[1, -1]])
Y = corr2d(X, K)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(50):
    Y_hat = conv2d(X)
    loss = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    loss.sum().backward()
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {loss.sum():.3f}')

        print(conv2d.weight.data)

