import torch
from torch import nn
from d2l import torch as d2l
from src.utils.train import load_array, train_regression
from matplotlib import pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# plt.plot(time, x)
# plt.show()

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]

labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
train_iter = load_array(features[:n_train], labels[:n_train], batch_size, is_train=True)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

net = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
net.apply(init_weights)

loss = nn.MSELoss()
train_regression(net, train_iter, loss, 5, 0.01)

def predict(net, features):
    features = features.to(device)
    net.eval()
    with torch.no_grad():
        output = net(features)
    return output.detach().cpu().numpy()



def pred_next_next(net, features):
    next = torch.from_numpy(predict(net, features))
    new_features = torch.zeros_like(features)
    new_features[:, :3] = features[:, 1:]
    new_features[:,3] = next[:, 0]
    return new_features

new_features = pred_next_next(net, features[:n_train])
inp = features[:n_train]
for _ in range(20):
    out = pred_next_next(net, inp)
    inp = out


plt.figure(figsize=(100, 5))
plt.scatter(time[tau + 1: n_train + tau + 1], predict(net, features[:n_train]), label='prediction', color='blue', s=10)
plt.scatter(time[tau + 1: n_train + tau + 1], x[tau:n_train + tau], label='True Value', color='green', s=10)
plt.scatter(time[tau + 2: n_train + tau + 2], predict(net, new_features), label='next_next', color='red', s=10)
plt.scatter(time[tau + 20: n_train + tau + 20], predict(net, out), label='20 Value', color='yellow', s=10)
plt.show()