import torch
from torch import nn
from matplotlib import pyplot as plt
import collections


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def evaluate_accuracy(net, data_iter, device=device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            correct += (net(X).argmax(axis=1) == y).sum().item()
            total += y.size(0)
    return correct / total

def train(net, train_iter, test_iter, num_epochs, lr=0.5, device=device, evaluate_accuracy=evaluate_accuracy):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)

    net.to(device)
    # 定义优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # 记录每一轮的损失，训练集的正确率，测试集的正确率
    train_epoch_loss, train_epoch_acc, test_epoch_acc = [], [], []
    for epoch in range(num_epochs):
        train_loss = 0
        correct, total = 0, 0
        net.train()

        for X, y in train_iter:
            # t1 = time.time()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            # 此处的损失是平均损失，所以不用sum
            l = loss(y_hat, y)
            train_loss += l.item()
            l.backward()
            optimizer.step()
            # t2 = time.time()
            correct += (y_hat.argmax(axis=1) == y).sum().item()
            total += y.size(0)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, train loss {train_loss / len(train_iter)}, train acc {correct / total}, test acc {test_acc}')
        train_epoch_loss.append(train_loss / len(train_iter))
        train_epoch_acc.append(correct / total)
        test_epoch_acc.append(test_acc)

    plt.plot(range(1, num_epochs + 1), train_epoch_loss, label='Train Loss')
    plt.show()
    plt.plot(range(1, num_epochs + 1), train_epoch_acc, label='Train Acc')
    plt.plot(range(1, num_epochs + 1), test_epoch_acc, label='Test Acc')
    plt.show()


def train_regression(net, train_iter, loss, num_epochs, lr, device=device):
    def evaluate_loss(data_iter, net, loss, device):
        net.eval()
        loss_sum, n = 0.0, 0
        with torch.no_grad():
            for X, y in data_iter:
                X, y = X.to(device), y.to(device)
                l = loss(net(X), y)
                loss_sum += l.sum().item()
                n += y.numel()
        return loss_sum / n

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net = net.to(device)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print(f'epoch {epoch + 1}, loss {evaluate_loss(train_iter, net, loss, device)}')

def load_array(features, labels, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*(features, labels))
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def load_seq_data():
    import re
    path = r"..\seq.txt"
    with open(path, 'r') as f:
        lines = f.readlines()
    all_data = list(filter(lambda x: x != '',[re.sub('[^A-Za-z.]+', ' ', line).strip().lower() for line in lines]))

    you = Vocab(all_data)
    return you

class Vocab:

    def __init__(self, words):
        self.words = words
        # 把所有词放入一个list中，然后计算每个词出现的次数
        words = [word for token_list in words for word in token_list.split(" ")]
        counter = collections.Counter(words)
        unk_times = len([x for x in counter.items() if x[1] == 1])
        self._token_freqs = list(filter(lambda x: x[1] > 1, counter.items()))
        self._token_freqs.append(('<unk>', unk_times))
        self._token_freqs = sorted(self._token_freqs, key=lambda x: x[1], reverse=True)

    def __len__(self):
        return len(self._token_freqs)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._token_freqs[item][0]
        if isinstance(item, str):
            return self._token_freqs.index(item)

if __name__ == '__main__':
    print(load_seq_data()._token_freqs)