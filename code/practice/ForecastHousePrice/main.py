import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

test_data = pd.read_csv(r"F:\code\AI\wx_with_ly_DeepLearning_Record\code\practice\data\kaggle_house_pred_test.csv")
train_data = pd.read_csv(r"F:\code\AI\wx_with_ly_DeepLearning_Record\code\practice\data\kaggle_house_pred_train.csv")

# 数据清洗，需要先把数据拼接，然后一起处理
# 训练集最后一列 是 label，所以需要扔掉， 两个数据第一列都是 id ，所以都扔掉
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 将数值重新缩放到零均值，单位方差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 将所有缺失值替换为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理字符串
# 采用one-hot编码 我个人觉得不是一个很好的方法，我想用index来替换
# all_features = pd.get_dummies(all_features, dummy_na=True)

string_columns = all_features.select_dtypes(include=["object"]).columns
for s in string_columns:
    all_features[s], _ = pd.factorize(all_features[s])

n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values, dtype=np.float32)
test_features = np.array(all_features[n_train:].values, dtype=np.float32)
train_labels = np.array(train_data.SalePrice.values.reshape(-1, 1), dtype=np.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = DataLoader(TensorDataset(train_features, train_labels),batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    """将数据分为k部分，取第i部分作为验证集"""
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plt.plot(range(1, num_epochs + 1),
                     train_ls, label='Train')
            plt.plot(range(1, num_epochs + 1),
                     valid_ls, label='Valid')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    # train_ls, _ = train(get_net(), torch.tensor(np.array(all_features[:n_train]), dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32), test_features, None, num_epochs=100, learning_rate=0.01, weight_decay=0, batch_size=64)
    # plt.plot(range(100), train_ls, label='Train Loss')
    # plt.show()
    k_fold(5, torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.float32), 100, 5, 0, 64)