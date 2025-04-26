import hashlib
import os
import tarfile
import zipfile
import requests


#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data =pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(train_data.shape)
print(test_data.shape)

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))#把没用的特征都删掉
print(all_features.shape)

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index#把非字符串的数据对应的index取出来
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)#不是数字的地方用0填，用0填充缺失值等价于用原数据的均值填充，标准化后，数据分布均值为0

#将数据中的分类特征（非数值型）转换为 虚拟变量（Dummy Variables），即生成一组二进制列（0或1），表示原始特征中每个类别的存在性。
all_features = pd.get_dummies(all_features, dummy_na=True)#将缺失值 NaN 视为一个有效的特征类别，并为其生成独立的虚拟变量列,这样强制让get_dummies转为int类型
all_features = all_features*1.0
print(all_features.shape)

n_train = train_data.shape[0]  #shape[0] 返回其行数（即训练样本的数量）。
train_features = torch.tensor(all_features[:n_train].values,dtype = torch.float32)
test_features = torch.tensor(all_features[n_train:].values,dtype = torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

def log_rmse(net, features, labels):#对数均方根误差
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    #net 是一个定义好的神经网络,net(features) 的作用是获取模型对输入特征 features 的预测值。
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    train_iter = d2l.load_array((train_features,train_labels),batch_size)
    #将训练数据封装为批次迭代器。d2l.load_array 功能类似于PyTorch的 DataLoader输入为 (features, labels) 元组，输出为可迭代对象，每次返回一个批次的数据 (X, y)。
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay= weight_decay)#Adam优化器，更平滑，对学习率没那么敏感
    #net.parameters(): 待优化的模型参数，weight_decay: L2正则化系数（权重衰减），防止过拟合
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()#每个批次开始前清空优化器中的梯度，防止梯度累积。
            l = loss(net(X),y)
            l.backward()
            optimizer.step()#优化器根据梯度更新模型参数
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

def get_k_fold_data(k, i, X, y):# i为当前选中的折索引（范围 [0, k-1]），该折数据作为验证集
    assert k > 1
    fold_size = X.shape[0] // k  #样本数整除（//）K， fold_size每折多少个样本。，//是Python中的整除运算符，它会自动向下取整，得到整数结果。
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)#通过 slice 生成当前折 j 的索引范围。例如：j=0 → 索引 0:fold_size
        X_part, y_part = X[idx, :], y[idx]
        if j == i:   #验证集：当 j == i 时，当前折作为验证集
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:   #后续非 i 的折通过 torch.cat 拼接在训练集末尾
            X_train = torch.cat([X_train, X_part], 0)  #沿行方向（第0维度）拼接张量。假设 X_train 原为 (N, D)（N个样本，D个特征），X_part 为 (M, D)，拼接后变为 (N+M, D)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k,i,X_train,y_train)
        net = get_net()
        train_ls ,valid_ls = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum += train_ls[-1]   #train函数返回的是训练和验证损失数组，数组中每个元素是每一轮计算的损失。这里-1代表只取最后一轮损失,取每一折最后一epoch的loss，在训练过程的最后一个epoch，模型通常已接近收敛，参数已经基本稳定
        valid_l_sum += valid_ls[-1]   # 1 表示取列表最后一个元素（即最后一个 epoch 的损失值）
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
#假设 num_epochs=10，k=5：
#训练过程：每折训练 10 个 epoch，每 epoch 记录一次损失。
#绘图：仅在 i=0（第一折）绘制 train_ls 和 valid_ls 的 10 个 epoch 的曲线。
#损失累加：每折取第 10 个 epoch 的损失值，5 折后计算平均损失。

k,num_epochs,lr,weight_dacay,batch_size = 5,100,5,0,64
train_l,valid_l = k_fold(k,train_features,train_labels,num_epochs,lr,weight_dacay,batch_size)
print(f'{k}-折验证：平均训练log rmse:{float(train_l):f},'f'平均验证log rmse:{float(valid_l):f}')