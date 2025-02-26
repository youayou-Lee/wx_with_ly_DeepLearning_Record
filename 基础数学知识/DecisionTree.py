import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# load data
train_data = pd.read_csv('../dataset/titanic/train.csv')

# print("train data shape: ",train_data.shape)
# print(train_data.head())

# 数据信息总览
# print(train_data.info())
"""
[5 rows x 12 columns]
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
None

首先，我们发现PassengerID、Name、Ticket这三个特征只做标识乘客的作用，与是否幸存无关，所以我们去掉这两个特征。
另外，通过输出数据信息可知，age、Cabin、Embarked均存在缺失值的情况，尤其是Cabin，缺失了大部分信息，所以我们暂且先丢弃这个特征。
Embarked是港口信息，我们需要将其转换为数值型数据。

"""
# 指定第一行作为行索引
train_data = pd.read_csv("../dataset/titanic/train.csv", index_col=0)
# 丢弃无用的数据
train_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
# print(train_data.head())
# 处理性别数据
train_data["Sex"] = (train_data["Sex"] == "male").astype(int)

labels = train_data["Embarked"].unique().tolist()
train_data["Embarked"] = train_data["Embarked"].apply(lambda x: labels.index(x))
train_data = train_data.fillna(0)
# print(train_data.head())
# print(train_data.info())

# 拆分数据集
from sklearn.model_selection import train_test_split

y = train_data["Survived"].values
X = train_data.drop(["Survived"], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

# model train
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print("train score:", train_score, "test score:", test_score)

# 模型参数调优
import numpy as np
def cv_score(d):
    """
    在不同depth值下，train_score和test_score的值
    :param
    d: max_depth值
    :return: (train_score, test_score)
    """
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    return (train_score, test_score)

# 指定参数的范围， 训练模型计算得分
depths = range(2,15)
scores = [cv_score(d) for d in depths]
train_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

# 找出交叉验证集评分最高的模型参数
best_score_index = np.argmax(cv_scores)
best_depth = depths[best_score_index]
best_score = cv_scores[best_score_index]

print("best_depth:", best_depth, "best_score:", best_score)

"""
    参数调优可视化
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4), dpi=200)
plt.grid()
plt.xlabel("Max depth of Decision Tree")
plt.ylabel("score")
plt.plot(depths, cv_scores, ".g--", label="cross validation score")
plt.plot(depths, train_scores, ".r--", label="training score")
plt.legend()
plt.show()
