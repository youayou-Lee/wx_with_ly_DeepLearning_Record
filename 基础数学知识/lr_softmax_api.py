import torch, sys
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 256


def load_data_fashion_mnist(batch_size, resize=None):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size, shuffle=True, num_workers=0 if sys.platform == 'win32' else 4,
                   pin_memory=torch.cuda.is_available()),
        DataLoader(test, batch_size, shuffle=False, num_workers=0 if sys.platform == 'win32' else 4,
                   pin_memory=torch.cuda.is_available())
    )


train_iter, test_iter = load_data_fashion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 10)
)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.zeros_(m.bias)


net.apply(init_weights)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10

# 训练模型并记录数据
train_losses = []
test_accuracies = []

# 类别名称
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for epoch in range(num_epochs):
    # 训练阶段
    net.train()
    total_loss = 0.0
    total_samples = 0

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        total_loss += l.item() * X.size(0)  # 累加损失
        total_samples += X.size(0)

    avg_loss = total_loss / total_samples
    train_losses.append(avg_loss)

    # 测试阶段
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            predicted = y_hat.argmax(axis=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    test_accuracies.append(test_acc)

    print(f'Epoch {epoch + 1:02d}: Loss={avg_loss:.4f}, Test Acc={test_acc:.3f}')

# 可视化训练损失和测试准确率
plt.figure(figsize=(12, 4))

# 训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'o-', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()

# 测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, 'o-', label='Test Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Curve')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 可视化部分测试样本的预测结果
net.eval()
with torch.no_grad():
    # 获取一个测试batch
    X_test, y_test = next(iter(test_iter))
    y_pred = net(X_test).argmax(axis=1)

# 显示25个样本的预测结果
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    img = X_test[i].squeeze().numpy()
    plt.imshow(img, cmap='gray')

    pred_label = classes[y_pred[i]]
    true_label = classes[y_test[i]]

    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()