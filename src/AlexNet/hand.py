from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from src.utils.buildDataset import MyDataset
from src.utils.train import train

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

net = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
    )
# 代码需要补上，但是暂时没有对应数据集，所以待定
if __name__ == '__main__':
    img_path = r'F:\code\AI\wx_with_ly_DeepLearning_Record\dataset\flower_photos'
    data_set = MyDataset(img_path, transform)
    train_dataset, test_dataset = random_split(data_set, [0.8, 0.2])
    train_iter = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_iter = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    train(net, train_iter, test_iter, num_epochs=30, lr=0.1)
