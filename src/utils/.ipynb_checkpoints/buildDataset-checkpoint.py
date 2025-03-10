import os
import torch
import torchvision.utils
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

class MyDataset(Dataset):
    def __init__(self, img_path, transform=None):
        super(MyDataset, self).__init__()
        self.img_path = img_path
        self.txt_path = img_path + '\\' + 'data.txt'
        with open(self.txt_path, 'r') as f:
            all_data = f.readlines()
        self.imgs, self.labels = [], []
        for line in all_data:
            word = line.strip().split(' ')
            self.imgs.append(os.path.join(img_path, word[2], word[0]))
            self.labels.append(int(word[1]))
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = torch.from_numpy(np.array(self.labels[index]).astype(np.int64))
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

if __name__ == '__main__':
    path = r"/dataset/flower_photos"
    you_dataset = MyDataset(path, transforms)
    data_loader = DataLoader(you_dataset, batch_size=32, shuffle=True)
    for i, (img, label) in enumerate(data_loader):
        print(img.shape)
        print(label.shape)
        img = torchvision.utils.make_grid(img).numpy()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.show()
        break