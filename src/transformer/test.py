import torch
from transformers import BertTokenizer
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW

class YouDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset.map(self.preprocess_data, batched=True)

    def preprocess_data(self, data):
        return tokenizer(data['text'], truncation=True, padding='max_length', max_length=512)

    def __getitem__(self, item):
        return {
            'input_ids': torch.tensor(self.dataset[item]['input_ids']),
            'attention_mask': torch.tensor(self.dataset[item]['attention_mask']),
            'labels': torch.tensor(self.dataset[item]['label'])
        }
    def __len__(self):
        return len(self.dataset)

def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs.logits, dim=1)
            correct += (predicted == labels).sum().item()

            total += labels.size(0)
        print('Accuracy: {:.2f}%'.format(100 * correct / total))


batch_size = 16
tokenizer = BertTokenizer.from_pretrained(r'G:\CS\models\bert\bert-base-chinese')

dataset = load_from_disk(r'G:\CS\Datasets\ChnSentiCorp')
train_data = dataset['train'].select(range(2400))
test_data = dataset['test']
train_iter = DataLoader(YouDataset(train_data), batch_size=batch_size, shuffle=True)
test_iter = DataLoader(YouDataset(test_data), batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained(r'G:\CS\models\bert\bert-base-chinese', num_labels=2)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
# 设置训练轮数
epochs = 3

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_iter:
        # 将数据移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 打印每个 epoch 的平均损失
    avg_loss = total_loss / len(train_iter)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')
    evaluate(model, test_iter)