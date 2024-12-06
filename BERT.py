import os
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# 设置环境变量
os.environ['HF_MIRROR'] = 'https://hf-mirror.com'

# 加载与配置模型
print("加载数据...")
raw_dataset = load_dataset('ag_news')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
classification_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classification_model.to(device)

# 模型参数
lr = 2e-5
batch_size = 32
max_seq_len = 128
num_epochs = 30


# 定义自定义数据集
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encoded_data = tokenizer(texts,
                                      truncation=True,
                                      padding='max_length',
                                      max_length=max_len,
                                      return_tensors='pt')
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {key: value[index] for key, value in self.encoded_data.items()}
        sample['labels'] = self.labels[index]
        return sample


# 数据预处理
print("预处理数据...")
train_texts = raw_dataset['train']['text']
train_labels = raw_dataset['train']['label']
test_texts = raw_dataset['test']['text']
test_labels = raw_dataset['test']['label']

train_data = TextClassificationDataset(train_texts, train_labels, tokenizer, max_seq_len)
test_data = TextClassificationDataset(test_texts, test_labels, tokenizer, max_seq_len)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# 数据分析
def visualize_data():
    lengths = [len(text.split()) for text in train_texts]
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, color='skyblue')
    plt.title('分词后文本长度分布')
    plt.xlabel('单词数')
    plt.ylabel('频次')
    plt.savefig('length_distribution.png')
    plt.close()


# 模型训练
optimizer = AdamW(classification_model.parameters(), lr=lr)


def train_one_epoch():
    classification_model.train()
    epoch_loss = 0
    progress = tqdm(train_loader, desc='训练中')

    for batch in progress:
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = classification_model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})

    return epoch_loss / len(train_loader)


# 模型评估
def evaluate_model():
    classification_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='评估中'):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = classification_model(inputs, attention_mask=masks).logits
            predictions = torch.argmax(logits, dim=-1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds,
                                target_names=['World', 'Sports', 'Business', 'Tech']))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['World', 'Sports', 'Business', 'Tech'],
                yticklabels=['World', 'Sports', 'Business', 'Tech'])
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return all_preds, all_labels


# 主程序
if __name__ == "__main__":
    print("开始分析数据...")
    visualize_data()

    print(f"训练模型 (设备: {device})...")
    min_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = train_one_epoch()
        print(f"平均损失: {avg_loss:.4f}")

        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(classification_model.state_dict(), 'best_model.pth')

    # 评估模型
    classification_model.load_state_dict(torch.load('best_model.pth'))
    predictions, ground_truth = evaluate_model()
