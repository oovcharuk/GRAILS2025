
import pandas as pd
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset

from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    data = pd.read_csv('content/train.csv')
    print(data.head())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].tolist(),
        data['new_label'].tolist(),
        test_size=0.2,
        random_state=42
    )

    model_name = "microsoft/deberta-v3-small"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = PTSDDataset(train_texts, train_labels, tokenizer)
    val_dataset = PTSDDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        print(f"Epoch {epoch + 1}/{3}")
        train_model(model, train_loader, optimizer)

class PTSDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Training loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
