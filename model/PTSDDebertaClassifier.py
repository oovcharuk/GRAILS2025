import pandas as pd
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
from bertviz import head_view, model_view
import torch
from torch.utils.data import DataLoader

from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from PTSDDataset import PTSDDataset

class PTSDDebertaClassifier:
    def __init__(self, model_name="microsoft/deberta-v3-small", model_path="./", device=None):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(self.device)

    def load_data(self, csv_path, text_column="text", label_column="new_label", test_size=0.2):
        data = pd.read_csv(csv_path)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data[text_column].tolist(),
            data[label_column].tolist(),
            test_size=test_size,
            random_state=42
        )
        self.train_dataset = PTSDDataset(train_texts, train_labels, self.tokenizer)
        self.val_dataset = PTSDDataset(val_texts, val_labels, self.tokenizer)

    def train(self, epochs=3, batch_size=8, learning_rate=2e-5):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            self._train_model(train_loader, optimizer)
            self._evaluate_model(val_loader)

        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)
        print(f"Model and tokenizer saved to {self.model_path}")

    def _train_model(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Training loss: {avg_loss:.4f}")

    def _evaluate_model(self, val_loader):
        self.model.eval()
        total_accuracy = 0
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                total_accuracy += (predictions == labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader.dataset)
        print(f"Validation loss: {avg_loss:.4f}, Validation accuracy: {avg_accuracy:.4f}")

    def predict_with_attention(self, text: str):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)
        self.model = DebertaV2ForSequenceClassification.from_pretrained(self.model_path, output_attentions=True)
        self.model.to(self.device)

        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        attention = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        head_view(attention, tokens)

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        predicted_probs = probs.detach().cpu().numpy()

        print(f"Predicted class: {predicted_class}")
        print(f"Probability of PTSD: {predicted_probs[0][1]:.4f}" if predicted_probs.shape[1] > 1 else predicted_probs)

        explainer = SequenceClassificationExplainer(self.model, self.tokenizer)
        explainer(text)
        attributions = explainer.word_attributions
        print(attributions)
        explainer.visualize()
