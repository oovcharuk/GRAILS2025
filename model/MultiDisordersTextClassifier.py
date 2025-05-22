import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

class MultiDisordersTextClassifier:
    def __init__(self, dataset_path, output_dir, target_label):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.target_label = target_label
        os.environ["WANDB_DISABLED"] = "true"
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def load_and_prepare_data(self):
        data = pd.read_csv(self.dataset_path)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        self.train_dataset = Dataset.from_pandas(train_data)
        self.test_dataset = Dataset.from_pandas(test_data)

        self.train_dataset = self.train_dataset.map(self.tokenize_function, batched=True)
        self.test_dataset = self.test_dataset.map(self.tokenize_function, batched=True)

        self.train_dataset = self.train_dataset.remove_columns(['Text'])
        self.test_dataset = self.test_dataset.remove_columns(['Text'])

        self.train_dataset = self.train_dataset.map(self.format_labels, batched=True)
        self.test_dataset = self.test_dataset.map(self.format_labels, batched=True)

        self.train_dataset.set_format('torch')
        self.test_dataset.set_format('torch')

    def tokenize_function(self, examples):
        return self.tokenizer(examples['Text'], truncation=True, padding='max_length', max_length=128)

    def format_labels(self, examples):
        examples['label'] = [1 if label == self.target_label else 0 for label in examples['label']]
        return examples

    def train_model(self):
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_dir, 'results'),
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Evaluation Results: {eval_results}")

        model.save_pretrained(os.path.join(self.output_dir, 'trained_model'))
        self.tokenizer.save_pretrained(os.path.join(self.output_dir, 'trained_model'))

    def compute_metrics(self, p):
        preds = p.predictions.argmax(-1)
        return {'accuracy': accuracy_score(p.label_ids, preds)}

    def run(self):
        self.load_and_prepare_data()
        self.train_model()