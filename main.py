import os
import sys
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QPushButton, QLabel, QVBoxLayout
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from model.PTSDTextClassifier import PTSDTextClassifier

RESULT_FOLDER = "./result_interpretation"
COMORBID_MODELS = [
    ("./trained_models/NarcissisticDisorder/trained_model", "Narcissistic Disorder"),
    ("./trained_models/AnxietyDisorder/trained_model", "Anxiety Disorder"),
    ("./trained_models/Depression/trained_model", "Depression"),
    ("./trained_models/PanicDisorder/trained_model", "Panic Disorder"),
    ("./trained_models/AngerIntermittentExplosiveDisorder/trained_model", "Anger Intermittent Explosive Disorder"),
]

def load_model_and_tokenizer(model_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def predict_label(text, model_path, target_label):
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).squeeze().tolist()
    target_probability = round(probabilities[1] * 100, 2)
    print(f"Probability of '{target_label}': {target_probability}%")
    return f"Probability of '{target_label}': {target_probability}%"

def calculate_result(text):
    result_lines = []
    classifier = PTSDTextClassifier()
    PTSDprobability = classifier.predict_with_attention(text)

    if PTSDprobability > 50:
        result_lines.append(f"Probability of PTSD: {PTSDprobability}%\n")
        result_lines.append("Risk of Comorbid Disorders:\n")
        for model_path, label in COMORBID_MODELS:
            result_lines.append(predict_label(text, model_path, label))
        return "\n".join(result_lines)
    else:
        return "Probability of having PTSD is low."

class TextAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Echo Within")
        self.setGeometry(100, 100, 400, 300)

        self.text_edit = QTextEdit(self)
        self.button = QPushButton("Check text", self)
        self.open_folder_button = QPushButton("Open result", self)
        self.result_label = QLabel("", self)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.button)
        layout.addWidget(self.open_folder_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.button.clicked.connect(self.analyze_text)
        self.open_folder_button.clicked.connect(self.open_project_folder)

    def analyze_text(self):
        input_text = self.text_edit.toPlainText()
        result = calculate_result(input_text)
        self.result_label.setText(result)

    def open_project_folder(self):
        project_path = os.path.abspath(RESULT_FOLDER)
        if os.path.exists(project_path):
            os.startfile(project_path)
        else:
            self.result_label.setText("Folder not found!")

def main():
    app = QApplication(sys.argv)
    window = TextAnalysisApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
