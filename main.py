import torch
import torch.nn.functional as F
from model.PTSDTextClassifier import PTSDTextClassifier
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def main():

    text = "Through work I have been in some dodgy situations abroad. A number of times my life has been at risk, and I've had to defend myself a few times. I'm a civilian and have only had basic weapon training. &#x200B; We've always had an ex-military security contractor with us, but still had to perform aggressive roles when approached."

    classifier = PTSDTextClassifier()
    classifier.predict_with_attention(text)

    model_path = './trained_models/NarcissisticDisorder/trained_model'
    target_label = 'Narcissistic Disorder'
    predict_label(text, model_path, target_label)

    model_path = './trained_models/AnxietyDisorder/trained_model'
    target_label = 'Anxiety Disorder'
    predict_label(text, model_path, target_label)

    model_path = './trained_models/Depression/trained_model'
    target_label = 'Depression'
    predict_label(text, model_path, target_label)
    
    model_path = './trained_models/PanicDisorder/trained_model'
    target_label = 'Panic Disorder'
    predict_label(text, model_path, target_label)
    
    model_path = './trained_models/AngerIntermittentExplosiveDisorder/trained_model'
    target_label = 'Anger Intermittent Explosive Disorder'
    predict_label(text, model_path, target_label)

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
    return target_probability

if __name__ == "__main__":
    main()
