from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re


MODEL_DIR = "./mbert_finetuned_news"


print("Loading trained model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)


def clean_text_test(text):
text = text.lower()
text = re.sub(r"[^\w\s\-']", " ", text)
return re.sub(r"\s+", " ", text).strip()




def predict_headline(headline: str):
cleaned = clean_text_test(headline)
enc = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
logits = model(**enc).logits
pred = torch.argmax(logits, dim=1).item()


# Load id2label mapping
import json
with open(MODEL_DIR + "/id2label.json", "r") as f:
id2label = json.load(f)


return id2label[str(pred)]


print("Testing module ready.")
