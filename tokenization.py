import re
import pandas as pd
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128


def clean_text(text: str) -> str:
if pd.isna(text):
return ""
text = text.lower()
text = re.sub(r"https?://\S+|www\.\S+", " ", text)
text = re.sub(r"[^\w\s\-']", " ", text)
text = re.sub(r"\s+", " ", text).strip()
return text


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)




def tokenize_batch(text_list):
cleaned = [clean_text(t) for t in text_list]
return tokenizer(cleaned, truncation=True, max_length=MAX_LENGTH, padding=False)


print("Tokenization module ready.")
