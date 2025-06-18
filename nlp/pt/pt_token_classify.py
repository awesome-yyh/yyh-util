
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


MODEL_PATH = "/data/app/base_model/ckiplab-bert-base-chinese-ws"
MODEL_PATH = "/data/app/base_model/ckiplab-bert-base-chinese-ner"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

text = "这个宾馆比较陈旧了，特价的房间也很一般。"

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
