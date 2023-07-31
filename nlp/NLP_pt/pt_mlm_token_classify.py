import operator
import torch
from transformers import BertTokenizerFast, BertForMaskedLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizerFast.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
model.to(device)

texts = ["这时亿个文本纠错的安例"]

text_tokens = tokenizer(texts, padding=True, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = model(**text_tokens)

for ids, (i, text) in zip(outputs.logits, enumerate(texts)):
    _text = tokenizer.decode((torch.argmax(ids, dim=-1) * text_tokens.attention_mask[i]), skip_special_tokens=True).replace(' ', '')
    print("_text: ", _text)
