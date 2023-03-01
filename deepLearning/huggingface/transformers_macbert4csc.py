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


def get_errors(corrected_text, origin_text):
    sub_details = []
    for i, ori_char in enumerate(origin_text):
        if ori_char in [' ', '“', '”', '‘', '’', '\n', '…', '—', '擤']:
            # add unk word
            corrected_text = corrected_text[:i] + ori_char + corrected_text[i:]
            continue
        if i >= len(corrected_text):
            break
        if ori_char != corrected_text[i]:
            if ori_char.lower() == corrected_text[i]:
                # pass english upper char
                corrected_text = corrected_text[:i] + ori_char + corrected_text[i + 1:]
                continue
            sub_details.append((ori_char, corrected_text[i], i, i + 1))
    sub_details = sorted(sub_details, key=operator.itemgetter(2))
    return corrected_text, sub_details


result = []
for ids, (i, text) in zip(outputs.logits, enumerate(texts)):
    _text = tokenizer.decode((torch.argmax(ids, dim=-1) * text_tokens.attention_mask[i]),
                             skip_special_tokens=True).replace(' ', '')
    corrected_text, details = get_errors(_text, text)
    # print(text, ' => ', corrected_text, details)
    result.append((corrected_text, details))
print(result)  # [('这是一个文本纠错的案例', [('时', '是', 1, 2), ('亿', '一', 2, 3), ('安', '案', 9, 10)])]