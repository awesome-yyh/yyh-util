from PIL import Image
import requests
# import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, CLIPProcessor, CLIPModel
import numpy as np


txt_pretrained_model_name = "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"
text_tokenizer = BertTokenizer.from_pretrained(txt_pretrained_model_name)
text_encoder = BertForSequenceClassification.from_pretrained(txt_pretrained_model_name).eval()

img_pretrained_model_name = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(img_pretrained_model_name)
processor = CLIPProcessor.from_pretrained(img_pretrained_model_name)

query_texts = ["一只猫", "一只狗", '两只猫', '两只老虎', '一只老虎']  # 这里是输入文本的，可以随意替换。
text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 这里可以换成任意图片的url
image = processor(images=Image.open(requests.get(url, stream=True).raw), return_tensors="pt")
# inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    image_features = clip_model.get_image_features(**image)
    text_features = text_encoder(text).logits
    print(image_features.shape)
    print(text_features.shape)
    # 归一化
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # 计算余弦相似度 logit_scale是尺度系数
    logit_scale = clip_model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print(query_texts[np.argmax(probs)])
