from PIL import Image
import requests
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch


pretrained_model_name = "microsoft/resnet-50"
feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name)
model = ResNetForImageClassification.from_pretrained(pretrained_model_name)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    print(logits.shape)

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
