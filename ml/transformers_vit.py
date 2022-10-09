from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests


pretrained_model_name = 'google/vit-base-patch16-224'
feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name)
model = ViTForImageClassification.from_pretrained(pretrained_model_name)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs).logits
print(outputs.shape)

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = outputs.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
