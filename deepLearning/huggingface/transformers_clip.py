from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel


txt_pretrained_model_name = "openai/clip-vit-large-patch14"
model = CLIPModel.from_pretrained(txt_pretrained_model_name)
processor = CLIPProcessor.from_pretrained(txt_pretrained_model_name)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text=["a photo of a dog", "a photo of a cat"]

inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

print(outputs.image_embeds.shape)
print(outputs.text_embeds.shape)

logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
print(text[probs.argmax(-1).item()])
