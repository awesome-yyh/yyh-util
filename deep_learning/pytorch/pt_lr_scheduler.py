import torch
from transformers import BertModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt


pretrained_model_name = "hfl/chinese-roberta-wwm-ext"
model = BertModel.from_pretrained(pretrained_model_name)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

cos_lr = []
cos_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=1000)

linear_lr = []
linear_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=50, num_training_steps=1000)

for epoch in range(1000):
    cos_scheduler.step()
    cos_lr.append(cos_scheduler.get_last_lr())
    
    linear_scheduler.step()
    linear_lr.append(linear_scheduler.get_last_lr())

plt.plot(cos_lr, label='cosine_schedule_with_warmup')
plt.plot(linear_lr, label='linear_schedule_with_warmup')

plt.legend()  # 显示图示
plt.show()
