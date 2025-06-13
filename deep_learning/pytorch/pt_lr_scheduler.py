'''
Author: yyh owyangyahe@126.com
Date: 2023-08-22 15:20:56
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-09-05 08:06:57
FilePath: /mypython/yyh-util/deep_learning/pytorch/pt_lr_scheduler.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
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

cos_anneal_lr = []
cos_anneal_scheduler = CosineAnnealingLR(optimizer, T_max=100)

cos_anneal_scheduler_warm_lr = []
cos_anneal_scheduler_warm = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

for epoch in range(1000):
    cos_scheduler.step()
    cos_lr.append(cos_scheduler.get_last_lr())
    
    linear_scheduler.step()
    linear_lr.append(linear_scheduler.get_last_lr())
    
    cos_anneal_scheduler.step()
    cos_anneal_lr.append(cos_anneal_scheduler.get_last_lr())
    
    cos_anneal_scheduler_warm.step()
    cos_anneal_scheduler_warm_lr.append(cos_anneal_scheduler_warm.get_last_lr())

plt.plot(cos_lr, label='cosine_schedule_with_warmup')
plt.plot(linear_lr, label='linear_schedule_with_warmup')
plt.plot(cos_anneal_lr, label='cos_anneal_lr')
plt.plot(cos_anneal_scheduler_warm_lr, label='cos_anneal_scheduler_warm_lr')

plt.legend()  # 显示图示
plt.show()
