'''
Author: yyh owyangyahe@126.com
Date: 2023-08-22 15:20:56
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2025-06-26 11:35:05
FilePath: /mypython/yyh-util/deep_learning/pytorch/pt_lr_scheduler.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import BertModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from deepspeed.runtime.lr_schedules import WarmupCosineLR
import matplotlib.pyplot as plt


pretrained_model_name = "hfl/chinese-roberta-wwm-ext"
model = BertModel.from_pretrained(pretrained_model_name)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

trans_cos_lr = []
trans_cos_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=1000)

trans_linear_lr = []
trans_linear_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=50, num_training_steps=1000)

torch_cos_anneal_lr = []
torch_cos_anneal_scheduler = CosineAnnealingLR(optimizer, T_max=100)

torch_cos_anneal_scheduler_warm_lr = []
torch_cos_anneal_scheduler_warm = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

ds_cos_lr = []
ds_cos_lr_scheduler = WarmupCosineLR(optimizer,
                            total_num_steps=1000,
                            warmup_num_steps=50,
                            warmup_type="linear")


for epoch in range(1000):
    trans_cos_scheduler.step()
    trans_cos_lr.append(trans_cos_scheduler.get_last_lr())
    
    trans_linear_scheduler.step()
    trans_linear_lr.append(trans_linear_scheduler.get_last_lr())
    
    torch_cos_anneal_scheduler.step()
    torch_cos_anneal_lr.append(torch_cos_anneal_scheduler.get_last_lr())
    
    torch_cos_anneal_scheduler_warm.step()
    torch_cos_anneal_scheduler_warm_lr.append(torch_cos_anneal_scheduler_warm.get_last_lr())
    
    ds_cos_lr_scheduler.step()
    ds_cos_lr.append(ds_cos_lr_scheduler.get_last_lr())

# plt.plot(trans_cos_lr, label='trans_cos_lr')
plt.plot(trans_linear_lr, label='trans_linear_lr')
# plt.plot(torch_cos_anneal_lr, label='torch_cos_anneal_lr')
# plt.plot(torch_cos_anneal_scheduler_warm_lr, label='torch_cos_anneal_scheduler_warm_lr')
plt.plot(ds_cos_lr, label='ds_cos_lr')

plt.legend()  # 显示图示
plt.show()
