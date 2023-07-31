# -*- coding: utf-8 -*-
import os
import re
import matplotlib.pyplot as plt
import numpy as np


folder_path = "/Users/yaheyang/mypython/boundary-detector-for-text-correction/logs"
nohup_files = []
error_files = []
# 遍历目录下所有文件和子目录
for dirpath, dirnames, filenames in os.walk(folder_path):
    # 按文件名排序
    filenames.sort()
    for filename in filenames:
        # 判断文件名是否以nohup开头
        if filename.endswith(".log") and filename not in error_files:
            # 将文件路径添加到数组中
            nohup_files.append(os.path.join(dirpath, filename))
# 打印所有找到的nohup文件路径
print(nohup_files)

index = 0
min_loss = 2.175
min_loss_epoch = set()
max_loss = 2.23
max_loss_epoch = set()
epochs = []
losses = []
losses_list = []
losses_list_statr_step = []
valid_loss = []
nll_losses = []
ppls = []
wps = []
ups = []
wpbs = []
bszs = []
num_updates = []
lrs = []
gnorms = []
clips = []
loss_scales = []
walls = []
train_walls = []
step = 0

train_loss_list = []
train_p_list = []
train_r_list = []
train_f1_list = []
valid_loss_list = []
valid_p_list = []
valid_r_list = []
valid_f1_list = []
for log_file in nohup_files:
    print(log_file)
    with open(log_file, "r", encoding='utf-8') as f:
        for line in f:
            if "_train" in line:
                loss_match = re.search("avg_loss=([0-9.]+)", line)
                if loss_match:
                    train_loss_list.append(float(loss_match.group(1)))
                p_match = re.search("avg_loss=([0-9.]+)", line)
                if p_match:
                    train_p_list.append(float(p_match.group(1)))
                r_match = re.search("avg_loss=([0-9.]+)", line)
                if r_match:
                    train_r_list.append(float(r_match.group(1)))
                f1_match = re.search("avg_loss=([0-9.]+)", line)
                if f1_match:
                    train_f1_list.append(float(f1_match.group(1)))
            if "_test" in line:
                loss_match = re.search("avg_loss=([0-9.]+)", line)
                if loss_match:
                    valid_loss_list.append(float(loss_match.group(1)))
                p_match = re.search("avg_loss=([0-9.]+)", line)
                if p_match:
                    valid_p_list.append(float(p_match.group(1)))
                r_match = re.search("avg_loss=([0-9.]+)", line)
                if r_match:
                    valid_r_list.append(float(r_match.group(1)))
                f1_match = re.search("avg_loss=([0-9.]+)", line)
                if f1_match:
                    valid_f1_list.append(float(f1_match.group(1)))

# print("目前总的epoch:", epochs[-1])
# print("每个文件的loss开始的step: ", losses_list_statr_step)
# print("目前训练集最小loss: ", min(losses))
# print("目前验证集最小loss: ", min(valid_loss))

# 画图
fig, ax = plt.subplots(1, 4, figsize=(13, 5))
ax[0].plot(train_loss_list[:100], label='train')
ax[0].plot(valid_loss_list[:100], label='valid')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(train_p_list[:100], label='train')
ax[1].plot(valid_p_list[:100], label='valid')
ax[1].set_ylabel('p')
ax[1].legend()

ax[2].plot(train_r_list[:100], label='train')
ax[2].plot(valid_r_list[:100], label='valid')
ax[2].set_ylabel('r')
ax[2].legend()

ax[3].plot(train_f1_list[:100], label='train')
ax[3].plot(valid_f1_list[:100], label='valid')
ax[3].set_ylabel('f1')
ax[3].legend()

plt.show()
