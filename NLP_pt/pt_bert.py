# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from transformers import BertTokenizer, BertConfig, BertModel


# 设置环境
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu') # mac m1 gpu: mps

# 设置训练参数和模型参数
batch_size = 3
epoches = 1
MODEL_PATH = "bert-base-chinese"
hidden_size = 768
n_class = 2
maxlen = 8


# 构造样本格式
class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True,):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.sentences = sentences
        self.labels = labels
        self.with_labels = with_labels
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]

        encoded_pair = self.tokenizer(sent,
                        padding='max_length',
                        truncation=True,
                        max_length=maxlen,  
                        return_tensors='pt')
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

# 创建模型
class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        model_config = BertConfig.from_pretrained(MODEL_PATH) # 加载模型超参数
        # model_config.output_hidden_states = True
        # model_config.output_attentions = True
        model_config.return_dict = True
        self.bert = BertModel.from_pretrained(MODEL_PATH, config = model_config) # 加载模型
        self.linear = nn.Linear(hidden_size, n_class) # 直接用cls向量接全连接层分类
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids) # 返回一个output字典
        print(outputs.keys())
        logits = self.linear(self.dropout(outputs.pooler_output)) # [bs, hidden_size]
        # ['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions']
        # output_hidden_states: 最后一层输出的隐藏状态，（通常用于命名实体识别）
        # pooler_output: 将[CLS]这个token再过一下全连接层+Tanh激活函数，作为该句子的特征向量，通常用于句子分类
        # hidden_states: 第一个元素是embedding，其余元素是各层的输出
        # attentions: 每一层的注意力权重，用于计算self-attention heads的加权平均值

        return logits


if __name__ == '__main__': # 这个地方可以解决多线程的问题
    
    # data，构造一些训练数据
    sentences = ["我喜欢打篮球", "这个相机很好看", "今天玩的特别开心", "我不喜欢你", "太糟糕了", "真是件令人伤心的事情"]
    labels = [1, 1, 1, 0, 0, 0]  # 1积极, 0消极.
    train_data = Data.DataLoader(dataset=MyDataset(sentences, labels),
                                 batch_size=batch_size, shuffle=True, num_workers=1)

    bc = BertClassify().to(device)
    optimizer = optim.Adam(bc.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # 训练模型
    train_curve = []
    total_step = len(train_data) # = 总样本量 / batch_size
    for epoch in range(epoches):
        sum_loss = 0
        for i, batch in enumerate(train_data):
            batch = tuple(p.to(device) for p in batch)
            pred = bc([batch[0], batch[1], batch[2]])
            loss = loss_fn(pred, batch[3])
            sum_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'epoch:[{epoch+1}|{epoches}] step:{i+1}/{total_step} loss:{loss.item():.4f}')
        train_curve.append(sum_loss)

    # test
    bc.eval()
    with torch.no_grad():
        test_text = ['我不喜欢打篮球']
        test = MyDataset(test_text, labels=None, with_labels=False)
        x = test.__getitem__(0)
        x = tuple(p.unsqueeze(0).to(device) for p in x) # 增加一维，因为是要按batch输入模型
        pred = bc([x[0], x[1], x[2]])
        pred = pred.data.max(dim=1, keepdim=True)[1]
        if pred[0][0] == 0:
            print('消极')
        else:
            print('积极')

    # pd.DataFrame(train_curve).plot() # loss曲线
    # plt.show()

