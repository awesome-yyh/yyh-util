import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import transformers
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_scheduler


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


# 设置环境
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
print("CPU or GPU: ", device)
seed_everything(42)


# 设置训练参数和模型参数
epoch_num = 5
batch_size = 2
learning_rate = 1e-5
MODEL_PATH = "bert-base-chinese"
# MODEL_PATH = "./prune-bert-base-chinese"
n_class = 2
maxlen = 15


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

        encoded_pair = self.tokenizer(
            sent,
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
        model_config = BertConfig.from_pretrained(MODEL_PATH)  # 加载模型超参数
        # model_config.output_hidden_states = True
        # model_config.output_attentions = True
        model_config.return_dict = True
        self.bert = BertModel.from_pretrained(MODEL_PATH, config=model_config)  # 加载模型
        self.linear = nn.Linear(model_config.hidden_size, n_class)  # 直接用cls向量接全连接层分类
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        # print(input_ids, attention_mask, token_type_ids)
        
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        logits = self.linear(self.dropout(outputs.pooler_output))  # [bs, hidden_size]
        
        # print(outputs.keys()) # ['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions']
        """
        last_hidden_state:
            (batch_size, sequence_length, hidden_size)
            最后一层输出的各token的隐藏状态
            通常用于命名实体识别
            cls_vectors = last_hidden_state[:, 0, :]
        pooler_output:
            (batch_size, hidden_size)
            将[CLS]这个token做NSP任务（再过一下全连接层+Tanh激活函数），作为句子向量
            通常用于句子分类
            对hidden_states做平均池化可能比 pooler_output 更好地代表句子
        hidden_states:
            可选的输出, tuple, 13个元素
            第1个元素是embedding(即输入的embedding层，同self.bert.embeddings(input_ids))
            其余元素是各层的输出, 最后一个元素同 last_hidden_state
        attentions:
            可选的输出, tuple, 12个元素
            每个元素: (batch_size, num_heads, sequence_length, sequence_length)
            每一层的注意力取softmax之后的权重，用于计算self-attention heads的加权平均值
        """

        return logits


class Trainer():
    def __init__(self) -> None:
        pass


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler):
    total_step = len(dataloader)  # = 总样本量 / batch_size
    progress_bar = tqdm(range(total_step))
    progress_bar.set_description(f'loss: {0:>7f}')
    
    model.train()  # 启用train模式
    total_loss = 0
    for step, batch in enumerate(dataloader, start=1):
        batch = tuple(p.to(device) for p in batch)
        pred = model([batch[0], batch[1], batch[2]])
        loss = loss_fn(pred, batch[3])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {loss.item():>7f}, step:{step/total_step}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()  # 启用predict模式
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch = tuple(p.to(device) for p in batch)
            pred = model([batch[0], batch[1], batch[2]])
            correct += (pred.argmax(1) == batch[3]).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct


if __name__ == '__main__':  # 这个地方可以解决多线程的问题
    print(f'Using {device} device')
    # 构造训练/验证数据
    sentences = ["我喜欢打篮球", "这个相机很好看", "今天玩的特别开心", "我不喜欢你", "太糟糕了", "真是件令人伤心的事情", "我不喜欢打篮球", "这个相机很难看", "今天玩的特别伤心", "我喜欢你", "太棒了", "真是件令人高兴的事情"]
    labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  # 1积极, 0消极
    
    full_dataset = MyDataset(sentences, labels)
    
    train_size = int(0.9 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = Data.random_split(full_dataset, [train_size, valid_size])  # 划分训练集和测试集
    
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_dataloader = Data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    train_features = next(iter(train_dataloader))
    # print("查看数据:", train_features)
    
    # 设置模型超参数
    bc_model = BertClassify().to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(bc_model.parameters(), lr=learning_rate, weight_decay=1e-2)
    optimizer = AdamW(bc_model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader),
    )

    # 训练模型
    train_curve = []
    best_acc = 0
    for epoch in range(epoch_num):
        print(f"Epoch {epoch+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(train_dataloader, bc_model, loss_fn, optimizer, lr_scheduler)
        train_curve.append(total_loss)
        valid_acc = test_loop(valid_dataloader, bc_model, mode='Valid')
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('saving new weights...\n')
            torch.save(bc_model.state_dict(), f'epoch_{epoch+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
    print("Done!")
    # pd.DataFrame(train_curve).plot() # loss曲线
    # plt.show()

    # test
    bc_model = BertClassify().to(device)
    bc_model.load_state_dict(torch.load('epoch_1_valid_acc_50.0_model_weights.bin'))
    bc_model.eval()
    with torch.no_grad():
        test_text = ['我不喜欢打篮球']
        test = MyDataset(test_text, labels=None, with_labels=False)
        x = test.__getitem__(0)
        x = tuple(p.unsqueeze(0).to(device) for p in x)  # 增加一维，因为是要按batch输入模型
        pred = bc_model([x[0], x[1], x[2]])
        pred = pred.argmax(1)
        if pred.cpu()[0] == 0:
            print('消极')
        else:
            print('积极')
