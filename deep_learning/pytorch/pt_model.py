'''
Author: yyh owyangyahe@126.com
Date: 2023-09-20 11:40:14
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-07-05 17:19:44
FilePath: /mypython/yyh-util/deepLearning/pytorch/pt_model.py
Description: 
'''
import torch
import torch.nn as nn


device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')


# 定义 MLP 网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义 CNN 网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 定义 RNN 网络
class RNN(nn.Module):
    def __init__(self, input_size=28, hidden_size=512, num_layers=2, num_classes=10):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 初始隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 RNN
        out, _ = self.rnn(x, h0)
        
        # 解码最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 定义 双向RNN 网络
class BiRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiRNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向 RNN

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)  # [batch_size, 28, 28]
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(device)  # 2 for bidirectional
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 定义双向 LSTM 模型
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向 LSTM

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)  # [batch_size, 28, 28]
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(device)  # 2 for bidirectional
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 定义双向 GRU 模型
class BiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向 GRU

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)  # [batch_size, 28, 28]
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.hidden_dim).to(device)  # 2 for bidirectional
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out


# 定义 Transformer 网络
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.position_encoding = nn.Parameter(torch.zeros(1, 28*28, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        # 将输入展平为二维张量
        x = x.view(x.size(0), -1, 28*28)  # [batch_size, input_dim, 28*28]
        x = self.embedding(x) + self.position_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # 平均池化
        x = self.fc(x)
        return x
