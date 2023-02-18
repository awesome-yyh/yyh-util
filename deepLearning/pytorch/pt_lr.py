# 模型演示-线性回归, 即不加激活函数的全连接层
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 自动随机切分训练数据和测试数据
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim


print("------模型演示-线性回归-------")
# 设置环境
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else 'cpu') # mac m1 gpu: mps
print("CPU or GPU: ", device)

# 设置训练参数和模型参数
batch_size = 2
epoches = 10

# 构造样本格式
class MyDataset(Data.Dataset):
    def __init__(self, inputs, labels=None, with_labels=True,):
        self.inputs = inputs
        self.labels = labels
        self.with_labels = with_labels
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_i = self.inputs[index]
        
        if self.with_labels:
            label = self.labels[index]
            return input_i, label
        else:
            return input_i

# 搭建模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1) # 包括两个可训练参数: weight和bias

    def forward(self, X): # 直接调用对象时, 会自动将传入的参数传到forward函数当中进行计算
        logits = self.linear(X)

        return logits


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.FloatTensor)
    # 读取数据
    # y = 55*x - 33
    xs = np.array([i for i in range(10)], dtype=np.float32)
    ys = np.array([55*i-33+np.random.rand(1) for i in xs], dtype=np.float32)
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    # # 探索分析
    # plt.plot(xs, ys, 'ro', label='Original data')
    # plt.show()

    # 数据清洗(缺失值、重复值、异常值、大小写、标点)

    # 数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)

    # 特征工程(数值、类别、时间、文本、图像)

    # 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
    train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.2, random_state=1)
    # print(train_x, test_x, train_y, test_y)

    train_data = Data.DataLoader(dataset=MyDataset(train_x, train_y),
                                    batch_size=batch_size, shuffle=True, num_workers=1) # num_workers: 读取数据的进程数
    # 模式数据格式: [(data1, label1), (data2, label2), (data3, label3), ......]
    # 如果需要别的格式, 需要在DataLoader中设置collate_fn
    
    lr_model = LinearRegression().to(device)
    optimizer = optim.SGD(lr_model.parameters(), lr=1e-3, weight_decay=1e-2) # 优化器对象创建时需要传入参数，这里的参数取得是模型对象当中的w和bias
    loss_fn = nn.MSELoss()

    # 查看模型结构
    print(lr_model)
    print(list(lr_model.parameters()))

    # 训练模型
    train_curve = []
    total_step = len(train_data) # = 总样本量 / batch_size
    for epoch in range(epoches):
        sum_loss = 0
        for i, batch in enumerate(train_data):
            # 前向传播
            batch = tuple(p.to(device) for p in batch)
            pred = lr_model(batch[0])
            loss = loss_fn(pred, batch[1])
            sum_loss += loss.item()

            # 反向传播
            optimizer.zero_grad() # 每一次迭代梯度归零
            loss.backward()
            optimizer.step() # 权重更新
            # if epoch % 10 == 0:
            print(f'epoch:[{epoch+1}|{epoches}] step:{i+1}/{total_step} loss:{loss.item():.4f}')
        train_curve.append(sum_loss)
    
    # 经过迭代后输出权重和偏置
    print("Weight=", lr_model.linear.weight.item())
    print("Bias=", lr_model.linear.bias.item())
    
    # 模型的保存、加载和预测
    torch.save(lr_model.state_dict(), 'pt_model.ckpt')
    lr_model.load_state_dict(torch.load('pt_model.ckpt'))

    pd.DataFrame(train_curve).plot() # loss曲线
    plt.show()
    
    # Plot the graph
    predicted = lr_model(torch.from_numpy(train_x).to(device)).cpu().detach().numpy()
    plt.plot(train_x, train_y, 'ro', label='Original data')
    plt.plot(train_x, predicted, label='Fitted line')
    plt.legend()
    plt.show()
