"""
模型演示-线性回归, 即不加激活函数的全连接层
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


class Trainer():
    def __init__(self, model, criterion, optimizer, dataloader, epochs, device, use_cuda=False, seed=42):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.use_cuda = use_cuda
        self.epochs = epochs
        self.device = device

        if self.use_cuda:
            self.model = self.model.to(device)
        self.seed = seed

    def _fix_seed(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    def run(self):
        self._fix_seed()
        train_curve = []
        for i in range(self.epochs):
            sum_loss = self.train()
            print(f'epoch:[{i+1}|{self.epochs}] sum_loss:{sum_loss:.4f}')
            train_curve.append(sum_loss)
        return train_curve

    def train(self):
        self.model.train()
        sum_loss = 0
        for i, batch in enumerate(self.dataloader):
            # 输入数据
            if self.use_cuda:
                batch = tuple(p.to(self.device) for p in batch)

            # 每一次迭代先梯度归零
            self.optimizer.zero_grad()

            # forward + backward 
            pred = self.model(batch[0])
            loss = self.criterion(pred, batch[1])
            loss.backward()

            # 更新参数
            self.optimizer.step()

            sum_loss += loss.item()

        return sum_loss

    def evaluation(self):
        self.model.eval()  # 主要是控制batchnorm 和 dropout 层
        with torch.no_grad():  # 或者@torch.no_grad() 被他们包裹的代码块不需要计算梯度， 也不需要反向传播, 能节省显存和加速
            eval_loss = 0
            eval_acc = 0
            for i, batch in enumerate(self.dataloader):
                if self.use_cuda:
                    batch = tuple(p.to(self.device) for p in batch)
                pred = self.model(batch[0])
                loss = self.criterion(pred, batch[1])
                eval_loss += loss.item()
                prediction = torch.max(pred, 1)[1]
                pred_correct = (prediction == batch[1]).sum()
                eval_acc += pred_correct.item()
            print('evaluation loss : {:.6f}, acc : {:.6f}'.format(eval_loss / len(self.dataloader), eval_acc / len(self.dataloader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=2)
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=10)
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-3)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    print("CPU or GPU: ", device)

    # 读取数据
    random.seed(42)
    # y = 5*x - 3
    xs = np.array([i for i in range(10)], dtype=np.float32)
    ys = np.array([5 * i - 3 + 10 * np.random.rand(1) for i in xs], dtype=np.float32)
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    # # 探索分析
    # plt.plot(xs, ys, 'ro', label='Original data')
    # plt.show()

    full_dataset = MyDataset(xs, ys)

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = Data.random_split(full_dataset, [train_size, test_size])  # 划分训练集和测试集

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1) 
    # num_workers: 读取数据的进程数
    # 默认数据格式: [(data1, label1), (data2, label2), ...]
    # 在DataLoader中设置collate_fn, 可以设置数据格式，也可以进行数据清洗(缺失值、重复值、异常值)
    # 在DataLoader中设置sampler, 可以进行数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)
    # 如果训练数据集有1000个样本，并且batch_size的大小为10，则dataloader的长度就是100
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1) 

    lr_model = Linear(1, 1)
    print("查看模型结构: ")
    print(lr_model)
    print("Total Parameters:", sum([p.nelement() for p in lr_model.parameters()]))
    print("模型的可训练参数: ")
    for name, parameters in lr_model.named_parameters():
        print(name, ':', parameters.size())
    print(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in lr_model.parameters()),
            sum(p.numel() for p in lr_model.parameters() if p.requires_grad),
        )
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(lr_model.parameters(), lr=args.lr, weight_decay=1e-2)  # 优化器对象创建时需要传入参数，这里的参数取得是模型对象当中的w和bias
    # 损失函数和优化器不需要专门传到gpu上，因为已包含模型的parameters，就在gpu上

    # 训练模型
    trainer = Trainer(lr_model, criterion, optimizer, train_loader, args.n_epoch, device, use_cuda=True)
    train_curve = trainer.run()

    # 经过迭代后输出权重和偏置
    print("Weight=", lr_model.linear.weight.item())
    print("Bias=", lr_model.linear.bias.item())

    # 模型的保存、加载和预测
    torch.save(lr_model.state_dict(), 'pt_model.ckpt')
    lr_model.load_state_dict(torch.load('pt_model.ckpt'))

    pd.DataFrame(train_curve).plot()  # loss曲线
    # plt.show()
    # print(train_curve)

    tester = Trainer(lr_model, criterion, optimizer, test_loader, args.n_epoch, device, use_cuda=True)
    tester.evaluation()

    # Plot the graph
    # predicted = lr_model(test_loader.to(device)).cpu().detach().numpy()
    # plt.plot(train_x, train_y, 'ro', label='Original data')
    # plt.plot(train_x, predicted, label='Fitted line')
    # plt.legend()
    # plt.show()
